"""
Forex Telegram Alert Bot  (SLK / MSNR / CRT model)
===================================================
Data   : FOREXCOM feed from TradingView (unofficial tvdatafeed library)
Alerts : Telegram (same chat shows on phone and laptop)

MODEL IN ONE PARAGRAPH
1. Weekly and Daily structure set the trend (HH/HL = up, LH/LL = down).
2. A recent Weekly or Daily candle must REJECT a key level (V, A, OCL, QML)
   and close back above it (bullish) or below it (bearish).
   Sweep / CRT tags are added when the same candle also swept liquidity.
3. On H4, after the rejection, price must give a confirmation:
     - BREAKOUT       : H4 closes through an EXTERNAL or INTERNAL level
     - SWEEP AFTER BREAK : H4 already broke a level, then a candle sweeps the
                        previous candle and closes back inside its range
   EXTERNAL = the H4 swing high/low that formed BEFORE the rejection
   INTERNAL = the H4 swing high/low that formed AFTER the rejection
4. Alert is sent only when step 3 is present. Counter-trend setups are sent
   but labelled COUNTER-TREND.

HOW BEARISH IS HANDLED
Bearish logic is the bullish logic run on a mirrored (price * -1) chart, so
both directions use exactly the same code.

SETUP (see bottom of file for the commands)
"""
import os
import sys
import json
import time
import logging
import argparse

import numpy as np
import pandas as pd
import requests

# ============================== CONFIG ==============================
EXCHANGE = "FOREXCOM"

PAIRS = [
    # majors
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD",
    # minors / crosses
    "EURGBP", "EURJPY", "EURCHF", "EURCAD", "EURAUD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPCAD", "GBPAUD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    # gold
    "XAUUSD",
]

BARS = {"W": 120, "D": 250, "H4": 600}          # candles downloaded
SWING_N = {"W": 2, "D": 3, "H4": 2}             # fractal width for structure
H4_EXTERNAL_SWING_N = 3                         # wider swing = more meaningful external level
LEVEL_LOOKBACK = {"W": 52, "D": 90}             # how far back to look for key levels
RECENT_REJECTION = {"W": 2, "D": 3}             # rejection must be within last N closed candles
TOL_ATR = 0.15                                  # how close a wick must get to a level (x ATR)
REQUIRE_KEY_LEVEL = True                        # False = a plain sweep can also build bias
POST_BREAK_SWEEP_BARS = 12                      # H4 bars allowed between break and retest sweep
SCAN_EVERY_MIN = 15                             # scan interval in loop mode
STATE_FILE = "alert_state.json"                 # remembers alerts already sent
# ====================================================================

log = logging.getLogger("fxbot")


# ----------------------------- helpers ------------------------------
def mirror(df):
    """Flip the chart upside down so bearish = bullish logic."""
    m = pd.DataFrame(index=df.index)
    m["open"], m["high"], m["low"], m["close"] = -df["open"], -df["low"], -df["high"], -df["close"]
    return m


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def swings(df, n):
    """Confirmed fractal swings. Returns (highs, lows) as lists of (pos, price)."""
    h, l = df["high"].values, df["low"].values
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:i + n + 1].max():
            highs.append((i, h[i]))
        if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:i + n + 1].min():
            lows.append((i, l[i]))
    return highs, lows


def alt_swings(df, n):
    """Swings forced to alternate H, L, H, L (keeps the most extreme of a run)."""
    hi, lo = swings(df, n)
    seq = sorted([(p, pr, "H") for p, pr in hi] + [(p, pr, "L") for p, pr in lo])
    out = []
    for s in seq:
        if out and out[-1][2] == s[2]:
            if (s[2] == "H" and s[1] > out[-1][1]) or (s[2] == "L" and s[1] < out[-1][1]):
                out[-1] = s
        else:
            out.append(s)
    return out


def trend(df, n):
    """Structure trend from the last two swing highs and lows."""
    hi, lo = swings(df, n)
    if len(hi) < 2 or len(lo) < 2:
        return "range"
    up = hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]
    dn = hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]
    return "up" if up else "down" if dn else "range"


def is_sweep_bar(l, h, c, k):
    """Bullish CRT-style sweep: takes previous low, closes back inside previous range."""
    return l[k] < l[k - 1] and l[k - 1] < c[k] < h[k - 1]


# --------------------------- key levels -----------------------------
def bullish_levels(df, n, lookback):
    """
    Levels that can act as SUPPORT (bullish case):
      V   : bullish engulfing, level = close of the engulfed bearish candle
      OCL : two consecutive up candles, level = line between them
      QML : bullish Quasimodo. Swings L1, H1, L2 (L2 < L1), then a close above H1.
            Level = L1
    """
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    levels = []
    for i in range(max(1, len(df) - lookback), len(df)):
        if c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] >= o[i - 1] and o[i] <= c[i - 1]:
            levels.append(dict(kind="V", price=c[i - 1], pos=i))
        if c[i - 1] > o[i - 1] and c[i] > o[i]:
            levels.append(dict(kind="OCL", price=(c[i - 1] + o[i]) / 2, pos=i))

    seq = alt_swings(df, n)
    for j in range(len(seq) - 2):
        a, b, d = seq[j], seq[j + 1], seq[j + 2]
        if a[2] == "L" and b[2] == "H" and d[2] == "L" and d[1] < a[1] and a[0] >= len(df) - lookback:
            after = c[d[0] + 1:] > b[1]
            if after.any():
                brk = d[0] + 1 + int(np.argmax(after))
                levels.append(dict(kind="QML", price=a[1], pos=brk))

    # QMC: NOT ENABLED. Waiting for the exact definition from the trader.
    return levels


def find_rejections(df, tf, n):
    """All recent candles (newest first) that rejected a bullish key level and closed above it."""
    levels = bullish_levels(df, n, LEVEL_LOOKBACK[tf])
    a = atr(df).values
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    last = len(df) - 1
    found = []
    for i in range(last, last - RECENT_REJECTION[tf], -1):
        if i < 12 or np.isnan(a[i]):
            continue
        tol = TOL_ATR * a[i]
        hits = [x for x in levels if x["pos"] < i and l[i] <= x["price"] + tol and c[i] > x["price"]]
        crt = l[i] < l[i - 1] and l[i - 1] < c[i] < h[i - 1]
        prior_low = l[i - 10:i].min()
        sweep = l[i] < prior_low and c[i] > prior_low
        if not (hits or (not REQUIRE_KEY_LEVEL and (crt or sweep))):
            continue
        if i < last and (c[i + 1:] < l[i]).any():      # later candle closed under the rejection low
            continue
        best = min(hits, key=lambda x: abs(l[i] - x["price"])) if hits else None
        found.append(dict(tf=tf, pos=i, time=df.index[i], low=l[i], level=best, crt=crt, sweep=sweep))
    return found


# ------------------------- H4 confirmation --------------------------
def h4_confirm(h4, start, end, rej_low):
    """
    Bullish H4 confirmation after a higher timeframe rejection that lived in [start, end).
    Returns a dict describing the trigger, or None.
    """
    idx = h4.index
    pad = pd.Timedelta(hours=12)                        # pad the window so a timezone offset cannot hide the extreme
    win = np.where((idx >= start - pad) & (idx < end + pad))[0]
    if len(win) == 0:
        return None
    h, l, c = (h4[x].values for x in ("high", "low", "close"))
    last = len(h4) - 1
    anchor = int(win[np.argmin(np.abs(l[win] - rej_low))])   # H4 bar holding the swept extreme
    if abs(l[anchor] - rej_low) > np.median(h[-100:] - l[-100:]):
        return None                                     # cannot match the HTF low, do not guess
    if anchor >= last - 1:
        return None
    if l[anchor + 1:].min() < l[anchor]:                # new low made = setup invalid
        return None

    ext_hi = [p for p in swings(h4, H4_EXTERNAL_SWING_N)[0] if p[0] < anchor]
    int_hi = [p for p in swings(h4, SWING_N["H4"])[0] if p[0] > anchor]
    ext = ext_hi[-1] if ext_hi else None
    intl = int_hi[-1] if int_hi else None
    if ext and intl and intl[1] >= ext[1]:
        intl = None
    cands = [(name, s) for name, s in (("EXTERNAL", ext), ("INTERNAL", intl)) if s]
    if not cands:
        return None

    base = dict(anchor_low=l[anchor], anchor_time=idx[anchor], bar=idx[last], close=c[last])

    # Trigger 1: fresh H4 close through the level (external checked first)
    for name, (pos, price) in cands:
        if c[last] > price and c[last - 1] <= price:
            sweep_before = any(is_sweep_bar(l, h, c, k) for k in range(anchor + 1, last))
            return dict(base, trigger="BREAKOUT", btype=name, level=price, sweep_before=sweep_before)

    # Trigger 2: level already broken, now an H4 sweep candle holds above it
    for name, (pos, price) in cands:
        broke = [k for k in range(max(pos + 1, anchor + 1), last) if c[k] > price]
        if broke and last - broke[0] <= POST_BREAK_SWEEP_BARS \
                and is_sweep_bar(l, h, c, last) and c[last] > price:
            return dict(base, trigger="SWEEP AFTER BREAK", btype=name, level=price, sweep_before=True)
    return None


# ---------------------------- scanning ------------------------------
def trend_label(td, tw, want):
    opp = "down" if want == "up" else "up"
    if td == want and tw != opp:
        return "WITH-TREND"
    if td == opp and tw == opp:
        return "COUNTER-TREND"
    return "MIXED TREND"


def decimals(sym):
    return 2 if sym.startswith("XAU") else 3 if "JPY" in sym else 5


def kind_name(kind, bull):
    if kind == "V":
        return "V level" if bull else "A level"
    return f"{'Bullish' if bull else 'Bearish'} {kind}"


def scan_pair(sym, feed):
    w, d, h = feed.get(sym, "W"), feed.get(sym, "D"), feed.get(sym, "H4")
    if w is None or d is None or h is None or len(d) < 40 or len(w) < 20 or len(h) < 60:
        log.warning("%s: not enough data", sym)
        return []
    td, tw = trend(d, SWING_N["D"]), trend(w, SWING_N["W"])
    out = []
    for bull in (True, False):
        W, D, H = (w, d, h) if bull else (mirror(w), mirror(d), mirror(h))
        sgn = 1 if bull else -1
        found = []
        for tf, df, days in (("D", D, 1), ("W", W, 7)):
            for r in find_rejections(df, tf, SWING_N[tf]):
                conf = h4_confirm(H, r["time"], r["time"] + pd.Timedelta(days=days), r["low"])
                if conf:
                    found.append((r, conf))
        if not found:
            continue
        r, conf = found[0]
        tfs = sorted({x["tf"] for x, cf in found if cf["bar"] == conf["bar"]
                      and cf["trigger"] == conf["trigger"]}, key=lambda t: "DW".index(t))
        names = " + ".join({"D": "Daily", "W": "Weekly"}[t] for t in tfs)
        dp = decimals(sym)
        fmt = lambda x: f"{sgn * x:.{dp}f}"
        lvl = r["level"]
        lvl_txt = f"{kind_name(lvl['kind'], bull)} at {fmt(lvl['price'])}" if lvl else "no key level (sweep only)"
        tags = [t for t, on in (("CRT sweep", r["crt"]), ("liquidity sweep", r["sweep"])) if on]
        tag_txt = f" ({', '.join(tags)})" if tags else ""
        label = trend_label(td, tw, "up" if bull else "down")
        key = f"{sym}|{'bull' if bull else 'bear'}|{r['tf']}|{r['time']}|{conf['trigger']}|{conf['btype']}|{conf['bar']}"
        msg = (
            f"{'🟢 BULLISH' if bull else '🔴 BEARISH'} SETUP: {sym}\n"
            f"Trend: {label} (Weekly {tw}, Daily {td})\n"
            f"HTF rejection: {names}, candle {r['time']:%Y-%m-%d}\n"
            f"Rejected from: {lvl_txt}{tag_txt}\n"
            f"H4 confirmation: {conf['btype']} {conf['trigger']}"
            f"{' (H4 sweep before break)' if conf.get('sweep_before') and conf['trigger'] == 'BREAKOUT' else ''}\n"
            f"Level taken: {fmt(conf['level'])}\n"
            f"H4 close: {fmt(conf['close'])} at {conf['bar']:%Y-%m-%d %H:%M}\n"
            f"Invalidation (swept extreme): {fmt(conf['anchor_low'])}\n"
            f"Data: {EXCHANGE}. Always confirm on your chart before trading."
        )
        out.append((key, msg))
    return out


# ------------------------ data + telegram ---------------------------
class Feed:
    """TradingView data through the unofficial tvdatafeed library."""

    def __init__(self):
        from tvDatafeed import TvDatafeed, Interval
        user, pw = os.getenv("TV_USER"), os.getenv("TV_PASS")
        self.tv = TvDatafeed(user, pw) if user and pw else TvDatafeed()
        self.iv = {"W": Interval.in_weekly, "D": Interval.in_daily, "H4": Interval.in_4_hour}
        self.ttl = {"W": 4 * 3600, "D": 3600, "H4": 0}
        self.cache = {}

    def get(self, sym, tf):
        key, now = (sym, tf), time.time()
        if key in self.cache and now - self.cache[key][0] < self.ttl[tf]:
            return self.cache[key][1]
        df = None
        for _ in range(3):
            try:
                df = self.tv.get_hist(sym, EXCHANGE, self.iv[tf], n_bars=BARS[tf])
            except Exception as e:
                log.warning("%s %s fetch error: %s", sym, tf, e)
            if df is not None and len(df) > 30:
                break
            time.sleep(2)
        if df is None or len(df) < 30:
            return None
        df = df[["open", "high", "low", "close"]].astype(float).iloc[:-1]   # drop the forming candle
        self.cache[key] = (now, df)
        return df


def send_telegram(text):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[TELEGRAM NOT CONFIGURED, alert not marked as sent]\n" + text + "\n")
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=20)
    r.raise_for_status()
    return True


def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        s = {}
    cutoff = time.time() - 30 * 86400
    return {k: v for k, v in s.items() if v > cutoff}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)


def run_once(feed, state):
    for sym in PAIRS:
        try:
            alerts = scan_pair(sym, feed)
        except Exception:
            log.exception("%s scan failed", sym)
            continue
        for key, msg in alerts:
            if key in state:
                continue
            try:
                if send_telegram(msg):
                    state[key] = time.time()
                    save_state(state)
                    log.info("ALERT sent: %s", key)
            except Exception:
                log.exception("Telegram send failed")
        time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="send a test Telegram message and exit")
    ap.add_argument("--once", action="store_true", help="scan one time and exit")
    ap.add_argument("--check", action="store_true", help="test the TradingView data feed and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.test:
        send_telegram("Test message from your forex alert bot. Telegram is connected.")
        return
    feed, state = Feed(), load_state()
    if args.check:
        for sym in ("EURUSD", "XAUUSD"):
            for tf in ("W", "D", "H4"):
                df = feed.get(sym, tf)
                print(f"--- {sym} {tf}: {'NO DATA' if df is None else str(len(df)) + ' closed candles'}")
                if df is not None:
                    print(df.tail(3))
        return
    if args.once:
        run_once(feed, state)
        save_state(state)
        return
    while True:
        run_once(feed, state)
        log.info("Scan finished, sleeping %s min", SCAN_EVERY_MIN)
        time.sleep(SCAN_EVERY_MIN * 60)


if __name__ == "__main__":
    main()

# ------------------------------ SETUP -------------------------------
# 1. pip install pandas numpy requests
#    pip install --upgrade git+https://github.com/rongardF/tvdatafeed.git
# 2. Telegram: message @BotFather, send /newbot, copy the token.
#    Open your new bot and press Start, then open
#    https://api.telegram.org/bot<TOKEN>/getUpdates to read your chat id.
# 3. Set environment variables (Windows PowerShell shown):
#    $env:TELEGRAM_TOKEN="123:ABC"; $env:TELEGRAM_CHAT_ID="123456789"
#    Optional TradingView login (more reliable data): TV_USER, TV_PASS
# 4. python forex_alert_bot.py --test     (must arrive on Telegram)
#    python forex_alert_bot.py --once     (one scan)
#    python forex_alert_bot.py            (runs forever, scans every 15 min)
