"""
Forex Telegram Alert Bot  (SLK / MSNR model)   VERSION 4
=========================================================
Data   : FOREXCOM feed from TradingView (unofficial tvdatafeed library)
Alerts : Telegram, all times in Lagos time

THE MODEL, STEP BY STEP  (rejections and body-to-body breakouts only, no sweep / CRT)
1. Structure: Monthly, Weekly and Daily swing structure gives the trend label
   (WITH TREND / COUNTER TREND). It labels the trade, it never blocks it.
2. Higher timeframe evidence (at least one):
     - a Weekly rejection from a key level (the newest 2 Weekly candles, forming one included)
     - the most recent Weekly breakout (a body close through the last swing, no opposite break since)
     - the most recent Daily breakout (same rule on the Daily)
3. Daily rejection (required): today's candle (flagged if still forming), yesterday's or the
   day before's wicked clearly into a key level (V, A or OCL) and closed back on the trade side.
4. H4 breakout (required), body to body, for EVERY daily rejection above:
     EXTERNAL (Reversal)     the closest level BEFORE the rejection (never further back than 4 days);
                             a candle body closes through it
     INTERNAL (Continuation) the closest level AFTER the rejection; a candle body closes through it
   Levels are candle-body edges (open / close), never wick tips.
5. The alert is sent as soon as the breaking H4 candle has closed (checked right after every H4 close).
6. Invalidation: price trading beyond the Daily rejection extreme / the H4 rejection extreme.
7. At most MAX_ALERTS_PER_DAY alerts per Lagos day. If more qualify, external breakouts go first,
   then with-trend setups, then the rest are held back (you get one note saying so).

Bearish logic is the bullish logic run on a mirrored (price * -1) chart, so both
directions use exactly the same code.
"""
import os
import json
import time
import logging
import argparse
from zoneinfo import ZoneInfo  # noqa: F401  (needs the "tzdata" package on Windows)

import numpy as np
import pandas as pd
import requests

# ============================== CONFIG ==============================
EXCHANGE = "FOREXCOM"

PAIRS = [
    # majors (USDCHF removed)
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD",
    # minors / crosses (all XXXCHF pairs removed, CHFJPY kept)
    "EURGBP", "EURJPY", "EURCAD", "EURAUD", "EURNZD",
    "GBPJPY", "GBPCAD", "GBPAUD", "GBPNZD",
    "AUDJPY", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCAD",
    "CADJPY", "CHFJPY",
    # gold
    "XAUUSD",
]

BARS = {"M": 60, "W": 150, "D": 250, "H4": 600}   # candles downloaded
SWING_N = {"M": 2, "W": 2, "D": 3}                # swing width for the trend label
BREAK_N = {"W": 2, "D": 2}                        # swing width used to find the most recent structure break
PIVOT_N = 2                                       # H4 body-edge pivot width
LEVEL_LOOKBACK = {"W": 52, "D": 90}               # how far back key levels are searched
RECENT_EVENT = {"W": 2, "D": 3}                   # newest candles that may hold a rejection (forming candle counts)
TOL_ATR = 0.15                                    # how close a wick must get to a level (x ATR)
REJECT_WICK_FRAC = 0.35                           # a clear rejection: the wick is at least this share of the candle range
ENGULF_TOL_ATR = 0.05                             # feed noise allowed when a candle "opens at the previous close"
KIND_RANK = {"V": 0, "OCL": 1}                    # which level is named when several sit under one wick
EXT_LOOKBACK_BARS = 24                            # the level before the rejection must be within this many H4 candles (24 = 4 days), so a far-away level is never used
EXT_MIN_ATR = 0.5                                 # the level before the rejection must sit this far (x H4 ATR) beyond the rejection candle's body
PULLBACK_ATR = 0.30                               # the level after the rejection must have been pulled back from by this much (x H4 ATR)
FRESH_DAYS = 3                                    # an H4 breakout is reported if it happened within this many days
MAX_ALERTS_PER_DAY = 7                            # alerts per Lagos day (your wish: 5 to 7)
SCAN_EVERY_MIN = 15                               # only used when you run the bot in loop mode
STATE_FILE = "alert_state.json"                   # remembers what was sent and what was already scanned
DATA_TZ = "UTC"                                   # timezone of the candle timestamps the feed returns
DISPLAY_TZ = "Africa/Lagos"                       # timezone shown in your Telegram alerts

PERIOD = {"M": pd.DateOffset(months=1), "W": pd.Timedelta(days=7),
          "D": pd.Timedelta(days=1), "H4": pd.Timedelta(hours=4)}
# ====================================================================

log = logging.getLogger("fxbot")


# ----------------------------- helpers ------------------------------
def lagos(ts, add_hours=0):
    """Convert a feed timestamp to Lagos time. Only changes what you READ, never the logic."""
    t = pd.Timestamp(ts) + pd.Timedelta(hours=add_hours)
    if t.tzinfo is None:
        t = t.tz_localize(DATA_TZ)
    return t.tz_convert(DISPLAY_TZ)


def mirror(df):
    """Flip the chart upside down so bearish = bullish logic."""
    m = pd.DataFrame(index=df.index)
    m["open"], m["high"], m["low"], m["close"] = -df["open"], -df["low"], -df["high"], -df["close"]
    return m


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def swings(df, n):
    """Confirmed fractal swings on wicks. Returns (highs, lows) as lists of (pos, price)."""
    h, l = df["high"].values, df["low"].values
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:i + n + 1].max():
            highs.append((i, h[i]))
        if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:i + n + 1].min():
            lows.append((i, l[i]))
    return highs, lows


def trend(df, n):
    """Structure trend from the last two swing highs and lows."""
    if df is None or len(df) < 10:
        return "range"
    hi, lo = swings(df, n)
    if len(hi) < 2 or len(lo) < 2:
        return "range"
    up = hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]
    dn = hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]
    return "up" if up else "down" if dn else "range"


def body_top(df, pos):
    """Highest body edge (open or close) of the swing candle and its two neighbours."""
    o, c = df["open"].values, df["close"].values
    a, b = max(pos - 1, 0), min(pos + 2, len(df))
    return float(np.maximum(o[a:b], c[a:b]).max())


def body_pivots(df, n):
    """
    Pivot highs of the candle BODY edge (the larger of open and close).
    A flat run of equal body edges counts once, so a flat base is found as one level.
    """
    bh = np.maximum(df["open"].values, df["close"].values)
    return [i for i in range(n, len(df) - n)
            if bh[i] > bh[i - n:i].max() and bh[i] >= bh[i + 1:i + n + 1].max()]


# --------------------------- key levels -----------------------------
def bullish_levels(df, lookback):
    """
    Levels that can act as SUPPORT (bullish case):
      V   : bullish engulfing, level = close of the engulfed bearish candle
      OCL : two consecutive up candles, level = line between them
    (The bearish side is the same code on a mirrored chart: V becomes the A level.)
    """
    o, c = df["open"].values, df["close"].values
    a = atr(df).values
    levels = []
    for i in range(max(1, len(df) - lookback), len(df)):
        eps = ENGULF_TOL_ATR * a[i] if not np.isnan(a[i]) else 0.0
        if c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] >= o[i - 1] - eps and o[i] <= c[i - 1] + eps:
            levels.append(dict(kind="V", price=c[i - 1], pos=i))
        if c[i - 1] > o[i - 1] and c[i] > o[i]:
            levels.append(dict(kind="OCL", price=(c[i - 1] + o[i]) / 2, pos=i))
    return levels


def find_events(all_df, closed_df, tf):
    """
    Clear rejections (newest first): the wick reached a bullish key level, the body stayed above
    it, and the candle closed above it. The wick must be a real share of the candle range.
    `all_df` may end with a still-forming candle, which is then flagged.
    Invalid once a later candle trades beyond the rejection low.
    """
    if closed_df is None or len(closed_df) < 20:
        return []
    levels = bullish_levels(closed_df, LEVEL_LOOKBACK[tf])
    a = atr(closed_df).values
    o, h, l, c = (all_df[x].values for x in ("open", "high", "low", "close"))
    last = len(all_df) - 1
    forming_last = len(all_df) > len(closed_df)
    out = []
    for i in range(last, last - RECENT_EVENT[tf], -1):
        if i < 12:
            continue
        av = a[min(i, len(a) - 1)]
        rng = h[i] - l[i]
        if np.isnan(av) or rng <= 0:
            continue
        body_low = min(o[i], c[i])
        if (body_low - l[i]) < REJECT_WICK_FRAC * rng:           # not a clear rejection wick
            continue
        tol = TOL_ATR * av
        hits = [x for x in levels if x["pos"] < i and l[i] <= x["price"] + tol
                and body_low >= x["price"] - tol and c[i] > x["price"]]
        if not hits:
            continue
        if i < last and (l[i + 1:] < l[i]).any():
            continue
        best = min(hits, key=lambda x: (KIND_RANK[x["kind"]], abs(l[i] - x["price"])))
        out.append(dict(tf=tf, pos=i, time=all_df.index[i], low=l[i], level=best,
                        forming=bool(i == last and forming_last)))
    return out


# ----------------------- higher timeframe breaks --------------------
def latest_break(df, n):
    """
    The most recent body close through a confirmed swing high (bullish frame):
    returns (bar index, level) or None. Levels are body edges.
    """
    c = df["close"].values
    best = None
    for pos, _ in swings(df, n)[0]:
        p = body_top(df, pos)
        for k in range(pos + 1, len(df)):
            if c[k] > p:
                if best is None or k > best[0]:
                    best = (k, p)
                break
    return best


def recent_break(df, n):
    """Most recent structure break on this timeframe, if it points in the trade direction."""
    kb, ko = latest_break(df, n), latest_break(mirror(df), n)
    if kb and (ko is None or kb[0] > ko[0]):
        return dict(price=kb[1], time=df.index[kb[0]])
    return None


# ------------------------ H4 structure levels -----------------------
def structure_breaks(df, start, end, ref_low, pad):
    """
    H4 levels around ONE daily rejection whose extreme is `ref_low` and which lived in [start, end).
    Bullish frame, body-edge levels, broken by the FIRST candle that CLOSES through them:
      EXTERNAL : the closest pivot BEFORE the rejection (clearly above the rejection candle's body)
      INTERNAL : the closest pivot AFTER the rejection that price pulled back from
    """
    idx = df.index
    win = np.where((idx >= start - pad) & (idx < end + pad))[0]
    if len(win) == 0:
        return None
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    last = len(df) - 1
    anchor = int(win[np.argmin(np.abs(l[win] - ref_low))])      # bar holding the rejection extreme
    if abs(l[anchor] - ref_low) > np.median(h[-100:] - l[-100:]):
        return None                                             # cannot match it, do not guess
    if anchor >= last:
        return None
    if l[anchor + 1:].min() < l[anchor]:                        # traded beyond the extreme = invalid
        return None
    bh = np.maximum(o, c)
    av = atr(df).values
    a_here = av[anchor] if not np.isnan(av[anchor]) else float(np.median(h[-100:] - l[-100:]))

    def first_break(pos, price):
        for k in range(max(pos, anchor) + 1, last + 1):
            if c[k] > price:
                return k
        return None

    levels = []
    piv = body_pivots(df, PIVOT_N)
    ref_body = bh[max(anchor - 1, 0):anchor + 2].max()
    for i in reversed([p for p in piv if p < anchor and anchor - p <= EXT_LOOKBACK_BARS]):   # nearest first, never far away
        if bh[i] >= ref_body + EXT_MIN_ATR * a_here:
            levels.append(dict(kind="EXTERNAL", price=bh[i], pos=i, brk=first_break(i, bh[i])))
            break
    for i in [p for p in piv if p > anchor]:                    # earliest after the rejection first
        if bh[i] - l[i + 1:i + 4].min() >= PULLBACK_ATR * a_here:
            levels.append(dict(kind="INTERNAL", price=bh[i], pos=i, brk=first_break(i, bh[i])))
            break
    return dict(anchor=anchor, anchor_low=l[anchor], anchor_time=idx[anchor], levels=levels, last=last)


# ---------------------------- scanning ------------------------------
def trend_label(trends, want):
    opp = "down" if want == "up" else "up"
    vals = [trends["M"], trends["W"], trends["D"]]
    agree, oppose = sum(v == want for v in vals), sum(v == opp for v in vals)
    if agree > oppose:
        return "WITH TREND"
    return "COUNTER TREND" if oppose > 0 else "NO CLEAR TREND (range)"


def decimals(sym):
    return 2 if sym.startswith("XAU") else 3 if "JPY" in sym else 5


def kind_name(kind, bull):
    if kind == "V":
        return "V level" if bull else "A level"
    return f"{'Bullish' if bull else 'Bearish'} {kind}"


def orient(dd, bull):
    if dd is None or bull:
        return dd
    return dict(closed=mirror(dd["closed"]), all=mirror(dd["all"]), forming=dd["forming"])


def build_message(sym, bull, trends, d, w_events, w_brk, d_brk, st, lv, h4, also=()):
    sgn = 1 if bull else -1
    dp = decimals(sym)
    want = "up" if bull else "down"
    fmt = lambda x: f"{sgn * x:.{dp}f}"
    kn = lambda lvl: f"{kind_name(lvl['kind'], bull)} {fmt(lvl['price'])}"
    is_ext = lv["kind"] == "EXTERNAL"
    setup = "EXTERNAL (Reversal)" if is_ext else "INTERNAL (Continuation)"
    above = "above" if bull else "below"
    still = " (candle still forming)"

    bar = h4.index[lv["brk"]]
    closed = lagos(bar, 4)
    now = pd.Timestamp.now(tz="UTC")
    late = int((now - closed.tz_convert("UTC")).total_seconds() // 60)
    dday = lagos(d["time"], 12).strftime("%a %d %b")

    parts = []
    if w_events:
        parts.append(f"rejection from {kn(w_events[0]['level'])}" + (still if w_events[0]["forming"] else ""))
    if w_brk:
        parts.append(f"latest breakout, body closed {above} {fmt(w_brk['price'])} ({lagos(w_brk['time'], 12):%d %b})")
    weekly = ("✔ " + " + ".join(parts)) if parts else "– no Weekly signal"
    daily = f"✔ rejection from {kn(d['level'])} ({dday})" + (still if d["forming"] else "")
    if d_brk:
        daily += f" + latest breakout, body closed {above} {fmt(d_brk['price'])} ({lagos(d_brk['time'], 12):%d %b})"

    also_line = ""
    if also:
        also_line = "Also broken on this candle: " + "; ".join(f"{x['kind']} {fmt(x['price'])}" for x in also) + "\n"
    newer = sorted([x for x in st["levels"] if x["brk"] is not None and x["brk"] > lv["brk"]], key=lambda x: x["brk"])
    if newer:
        newer_line = "Newer break for this rejection: " + "; ".join(
            f"{x['kind']} {fmt(x['price'])} ({lagos(h4.index[x['brk']], 4):%a %d %b %H:%M})" for x in newer) + "\n"
    else:
        newer_line = "This is the latest break for this rejection\n"
    forming_note = ""
    if d["forming"] or (w_events and w_events[0]["forming"]):
        forming_note = "Note: a Daily/Weekly candle is still forming, so this can change before it closes.\n"

    return (
        f"{'🟢 BULLISH' if bull else '🔴 BEARISH'} | {sym} · W→D→H4\n"
        f"H4 BO: {setup} @ {fmt(lv['price'])} · {closed:%a %d %b, %H:%M}\n"
        f"Rejected on: {dday} (Daily), wick extreme {fmt(d['low'])}\n"
        f"Rejected from: {kn(d['level'])}\n"
        f"Trend: {trend_label(trends, want)} (Monthly {trends['M']}, Weekly {trends['W']}, Daily {trends['D']})\n"
        f"Alignment:\n"
        f"  Weekly: {weekly}\n"
        f"  Daily: {daily}\n"
        f"  H4: ✔ {lv['kind'].lower()} breakout, body closed {fmt(h4['close'].values[lv['brk']])} {above} {fmt(lv['price'])}\n"
        f"{also_line}"
        f"{newer_line}"
        f"Invalidation: Daily {fmt(d['low'])} | H4 {fmt(st['anchor_low'])}\n"
        f"{forming_note}"
        f"Sent {lagos(now):%a %d %b %H:%M} Lagos ({late} min after the H4 close). Data: {EXCHANGE}. Confirm on your chart before trading."
    )


def scan_pair(sym, feed):
    """Returns a list of alert dicts (maybe empty), or None when the data was missing."""
    data = {tf: feed.get(sym, tf) for tf in ("M", "W", "D", "H4")}
    if any(data[t] is None for t in ("W", "D", "H4")):
        log.warning("%s: missing data", sym)
        return None
    if len(data["D"]["closed"]) < 40 or len(data["W"]["closed"]) < 30 or len(data["H4"]["closed"]) < 100:
        log.warning("%s: not enough data", sym)
        return None
    has_m = data["M"] is not None and len(data["M"]["closed"]) >= 24
    trends = {"M": trend(data["M"]["closed"], SWING_N["M"]) if has_m else "range",
              "W": trend(data["W"]["closed"], SWING_N["W"]),
              "D": trend(data["D"]["closed"], SWING_N["D"])}
    out = []
    for bull in (True, False):
        W, D, H = orient(data["W"], bull), orient(data["D"], bull), orient(data["H4"], bull)
        h4 = H["closed"]
        d_events = find_events(D["all"], D["closed"], "D")        # today (if forming), yesterday, 2 days ago
        if not d_events:
            continue
        w_events = find_events(W["all"], W["closed"], "W")
        w_brk = recent_break(W["closed"], BREAK_N["W"])
        d_brk = recent_break(D["closed"], BREAK_N["D"])
        if not (w_events or w_brk or d_brk):                      # no higher timeframe evidence
            continue
        seen = set()
        for d in d_events:                                        # newest rejection first: it decides the label
            st = structure_breaks(h4, d["time"], d["time"] + PERIOD["D"], d["low"], pd.Timedelta(hours=12))
            if not st:
                continue
            cutoff = h4.index[st["last"]] - pd.Timedelta(days=FRESH_DAYS)
            groups = {}
            for lv in st["levels"]:                               # group the levels by the candle that broke them
                if lv["brk"] is not None and h4.index[lv["brk"]] >= cutoff:
                    groups.setdefault(lv["brk"], []).append(lv)
            for brk, lvs in groups.items():
                ext = [x for x in lvs if x["kind"] == "EXTERNAL"]
                lv = ext[0] if ext else lvs[0]
                also = [x for x in lvs if x is not lv]
                bar = h4.index[brk]
                once = (lv["kind"], round(lv["price"], 6), str(bar))   # same level and candle: newest rejection decides
                key = "|".join([sym, "bull" if bull else "bear", str(d["time"]), lv["kind"]])
                if key in seen or once in seen:
                    continue
                seen.add(key)
                seen.add(once)
                msg = build_message(sym, bull, trends, d, w_events, w_brk, d_brk, st, lv, h4, also)
                out.append(dict(key=key, msg=msg, bar=bar, sym=sym, ext=lv["kind"] == "EXTERNAL",
                                with_trend=trend_label(trends, "up" if bull else "down") == "WITH TREND"))
    return out


# ------------------------ data + telegram ---------------------------
def split_forming(df, tf, now=None):
    """Separate a candle that is still printing from the closed candles."""
    now = now or pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(df.index[-1])
    if start.tzinfo is None:
        start = start.tz_localize(DATA_TZ)
    forming = bool(start + PERIOD[tf] > now)
    return dict(closed=df.iloc[:-1] if forming else df, all=df, forming=forming)


class Feed:
    """TradingView data through the unofficial tvdatafeed library."""

    def __init__(self):
        from tvDatafeed import TvDatafeed, Interval
        user, pw = os.getenv("TV_USER"), os.getenv("TV_PASS")
        self.tv = TvDatafeed(user, pw) if user and pw else TvDatafeed()
        self.iv = {"M": Interval.in_monthly, "W": Interval.in_weekly,
                   "D": Interval.in_daily, "H4": Interval.in_4_hour}
        self.ttl = {"M": 3600, "W": 1800, "D": 1800, "H4": 120}
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
            if df is not None and len(df) > 20:
                break
            time.sleep(2)
        if df is None or len(df) < 20:
            return None
        out = split_forming(df[["open", "high", "low", "close"]].astype(float), tf)
        self.cache[key] = (now, out)
        return out


def send_telegram(text):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[TELEGRAM NOT CONFIGURED, alert not marked as sent]\n" + text + "\n")
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=20)
    r.raise_for_status()
    return True


def new_state():
    return {"seen": {}, "bars": {}, "daily": {}, "notes": {}}


def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        s = {}
    if "seen" not in s:                                           # older file: a flat {key: timestamp}
        s = {"seen": {k: v for k, v in s.items() if isinstance(v, (int, float))}}
    for k, v in new_state().items():
        s.setdefault(k, v)
    cutoff = time.time() - 30 * 86400
    s["seen"] = {k: v for k, v in s["seen"].items() if v > cutoff}
    return s


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)


def run_once(feed, state, send=True, force=False):
    """
    One pass over all symbols. A symbol is only scanned when it has a NEW closed H4 candle
    (so checking often is cheap). Qualifying alerts are ranked, capped per Lagos day, and sent
    oldest to newest.
    """
    cands, scanned = [], {}
    for sym in PAIRS:
        try:
            h4 = feed.get(sym, "H4")
            if h4 is None:
                continue
            bar = str(h4["closed"].index[-1])
            if not force and state["bars"].get(sym) == bar:
                continue
            res = scan_pair(sym, feed)
            if res is None:
                continue
            scanned[sym] = bar
            cands += [r for r in res if r["key"] not in state["seen"]]
        except Exception:
            log.exception("%s scan failed", sym)
        time.sleep(0.2)

    today = lagos(pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%d")
    room = MAX_ALERTS_PER_DAY - state["daily"].get(today, 0)
    cands.sort(key=lambda r: (not r["ext"], not r["with_trend"], -r["bar"].value))   # external, with-trend, newest first
    chosen, held = cands[:max(room, 0)], cands[max(room, 0):]
    failed = set()
    for r in sorted(chosen, key=lambda r: r["bar"]):              # send oldest to newest
        if not send:
            state["seen"][r["key"]] = time.time()
            continue
        try:
            if send_telegram(r["msg"]):
                state["seen"][r["key"]] = time.time()
                state["daily"][today] = state["daily"].get(today, 0) + 1
                log.info("ALERT sent: %s", r["key"])
            else:
                failed.add(r["sym"])
        except Exception:
            log.exception("Telegram send failed")
            failed.add(r["sym"])
    for r in held:                                                # over the daily limit: mark as seen, do not resend
        state["seen"][r["key"]] = time.time()
    if held and send and not state["notes"].get(today):
        try:
            if send_telegram(f"Daily limit of {MAX_ALERTS_PER_DAY} alerts reached. {len(held)} more setup(s) were held back "
                             f"({', '.join(sorted({r['sym'] for r in held}))}). Raise MAX_ALERTS_PER_DAY in the bot to see more."):
                state["notes"][today] = True
        except Exception:
            log.exception("Telegram note failed")
    for sym, bar in scanned.items():                              # only remember a scan that finished cleanly
        if sym not in failed:
            state["bars"][sym] = bar
    for k in ("daily", "notes"):
        state[k] = dict(sorted(state[k].items())[-14:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="send a test Telegram message and exit")
    ap.add_argument("--once", action="store_true", help="scan one time and exit")
    ap.add_argument("--check", action="store_true", help="test the TradingView data feed and exit")
    ap.add_argument("--baseline", action="store_true",
                    help="scan once and mark everything found as already seen, without sending")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.test:
        send_telegram("Test message from your forex alert bot. Telegram is connected.")
        return
    feed, state = Feed(), load_state()
    if args.check:
        for sym in ("EURUSD", "XAUUSD"):
            for tf in ("M", "W", "D", "H4"):
                dd = feed.get(sym, tf)
                if dd is None:
                    print(f"--- {sym} {tf}: NO DATA")
                    continue
                print(f"--- {sym} {tf}: {len(dd['closed'])} closed candles, last candle forming: {dd['forming']}")
                print(dd["all"].tail(2))
        return
    if args.baseline:
        run_once(feed, state, send=False, force=True)
        save_state(state)
        print(f"Baseline saved: {len(state['seen'])} setups marked as already seen.")
        return
    if args.once:
        run_once(feed, state)
        save_state(state)
        return
    while True:
        run_once(feed, state)
        save_state(state)
        log.info("Scan finished, sleeping %s min", SCAN_EVERY_MIN)
        time.sleep(SCAN_EVERY_MIN * 60)


if __name__ == "__main__":
    main()

# ------------------------------ SETUP -------------------------------
# 1. pip install -r requirements.txt
# 2. Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID (see the GitHub Secrets steps).
# 3. python forex_alert_bot.py --test      Telegram test
#    python forex_alert_bot.py --check     data feed test
#    python forex_alert_bot.py --baseline  mark current setups as seen, send nothing
#    python forex_alert_bot.py --once      one scan
#    python forex_alert_bot.py             loop forever
