"""
Daily market-regime classifier (trend vs chop).

Built from the cached 5-min data by collapsing each day to a daily OHLC bar,
then computing 14-day ADX and the Choppiness Index. Thresholds are the STANDARD
textbook values (NOT optimized) to avoid the #1 regime pitfall: fitting the filter.

  ADX  > 25  -> trending      ADX  < 20  -> choppy
  Chop < 38.2-> trending      Chop > 61.8-> choppy

Causality: the regime label assigned to day D uses indicator values computed
through day D-1 only (known at D's open). So a strategy may use regime[D] to
decide whether to trade on day D without peeking.
"""
import numpy as np

ADX_TREND = 25.0
ADX_CHOP  = 20.0
CHOP_TREND = 38.2
CHOP_CHOP  = 61.8
PERIOD = 14


def _daily_bars(day_groups):
    dates, o, h, l, c = [], [], [], [], []
    for date, d in day_groups:
        dates.append(date)
        o.append(float(d["open"].iloc[0]))
        h.append(float(d["high"].max()))
        l.append(float(d["low"].min()))
        c.append(float(d["close"].iloc[-1]))
    return dates, np.array(o), np.array(h), np.array(l), np.array(c)


def _adx(h, l, c, period=PERIOD):
    n = len(h)
    tr = np.zeros(n); plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        plus_dm[i]  = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))

    atr = np.full(n, np.nan); pdi = np.full(n, np.nan); mdi = np.full(n, np.nan)
    dx  = np.full(n, np.nan); adx = np.full(n, np.nan)
    if n <= period:
        return adx

    atr_v = tr[1:period+1].sum()
    pdm_v = plus_dm[1:period+1].sum()
    mdm_v = minus_dm[1:period+1].sum()
    for i in range(period+1, n):
        atr_v = atr_v - atr_v/period + tr[i]
        pdm_v = pdm_v - pdm_v/period + plus_dm[i]
        mdm_v = mdm_v - mdm_v/period + minus_dm[i]
        if atr_v == 0:
            continue
        pdi[i] = 100 * pdm_v / atr_v
        mdi[i] = 100 * mdm_v / atr_v
        s = pdi[i] + mdi[i]
        dx[i] = 100 * abs(pdi[i] - mdi[i]) / s if s > 0 else 0.0

    # ADX = Wilder-smoothed DX
    first = period * 2
    if n > first:
        seed = np.nanmean(dx[period+1:first+1])
        adx[first] = seed
        for i in range(first+1, n):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1]*(period-1) + dx[i]) / period
    return adx


def _choppiness(h, l, c, period=PERIOD):
    n = len(h)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    ci = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        hi = h[i-period+1:i+1].max()
        lo = l[i-period+1:i+1].min()
        rng = hi - lo
        if rng > 0 and atr_sum > 0:
            ci[i] = 100 * np.log10(atr_sum / rng) / np.log10(period)
    return ci


def classify(day_groups) -> dict:
    """Return {date: {'adx','chop','regime'}} where regime in {trend,chop,neutral}.
    Uses prior-day indicator values (causal)."""
    dates, o, h, l, c = _daily_bars(day_groups)
    adx = _adx(h, l, c)
    chop = _choppiness(h, l, c)

    out = {}
    for i, date in enumerate(dates):
        # value known at open of day i = indicator through day i-1
        a = adx[i-1] if i >= 1 else np.nan
        ch = chop[i-1] if i >= 1 else np.nan
        regime = "neutral"
        if not np.isnan(a) and not np.isnan(ch):
            trend_votes = (a > ADX_TREND) + (ch < CHOP_TREND)
            chop_votes  = (a < ADX_CHOP)  + (ch > CHOP_CHOP)
            if trend_votes >= 1 and chop_votes == 0:
                regime = "trend"
            elif chop_votes >= 1 and trend_votes == 0:
                regime = "chop"
        out[date] = {"adx": a, "chop": ch, "regime": regime}
    return out
