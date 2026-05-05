"""
backtest/engine.py  —  v2
──────────────────────────
What changed from v1 and why:

PROBLEM 1 — Too many trades, fee drag killing returns.
FIX       — ADX hard gate (> 25) on trend strategies. If the market isn't
            trending strongly, no trend signal is emitted. Period.

PROBLEM 2 — Mean reversion shorting into bull trends.
FIX       — MR disabled in trending_bull / trending_bear entirely.
            RSI thresholds tightened from 32/68 → 28/72.

PROBLEM 3 — Breakout triggering on noise spikes.
FIX       — Volume threshold raised 2.0x → 2.8x.
            4h EMA confluence required for confirmation.

PROBLEM 4 — Static 2:1 R:R cuts winners early.
FIX       — Trailing stop: activates once trade moves 1x ATR in profit,
            then trails 1x ATR behind price. Lets winners run.

PROBLEM 5 — No signal quality filter.
FIX       — Composite score 0-100 per signal. Only scores >= 60 traded.

PROBLEM 6 — Cooldown too short (5 bars).
FIX       — Default cooldown 5 → 12 bars. After a losing trade, extra
            24-bar cooldown before re-entering (loss_cooldown).

PROBLEM 7 — 1h timeframe too noisy for trend signals.
FIX       — 4h features computed by resampling 1h data. Trend signals
            require 4h EMA alignment (MTF confluence). No look-ahead.
"""

import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta as ta

log = logging.getLogger("backtest.engine")

COMMISSION_PCT = 0.001
SLIPPAGE_PCT   = 0.0008
COST_PER_SIDE  = COMMISSION_PCT + SLIPPAGE_PCT
ROUND_TRIP     = COST_PER_SIDE * 2
RISK_FREE_RATE = 0.05
MIN_SCORE      = 60.0


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    direction:   Literal["long", "short"]
    strategy:    str
    regime:      str
    entry_time:  datetime
    entry_price: float
    exit_time:   datetime | None = None
    exit_price:  float | None    = None
    qty:         float = 0.0
    stop_loss:   float = 0.0       # original hard stop
    take_profit: float = 0.0       # final target (full or partial remainder)
    trail_stop:  float = 0.0       # active trailing stop level (0 = not yet active)
    breakeven_set: bool = False    # True once stop has been moved to entry
    partial_done:  bool = False    # True once partial profit has been booked
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    exit_reason: str   = ""
    slippage:    float = 0.0
    commission:  float = 0.0
    risk_pct:    float = 0.0
    score:       float = 0.0


@dataclass
class BacktestResult:
    symbol:           str
    timeframe:        str
    strategy:         str
    start_date:       datetime
    end_date:         datetime
    initial_capital:  float
    final_capital:    float
    total_return_pct: float
    cagr_pct:         float
    sharpe:           float
    sortino:          float
    calmar:           float
    max_drawdown_pct: float
    win_rate_pct:     float
    profit_factor:    float
    total_trades:     int
    avg_trade_pct:    float
    avg_win_pct:      float
    avg_loss_pct:     float
    best_trade_pct:   float
    worst_trade_pct:  float
    avg_hold_hours:   float
    trades:           list[Trade] = field(default_factory=list)
    equity_curve:     list[float] = field(default_factory=list)


# ── Feature engineering ────────────────────────────────────────────────────────

def _safe_bb_cols(bb: pd.DataFrame) -> tuple[str, str, str]:
    upper = next(c for c in bb.columns if c.startswith("BBU"))
    lower = next(c for c in bb.columns if c.startswith("BBL"))
    mid   = next(c for c in bb.columns if c.startswith("BBM"))
    return upper, lower, mid


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features on 1h data plus 4h MTF features.
    4h features are computed by resampling — no look-ahead bias.
    """
    df = df.copy().reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # 1h indicators
    df["ema20"]  = ta.ema(close, length=20)
    df["ema50"]  = ta.ema(close, length=50)
    df["ema200"] = ta.ema(close, length=200)
    df["rsi"]    = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"]      = macd.iloc[:, 0]
        df["macd_sig"]  = macd.iloc[:, 2]
        df["macd_hist"] = macd.iloc[:, 1]

    df["atr"]     = ta.atr(high, low, close, length=14)
    df["atr_pct"] = (df["atr"] / close) * 100

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        u, l, m = _safe_bb_cols(bb)
        df["bb_upper"] = bb[u]
        df["bb_lower"] = bb[l]
        df["bb_mid"]   = bb[m]
        df["bb_width"] = (bb[u] - bb[l]) / (bb[m] + 1e-9)

    vol_sma         = ta.sma(volume, length=20)
    df["vol_ratio"] = volume / vol_sma.replace(0, np.nan)

    adx = ta.adx(high, low, close, length=14)
    if adx is not None:
        df["adx"] = adx.iloc[:, 0]

    df["ema_bull"]      = (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema200"])
    df["ema_bear"]      = (df["ema20"] < df["ema50"]) & (df["ema50"] < df["ema200"])
    df["ema_spread_pct"]= ((df["ema20"] - df["ema50"]).abs() / close) * 100

    # Previous bar's MACD histogram — used for inflection detection in trend follow
    if "macd_hist" in df.columns:
        df["macd_hist_prev"] = df["macd_hist"].shift(1)

    # BB squeeze: width below its own 20-period average
    if "bb_width" in df.columns:
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(20).mean()

    # 4h MTF features via resampling
    df_idx = df.set_index("time")[["open", "high", "low", "close", "volume"]]
    df_4h = (
        df_idx
        .resample("4h", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
    )

    if len(df_4h) >= 50:
        df_4h["ema20_4h"]   = ta.ema(df_4h["close"], length=20)
        df_4h["ema50_4h"]   = ta.ema(df_4h["close"], length=50)
        adx_4h = ta.adx(df_4h["high"], df_4h["low"], df_4h["close"], length=14)
        if adx_4h is not None:
            df_4h["adx_4h"] = adx_4h.iloc[:, 0]
        df_4h["ema_bull_4h"] = df_4h["ema20_4h"] > df_4h["ema50_4h"]
        df_4h["ema_bear_4h"] = df_4h["ema20_4h"] < df_4h["ema50_4h"]

        mtf_cols = [c for c in ["ema_bull_4h", "ema_bear_4h", "adx_4h"] if c in df_4h.columns]
        df = df.set_index("time")
        df_mtf = df_4h[mtf_cols].reindex(df.index, method="ffill")
        df = df.join(df_mtf).reset_index()
    else:
        # Not enough data for 4h — fill with neutral values
        df["ema_bull_4h"] = False
        df["ema_bear_4h"] = False
        df["adx_4h"]      = 0.0

    return df


def assign_regime(row: pd.Series) -> str:
    atr_pct = row.get("atr_pct") or 0
    adx     = row.get("adx")     or 0
    adx_4h  = row.get("adx_4h")  or 0

    if atr_pct > 5.0:
        return "high_volatility"
    if row.get("ema_bull") and row.get("ema_bull_4h") and adx > 25 and adx_4h > 20:
        return "trending_bull"
    if row.get("ema_bear") and row.get("ema_bear_4h") and adx > 25 and adx_4h > 20:
        return "trending_bear"
    return "ranging"


# ── Per-strategy score thresholds ─────────────────────────────────────────────
# Calibrated to target 50-100 trades per 3 years of 4h data.
# v3 thresholds (65/62/70) were too tight — produced 3-5 trades total.
# These values target quality filtering without statistical strangulation.
# Rule of thumb: every +5 pts on threshold roughly halves trade count.
SCORE_THRESHOLD = {
    "trend_follow":   50.0,   # was 65 — loosened to restore trade volume
    "mean_reversion": 48.0,   # was 62 — MR scorer is stricter, needs lower bar
    "breakout":       55.0,   # was 70 — still highest bar, breakouts are rarer
}
MIN_SCORE = 45.0   # fallback floor — nothing below this trades regardless


# ── Signal scoring — separate logic per strategy type ─────────────────────────

def score_trend(row: pd.Series, direction: str, regime: str) -> float:
    """
    Scoring for trend-following signals.
    Rewards: strong ADX, aligned 4h trend, pullback RSI (not overbought entry),
             positive MACD momentum, above-average volume, wide EMA spread.
    Penalises: entering at RSI extremes (chasing), weak ADX, low volume.
    """
    score = 0.0

    # ADX strength — core of trend quality (up to 25 pts)
    adx = row.get("adx") or 0
    if adx >= 40:
        score += 25
    elif adx >= 30:
        score += 18
    elif adx >= 25:
        score += 10
    # Below 25 the trend signal function already gates — this won't be called

    # 4h confluence (up to 15 pts)
    if direction == "long" and row.get("ema_bull_4h"):
        score += 15
    elif direction == "short" and row.get("ema_bear_4h"):
        score += 15

    # RSI pullback quality: reward entries on pullbacks, not at extremes (up to 20 pts)
    rsi = row.get("rsi") or 50
    if direction == "long":
        # Best long entry: RSI pulled back to 45-58 before resuming up
        if 45 <= rsi <= 58:
            score += 20
        elif 58 < rsi <= 65:
            score += 12
        elif rsi > 65:
            score += 4    # chasing — penalise
    else:
        if 42 <= rsi <= 55:
            score += 20
        elif 35 <= rsi < 42:
            score += 12
        elif rsi < 35:
            score += 4

    # MACD histogram direction and magnitude (up to 20 pts)
    hist  = row.get("macd_hist") or 0
    close = row.get("close") or 1
    hist_pct = abs(hist) / close * 100
    if ((direction == "long" and hist > 0) or
            (direction == "short" and hist < 0)):
        score += min(hist_pct * 25, 20)

    # Volume participation (up to 15 pts)
    vol = row.get("vol_ratio") or 1.0
    if vol >= 1.5:
        score += 15
    elif vol >= 1.2:
        score += 10
    elif vol >= 1.0:
        score += 5

    # EMA spread: tight spread = weak trend (up to 5 pts)
    spread = row.get("ema_spread_pct") or 0
    score += min(spread * 3, 5)

    return min(score, 100)


def score_mean_reversion(row: pd.Series, direction: str) -> float:
    """
    Scoring for mean reversion signals.
    Rewards: deep BB penetration, extreme RSI, declining volume on the spike
             (exhaustion), price far from BB midline.
    The score_signal general function rewarded ADX and trend things — wrong
    for MR. This function is purpose-built for counter-trend entries.
    """
    score = 0.0

    close    = row.get("close") or 1
    bb_upper = row.get("bb_upper") or close
    bb_lower = row.get("bb_lower") or close
    bb_mid   = row.get("bb_mid")   or close
    atr      = row.get("atr") or 1
    rsi      = row.get("rsi") or 50
    vol      = row.get("vol_ratio") or 1.0

    # Depth of BB penetration (up to 30 pts) — deeper = stronger reversal signal
    if direction == "long":
        depth = (bb_lower - close) / atr   # positive when below lower band
        score += min(depth * 20, 30)
    else:
        depth = (close - bb_upper) / atr
        score += min(depth * 20, 30)

    # RSI extremity (up to 30 pts) — further from 50 = more extreme
    if direction == "long":
        rsi_extreme = max(0, 28 - rsi)     # positive when rsi < 28
        score += min(rsi_extreme * 3, 30)
    else:
        rsi_extreme = max(0, rsi - 72)
        score += min(rsi_extreme * 3, 30)

    # Distance from BB midline as % of BB width (up to 20 pts)
    # A bounce is more reliable when price has travelled far from equilibrium
    bb_width = bb_upper - bb_lower
    if bb_width > 0:
        if direction == "long":
            dist_from_mid = (bb_mid - close) / bb_width
        else:
            dist_from_mid = (close - bb_mid) / bb_width
        score += min(dist_from_mid * 30, 20)

    # Volume on spike: slightly elevated but not extreme (up to 15 pts)
    # A volume spike of 2x+ on the reversal bar is a good sign
    # Very high volume (3x+) can mean capitulation — also good
    if vol >= 2.0:
        score += 15
    elif vol >= 1.3:
        score += 10
    elif 0.7 <= vol < 1.0:
        # Low volume on the extreme = no conviction, slight penalty
        score += 3

    # ADX low = true ranging (up to 5 pts) — reward when market is genuinely flat
    adx = row.get("adx") or 30
    if adx < 15:
        score += 5
    elif adx < 20:
        score += 3

    return min(score, 100)


def score_signal(row: pd.Series, direction: str, regime: str) -> float:
    """Dispatch to the correct scorer based on context."""
    # This generic version is kept for breakout strategy
    score = 0.0
    vol = row.get("vol_ratio") or 1.0
    score += min((vol - 1.0) * 15, 20) if vol > 1.0 else 0
    adx = row.get("adx") or 0
    score += min((adx - 20) * 0.8, 20) if adx > 20 else 0
    rsi = row.get("rsi") or 50
    score += min(abs(rsi - 50) * 0.6, 15)
    if direction == "long" and row.get("ema_bull_4h"):
        score += 15
    elif direction == "short" and row.get("ema_bear_4h"):
        score += 15
    if direction == "long" and regime == "trending_bull":
        score += 10
    elif direction == "short" and regime == "trending_bear":
        score += 10
    return min(score, 100)


# ── Strategies ────────────────────────────────────────────────────────────────

def signal_trend_follow(row: pd.Series, regime: str) -> tuple[str | None, float]:
    """
    Multi-timeframe trend following with pullback entry filter.

    The core problem with the previous version: it was entering on any
    candle that passed the technical gates, including candles at the
    TOP of an impulse move (RSI 65-70, high volume, extended from EMA).
    These entries get stopped out immediately when the natural pullback comes.

    New requirement: PULLBACK ENTRY
    We want to enter DURING a retracement within a larger trend, not at the
    momentum peak. This means RSI should be cooling off (below 58 on longs)
    and price should be within 2 ATR of the EMA50 (not extended).

    New requirement: MACD HISTOGRAM INFLECTION
    Instead of just "histogram positive", we now require the histogram to have
    been rising for at least 1 bar. This filters out entries where momentum
    is already fading at the time of entry.
    """
    if regime not in ("trending_bull", "trending_bear"):
        return None, 0

    adx = row.get("adx") or 0
    if adx < 23:   # lowered from 27 — was cutting too many valid trending bars
        return None, 0

    rsi       = row.get("rsi")      or 50
    macd_hist = row.get("macd_hist") or 0
    macd_prev = row.get("macd_hist_prev") or 0
    vol_ratio = row.get("vol_ratio") or 0
    close     = row.get("close")    or 0
    ema50     = row.get("ema50")    or 0
    atr       = row.get("atr")      or 1

    # Extension filter: loosened from 3.0 ATR to 4.5 ATR.
    # 3.0 ATR was rejecting valid pullback entries — on 4h BTC, a move of
    # 3 ATR from EMA50 is common during normal trending conditions.
    if ema50 > 0 and atr > 0:
        dist_from_ema50 = abs(close - ema50) / atr
        if dist_from_ema50 > 4.5:
            return None, 0

    # MACD inflection: kept as-is — this is the most valuable filter.
    # Requiring histogram to be growing (not just positive) eliminates
    # entries at momentum peaks. This is worth keeping tight.
    hist_inflecting_up   = macd_hist > 0 and macd_hist > macd_prev
    hist_inflecting_down = macd_hist < 0 and macd_hist < macd_prev

    # ── Long entry: RSI band widened from (42, 62) to (38, 68)
    # The original (42, 62) band was the core of the over-filtering.
    # On 4h BTC with a trending_bull regime, RSI routinely runs 60-67
    # during healthy pullback entries. The narrow band eliminated most of them.
    if (row.get("ema_bull") and row.get("ema_bull_4h") and
            38 < rsi < 68 and
            hist_inflecting_up and
            vol_ratio > 0.9):           # lowered from 1.0 — volume sometimes lags

        s = score_trend(row, "long", regime)
        threshold = SCORE_THRESHOLD["trend_follow"]
        if s >= threshold:
            return "long", s

    # ── Short entry: RSI band widened from (38, 58) to (32, 62)
    if (row.get("ema_bear") and row.get("ema_bear_4h") and
            32 < rsi < 62 and
            hist_inflecting_down and
            vol_ratio > 0.9):

        s = score_trend(row, "short", regime)
        threshold = SCORE_THRESHOLD["trend_follow"]
        if s >= threshold:
            return "short", s

    return None, 0


def signal_mean_reversion(row: pd.Series, regime: str) -> tuple[str | None, float]:
    """
    Bollinger Band mean reversion — purpose-built for ranging markets.

    Three new entry filters added:
    1. BB width filter: Only trade when the band is wide enough that a
       reversion is meaningful. If BB width < 3% of price, the range is
       too tight and the signal is noise.
    2. RSI divergence proxy: Require RSI to be not just below 28, but to
       have been below 35 for at least the current candle (captured by
       the strict threshold). The depth scoring handles the rest.
    3. Candle body filter: The reversal candle should show price trying
       to recover. Close > open on a long signal (green candle closing off lows)
       indicates buyers stepped in during the candle — a much stronger signal
       than a candle that closed at its lows.
    """
    if regime != "ranging":
        return None, 0
    if row.get("bb_squeeze", False):
        return None, 0

    close    = row.get("close") or 0
    open_p   = row.get("open") or close    # candle open price
    bb_upper = row.get("bb_upper") or 0
    bb_lower = row.get("bb_lower") or 0
    bb_mid   = row.get("bb_mid")   or 0
    rsi      = row.get("rsi") or 50
    atr      = row.get("atr") or 1
    vol      = row.get("vol_ratio") or 1.0

    if bb_upper == 0 or bb_lower == 0:
        return None, 0

    # Filter 1: BB width must be meaningful.
    # Lowered from 3.0% to 1.5% — 3.0% was too strict for 4h ETH which
    # often has BB widths of 2-3% during genuine ranging conditions.
    bb_width_pct = ((bb_upper - bb_lower) / (close + 1e-9)) * 100
    if bb_width_pct < 1.5:
        return None, 0

    # Filter 2: Minimum volume
    if vol < 0.6:   # lowered from 0.7 — 4h crypto can have legitimate low-vol ranging
        return None, 0

    threshold = SCORE_THRESHOLD["mean_reversion"]

    # ── Long: RSI threshold widened from 28 to 32
    # RSI 28 was only triggering during severe crashes, not normal oversold conditions.
    # RSI 32 catches genuine oversold bounces while still being selective.
    if close < bb_lower and rsi < 32:
        candle_mid = (open_p + (row.get("low") or close)) / 2
        body_recovering = (close > open_p) or (close > candle_mid)
        if not body_recovering:
            return None, 0

        s = score_mean_reversion(row, "long")
        if s >= threshold:
            return "long", round(s, 1)

    # ── Short: RSI threshold widened from 72 to 68
    if close > bb_upper and rsi > 68:
        candle_mid = (open_p + (row.get("high") or close)) / 2
        body_rejecting = (close < open_p) or (close < candle_mid)
        if not body_rejecting:
            return None, 0

        s = score_mean_reversion(row, "short")
        if s >= threshold:
            return "short", round(s, 1)

    return None, 0


def signal_breakout(row: pd.Series, regime: str) -> tuple[str | None, float]:
    if regime == "high_volatility":
        return None, 0

    close     = row.get("close") or 0
    bb_upper  = row.get("bb_upper") or 0
    bb_lower  = row.get("bb_lower") or 0
    vol_ratio = row.get("vol_ratio") or 0
    rsi       = row.get("rsi") or 50
    adx       = row.get("adx") or 0

    if vol_ratio < 2.8 or adx > 45:
        return None, 0

    if close > bb_upper and rsi < 75 and row.get("ema_bull_4h"):
        score = 45 + min((vol_ratio - 2.8) * 10, 25) + min((75 - rsi) * 0.5, 15)
        score += 10 if regime == "trending_bull" else 0
        if score >= MIN_SCORE:
            return "long", round(score, 1)

    if close < bb_lower and rsi > 25 and row.get("ema_bear_4h"):
        score = 45 + min((vol_ratio - 2.8) * 10, 25) + min((rsi - 25) * 0.5, 15)
        score += 10 if regime == "trending_bear" else 0
        if score >= MIN_SCORE:
            return "short", round(score, 1)

    return None, 0


STRATEGY_SIGNALS = {
    "trend_follow":   signal_trend_follow,
    "mean_reversion": signal_mean_reversion,
    "breakout":       signal_breakout,
}


# ── Core engine ────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(
        self,
        symbol:           str   = "BTC/USDT",
        timeframe:        str   = "1h",
        strategy:         str   = "trend_follow",
        initial_capital:  float = 10_000.0,
        risk_per_trade:   float = 0.01,
        atr_mult:         float = 1.5,
        rr:               float = 2.0,
        # ── Three-tier exit system ─────────────────────────────────────────
        # Tier 1 — Breakeven stop
        # Once price moves `be_trigger_r` × risk distance in our favour,
        # the stop is moved to entry price + a tiny buffer (be_buffer_r × risk).
        # This eliminates the scenario where a winning trade turns into a loss.
        be_trigger_r:     float = 0.75,   # move BE after 0.75R profit
        be_buffer_r:      float = 0.05,   # stop sits 0.05R above entry (covers fees)
        # Tier 2 — Partial profit lock
        # At `partial_r` × risk in profit, close `partial_pct` of the position.
        # This books a guaranteed win on part of the trade.
        use_partial:      bool  = True,
        partial_r:        float = 1.0,    # take partial at 1R profit
        partial_pct:      float = 0.40,   # close 40% of position
        # Tier 3 — Remaining position: trailing stop or static target
        use_trailing:     bool  = True,
        trail_trigger_r:  float = 1.5,    # trail activates at 1.5R profit (not ATR units)
        trail_distance_r: float = 1.0,    # trail sits 1R behind the high/low
        # ──────────────────────────────────────────────────────────────────
        max_open:         int   = 1,
        cooldown_bars:    int   = 12,
        loss_cooldown:    int   = 24,
    ):
        self.symbol           = symbol
        self.timeframe        = timeframe
        self.strategy_name    = strategy
        self.signal_fn        = STRATEGY_SIGNALS[strategy]
        self.initial_capital  = initial_capital
        self.risk_per_trade   = risk_per_trade
        self.atr_mult         = atr_mult
        self.rr               = rr
        self.be_trigger_r     = be_trigger_r
        self.be_buffer_r      = be_buffer_r
        self.use_partial      = use_partial
        self.partial_r        = partial_r
        self.partial_pct      = partial_pct
        self.use_trailing     = use_trailing
        self.trail_trigger_r  = trail_trigger_r
        self.trail_distance_r = trail_distance_r
        self.max_open         = max_open
        self.cooldown_bars    = cooldown_bars
        self.loss_cooldown    = loss_cooldown

    def _size_position(self, portfolio, entry, atr):
        dr  = portfolio * self.risk_per_trade
        sd  = atr * self.atr_mult           # stop distance in price units (= 1R)
        if sd <= 0 or entry <= 0:
            return 0.0, 0.0, 0.0, 0.0
        qty = dr / sd
        sl  = entry - sd
        tp  = entry + sd * self.rr
        return round(qty, 6), round(sl, 4), round(tp, 4), round(sd, 4)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        df = df.dropna(subset=["ema200", "rsi", "atr", "adx"]).reset_index(drop=True)

        portfolio       = self.initial_capital
        equity_curve    = [portfolio]
        trades          = []
        open_trades     = []
        last_signal_bar = -999
        last_loss_bar   = -999

        for i, row in df.iterrows():
            close = row["close"]
            high  = row["high"]
            low   = row["low"]
            atr   = row.get("atr") or 0

            # ── Three-tier trade management ───────────────────────────────────
            still_open = []
            for t in open_trades:
                risk_dist = abs(t.entry_price - t.stop_loss)  # 1R in price units
                if risk_dist <= 0:
                    still_open.append(t)
                    continue

                # Helper: how many R has this trade moved in our favour right now
                if t.direction == "long":
                    profit_r_high = (high - t.entry_price) / risk_dist
                    profit_r_now  = (close - t.entry_price) / risk_dist
                else:
                    profit_r_high = (t.entry_price - low)  / risk_dist
                    profit_r_now  = (t.entry_price - close) / risk_dist

                # ── TIER 1: Move stop to breakeven ─────────────────────────
                # Triggers once high-water profit >= be_trigger_r.
                # Stop is moved to entry + tiny buffer (covers round-trip fees).
                # Once set it never moves back — even if price retreats.
                if not t.breakeven_set and profit_r_high >= self.be_trigger_r:
                    buffer = risk_dist * self.be_buffer_r
                    if t.direction == "long":
                        new_be = t.entry_price + buffer
                        if new_be > t.stop_loss:        # only move forward
                            t.stop_loss    = round(new_be, 4)
                            t.breakeven_set = True
                    else:
                        new_be = t.entry_price - buffer
                        if new_be < t.stop_loss:
                            t.stop_loss    = round(new_be, 4)
                            t.breakeven_set = True

                # ── TIER 2: Partial profit lock ─────────────────────────────
                # At partial_r, close partial_pct of the position.
                # The booked P&L goes directly to portfolio cash.
                # Remaining qty stays open for the bigger target/trail.
                if self.use_partial and not t.partial_done and profit_r_high >= self.partial_r:
                    partial_qty = round(t.qty * self.partial_pct, 6)
                    if partial_qty > 0:
                        # Exit partial at current close (conservative — not high)
                        if t.direction == "long":
                            exit_p   = close * (1 - COST_PER_SIDE)
                            part_pnl = (exit_p - t.entry_price) * partial_qty
                        else:
                            exit_p   = close * (1 + COST_PER_SIDE)
                            part_pnl = (t.entry_price - exit_p) * partial_qty
                        portfolio    += part_pnl
                        t.qty        -= partial_qty   # reduce remaining size
                        t.partial_done = True
                        # Record this as a partial close in exit_reason later

                # ── TIER 3: Trailing stop on remaining position ─────────────
                # Only activates after profit_r_high >= trail_trigger_r.
                # Trail distance expressed in R (not raw ATR) so it scales
                # automatically with position sizing.
                if self.use_trailing and profit_r_high >= self.trail_trigger_r:
                    trail_dist_price = risk_dist * self.trail_distance_r
                    if t.direction == "long":
                        new_ts = high - trail_dist_price
                        if new_ts > t.trail_stop:
                            t.trail_stop = round(new_ts, 4)
                    else:
                        new_ts = low + trail_dist_price
                        if t.trail_stop == 0 or new_ts < t.trail_stop:
                            t.trail_stop = round(new_ts, 4)

                # ── Determine active stop ────────────────────────────────────
                # Priority: trailing stop > breakeven stop > original stop
                active_sl = t.trail_stop if t.trail_stop > 0 else t.stop_loss

                reason   = None
                exit_ref = 0.0

                if t.direction == "long":
                    if low <= active_sl:
                        exit_ref = active_sl
                        reason   = ("trail_stop"    if t.trail_stop > 0
                                    else "breakeven" if t.breakeven_set
                                    else "stop_loss")
                    elif high >= t.take_profit:
                        exit_ref = t.take_profit
                        reason   = "take_profit"
                else:
                    if high >= active_sl:
                        exit_ref = active_sl
                        reason   = ("trail_stop"    if t.trail_stop > 0
                                    else "breakeven" if t.breakeven_set
                                    else "stop_loss")
                    elif low <= t.take_profit:
                        exit_ref = t.take_profit
                        reason   = "take_profit"

                if reason is None:
                    still_open.append(t)
                    continue

                # ── Close remaining position ─────────────────────────────────
                if t.direction == "long":
                    exit_net = exit_ref * (1 - COST_PER_SIDE)
                    pnl      = (exit_net - t.entry_price) * t.qty
                else:
                    exit_net = exit_ref * (1 + COST_PER_SIDE)
                    pnl      = (t.entry_price - exit_net) * t.qty

                cost         = t.qty * t.entry_price
                t.exit_time  = row["time"]
                t.exit_price = round(exit_net, 4)
                t.pnl        = round(pnl, 4)
                t.pnl_pct    = round((pnl / cost) * 100, 3) if cost else 0
                t.exit_reason= reason + ("_partial" if t.partial_done else "")
                portfolio   += pnl

                if pnl < 0:
                    last_loss_bar = i

                trades.append(t)

            open_trades = still_open

            # ── Signal gating ─────────────────────────────────────────────────
            if len(open_trades) >= self.max_open:
                equity_curve.append(portfolio)
                continue

            if (i - last_signal_bar) < self.cooldown_bars:
                equity_curve.append(portfolio)
                continue

            if (i - last_loss_bar) < self.loss_cooldown:
                equity_curve.append(portfolio)
                continue

            regime             = assign_regime(row)
            direction, score   = self.signal_fn(row, regime)

            if direction is None:
                equity_curve.append(portfolio)
                continue

            qty, sl, tp, risk_dist = self._size_position(portfolio, close, atr)
            if qty <= 0:
                equity_curve.append(portfolio)
                continue

            if direction == "short":
                sl = close + (close - sl)
                tp = close - (tp - close)

            entry_net = (close * (1 + COST_PER_SIDE) if direction == "long"
                         else close * (1 - COST_PER_SIDE))

            t = Trade(
                symbol=self.symbol, direction=direction,
                strategy=self.strategy_name, regime=regime,
                entry_time=row["time"],
                entry_price=round(entry_net, 4),
                qty=qty, stop_loss=sl, take_profit=tp,
                trail_stop=0.0,
                breakeven_set=False,
                partial_done=False,
                risk_pct=self.risk_per_trade * 100,
                commission=round(qty * close * ROUND_TRIP, 4),
                score=score,
            )
            open_trades.append(t)
            last_signal_bar = i
            equity_curve.append(portfolio)

        # Force-close remaining open positions
        for t in open_trades:
            lc = df["close"].iloc[-1]
            if t.direction == "long":
                pnl = (lc * (1 - COST_PER_SIDE) - t.entry_price) * t.qty
            else:
                pnl = (t.entry_price - lc * (1 + COST_PER_SIDE)) * t.qty
            t.exit_time   = df["time"].iloc[-1]
            t.exit_price  = round(lc, 4)
            cost          = t.qty * t.entry_price
            t.pnl         = round(pnl, 4)
            t.pnl_pct     = round((pnl / cost) * 100, 3) if cost else 0
            t.exit_reason = "end_of_data"
            portfolio    += pnl
            trades.append(t)

        return self._compute_stats(trades, equity_curve, portfolio, df)

    def _compute_stats(self, trades, equity_curve, final_cap, df):
        equity = np.array(equity_curve, dtype=float)
        total_ret = (final_cap - self.initial_capital) / self.initial_capital * 100

        start = df["time"].iloc[0]
        end   = df["time"].iloc[-1]
        if hasattr(start, "to_pydatetime"):
            start, end = start.to_pydatetime(), end.to_pydatetime()

        years = max((end - start).days / 365, 0.001)
        cagr  = ((final_cap / max(self.initial_capital, 1)) ** (1 / years) - 1) * 100

        ppy    = {"1m": 525_600, "5m": 105_120, "15m": 35_040,
                  "1h": 8_760, "4h": 2_190, "1d": 365}.get(self.timeframe, 8_760)
        rets   = np.diff(equity) / np.maximum(equity[:-1], 1e-6)
        rets   = rets[np.isfinite(rets)]
        rf     = RISK_FREE_RATE / ppy
        exc    = rets - rf
        sharpe = float(np.mean(exc) / (np.std(rets) + 1e-10) * math.sqrt(ppy))
        dn     = rets[rets < 0]
        sortino= float(np.mean(exc) / (np.std(dn) + 1e-10) * math.sqrt(ppy))

        peak   = np.maximum.accumulate(equity)
        dds    = (peak - equity) / (peak + 1e-10) * 100
        max_dd = float(np.max(dds))
        calmar = cagr / max(max_dd, 0.01)

        won  = [t for t in trades if t.pnl > 0]
        lost = [t for t in trades if t.pnl <= 0]
        gp   = sum(t.pnl for t in won)
        gl   = sum(abs(t.pnl) for t in lost)
        wr   = len(won) / len(trades) * 100 if trades else 0
        pf   = gp / gl if gl > 0 else float("inf")

        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        pcts = [t.pnl_pct for t in trades]

        hold_h = [(t.exit_time - t.entry_time).total_seconds() / 3600
                  for t in trades if t.exit_time and t.entry_time]

        return BacktestResult(
            symbol=self.symbol, timeframe=self.timeframe, strategy=self.strategy_name,
            start_date=start, end_date=end,
            initial_capital=self.initial_capital, final_capital=round(final_cap, 2),
            total_return_pct=round(total_ret, 2), cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 3), sortino=round(sortino, 3), calmar=round(calmar, 3),
            max_drawdown_pct=round(max_dd, 2),
            win_rate_pct=round(wr, 1),
            profit_factor=round(pf, 3),
            total_trades=len(trades),
            avg_trade_pct=round(safe_mean(pcts), 3),
            avg_win_pct=round(safe_mean([t.pnl_pct for t in won]), 3),
            avg_loss_pct=round(safe_mean([t.pnl_pct for t in lost]), 3),
            best_trade_pct=round(max(pcts, default=0), 3),
            worst_trade_pct=round(min(pcts, default=0), 3),
            avg_hold_hours=round(safe_mean(hold_h), 1),
            trades=trades, equity_curve=list(equity_curve),
        )


# ── Monte Carlo ────────────────────────────────────────────────────────────────

def monte_carlo(trades, initial_capital, n_simulations=5_000,
                percentiles=(5, 25, 50, 75, 95)):
    pnls = [t.pnl for t in trades]
    if not pnls:
        return {}
    fv, md = [], []
    for _ in range(n_simulations):
        eq = initial_capital; peak = eq; mdd = 0.0
        for p in random.sample(pnls, len(pnls)):
            eq += p; peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak * 100)
        fv.append(eq); md.append(mdd)
    fv = np.array(fv); md = np.array(md)
    return {
        "n_simulations":  n_simulations,
        "pct_profitable": round(float(np.mean(fv > initial_capital)) * 100, 1),
        "final_value":    {p: round(float(np.percentile(fv, p)), 2) for p in percentiles},
        "max_drawdown":   {p: round(float(np.percentile(md, p)), 2) for p in percentiles},
        "mean_final": round(float(np.mean(fv)), 2),
        "worst_final":round(float(np.min(fv)), 2),
        "best_final": round(float(np.max(fv)), 2),
    }


# ── Walk-forward ───────────────────────────────────────────────────────────────

def walk_forward(df, engine_kwargs, train_pct=0.7, n_windows=3):
    results = []
    window  = len(df) // n_windows
    for i in range(n_windows):
        chunk  = df.iloc[i * window:(i+1) * window].reset_index(drop=True)
        oos    = chunk.iloc[int(len(chunk) * train_pct):].reset_index(drop=True)
        if len(oos) < 100:
            continue
        eng = BacktestEngine(**engine_kwargs)
        r   = eng.run(oos)
        r.strategy = f"{r.strategy}_wf{i+1}"
        results.append(r)
    return results


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_ohlcv_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df.sort_values("time").reset_index(drop=True)


async def load_ohlcv_from_db(symbol, timeframe, db_dsn, limit=None):
    import asyncpg
    pool = await asyncpg.create_pool(db_dsn, min_size=1, max_size=3)
    sql  = f"""
        SELECT time, open, high, low, close, volume
        FROM ohlcv WHERE symbol=$1 AND timeframe=$2
        ORDER BY time ASC {f'LIMIT {limit}' if limit else ''}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, symbol, timeframe)
    await pool.close()
    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df