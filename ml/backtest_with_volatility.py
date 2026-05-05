"""
ml/backtest_with_volatility.py
────────────────────────────────
Uses the trained volatility model to time entries, with trend/regime
logic to determine direction.

Architecture:
  Volatility model  → WHEN to trade (prob >= threshold = expect big move)
  Trend filter      → WHICH DIRECTION to trade
  BacktestEngine    → HOW MUCH to risk and WHEN to exit

Entry logic:
  1. Model predicts volatile (prob >= threshold)
  2. Regime is trending_bull → enter LONG
     Regime is trending_bear → enter SHORT
     Regime is ranging       → skip (no directional edge in chop)
     Regime is high_vol      → enter in EMA direction with reduced size

Exit logic (three-tier, inherited from BacktestEngine):
  Tier 1: Move stop to breakeven at 0.75R
  Tier 2: Take 40% partial profit at 1.5R
  Tier 3: Trail remaining position at 1.5R distance (activated at 2.0R)

Why wider stops than before:
  The volatility model tells us a big move is coming.
  A 1.5 ATR stop in a predicted-volatile bar is too tight — normal
  intrabar noise wicks it out before the directional move develops.
  We use 2.0 ATR stops when vol_prob >= threshold, matching the
  expected excursion magnitude from the training data (avg 3.77% BTC,
  5.08% ETH = roughly 2–3 ATR on a 4h chart).

Usage:
    python ml/backtest_with_volatility.py \
        --csv ml/data/BTC_USDT_4h_alpha.csv \
        --model ml/models/BTC_USDT_4h_alpha_volatility.json \
        --meta  ml/models/BTC_USDT_4h_alpha_volatility_meta.json

    python ml/backtest_with_volatility.py \
        --csv ml/data/ETH_USDT_4h_alpha.csv \
        --model ml/models/ETH_USDT_4h_alpha_volatility.json \
        --meta  ml/models/ETH_USDT_4h_alpha_volatility_meta.json \
        --threshold 0.65 \
        --atr-mult 2.0 \
        --save-trades

    # Compare: model-filtered vs unfiltered on same data
    python ml/backtest_with_volatility.py \
        --csv ml/data/ETH_USDT_4h_alpha.csv \
        --model ml/models/ETH_USDT_4h_alpha_volatility.json \
        --meta  ml/models/ETH_USDT_4h_alpha_volatility_meta.json \
        --compare
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
except ImportError:
    print("Run: pip install xgboost")
    sys.exit(1)

from backtest.engine import (
    build_features, load_ohlcv_from_csv,
    assign_regime, Trade, BacktestResult,
    COST_PER_SIDE, ROUND_TRIP, RISK_FREE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.vol_backtest")

ALPHA_COLS = [
    "funding_rate_8h", "funding_rate_24h_sum", "funding_rate_3d_sum",
    "funding_zscore", "funding_extreme_long", "funding_extreme_short",
    "oi_change_pct", "oi_change_3bar", "oi_zscore", "oi_acceleration",
    "volume_delta", "buy_ratio", "buy_ratio_ma10",
    "buy_ratio_zscore", "cum_delta_3bar", "delta_ma",
    "ls_ratio", "ls_ratio_ma10", "ls_ratio_zscore",
    "ls_ratio_change", "crowd_extreme_long", "crowd_extreme_short",
    "funding_oi_diverge", "delta_price_agree", "delta_price_diverge",
    "sentiment_agreement", "oi_delta_confirm",
    "funding_reversal_long", "funding_reversal_short",
]

REGIMES = ["ranging", "trending_bull", "trending_bear", "high_volatility"]


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(row: pd.Series, feat_cols: list[str]) -> np.ndarray:
    vals = []
    for col in feat_cols:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            vals.append(0.0)
        elif isinstance(val, bool):
            vals.append(float(val))
        else:
            vals.append(float(val))
    return np.array(vals, dtype=np.float32).reshape(1, -1)


# ── Direction logic ────────────────────────────────────────────────────────────

def get_direction(row: pd.Series, regime: str, method: str) -> tuple[str | None, str]:
    """
    Determine trade direction given the current bar features and regime.

    Returns (direction, reason) where direction is 'long', 'short', or None.

    method options:
      'trend'   — follow EMA alignment (recommended)
      'regime'  — strict: only trade in confirmed trending regimes
      'macd'    — use MACD histogram direction
    """
    ema_bull   = bool(row.get("ema_bull",    False))
    ema_bear   = bool(row.get("ema_bear",    False))
    bull_4h    = bool(row.get("ema_bull_4h", False))
    bear_4h    = bool(row.get("ema_bear_4h", False))
    macd_hist  = row.get("macd_hist") or 0
    rsi        = row.get("rsi") or 50

    if method == "regime":
        # Only trade when regime is unambiguously trending
        if regime == "trending_bull":
            return "long", "regime=trending_bull"
        elif regime == "trending_bear":
            return "short", "regime=trending_bear"
        else:
            return None, f"skip: regime={regime}"

    elif method == "macd":
        if macd_hist > 0 and rsi < 75:
            return "long", "macd_positive"
        elif macd_hist < 0 and rsi > 25:
            return "short", "macd_negative"
        else:
            return None, "macd_neutral"

    else:  # method == "trend" (default)
        # Use 1h + 4h EMA alignment for strongest signal
        # In high_volatility regime, require 4h alignment too (more volatile = more careful)
        if regime == "ranging":
            return None, "skip: ranging (no directional edge)"

        if regime == "high_volatility":
            # More selective in chaotic conditions — require both timeframes
            if ema_bull and bull_4h and rsi < 70:
                return "long", "trend_bull_both_tf"
            elif ema_bear and bear_4h and rsi > 30:
                return "short", "trend_bear_both_tf"
            else:
                return None, "skip: high_vol no clear trend"

        if ema_bull and rsi < 72:
            return "long", "ema_bull"
        elif ema_bear and rsi > 28:
            return "short", "ema_bear"
        else:
            return None, "skip: ema_conflicted"


# ── Volatility-aware engine ────────────────────────────────────────────────────

class VolatilityBacktestEngine:
    """
    Backtest engine driven by volatility model predictions.

    Key differences from BacktestEngine:
    1. Entry triggered by vol_prob >= threshold (not by strategy signal)
    2. Stop distance scales with vol_prob (higher probability = wider stop)
    3. Three-tier exit system inherited and always active
    4. Direction determined by separate trend/regime logic
    5. Regime=ranging is explicitly skipped
    """

    def __init__(
        self,
        model:             xgb.XGBClassifier,
        feat_cols:         list[str],
        threshold:         float  = 0.60,
        symbol:            str    = "UNKNOWN",
        timeframe:         str    = "4h",
        initial_capital:   float  = 10_000.0,
        risk_per_trade:    float  = 0.01,
        # Stop sizing: base_atr_mult used at threshold, scales up with prob
        base_atr_mult:     float  = 1.8,
        max_atr_mult:      float  = 2.5,
        rr:                float  = 2.5,
        direction_method:  str    = "trend",
        # Three-tier exit
        be_trigger_r:      float  = 0.75,
        be_buffer_r:       float  = 0.05,
        use_partial:       bool   = True,
        partial_r:         float  = 1.5,
        partial_pct:       float  = 0.40,
        use_trailing:      bool   = True,
        trail_trigger_r:   float  = 2.0,
        trail_distance_r:  float  = 1.5,
        # Sizing modifiers by regime
        high_vol_size_mult:float  = 0.6,   # reduce size in chaotic conditions
        max_open:          int    = 1,
        cooldown_bars:     int    = 6,
        loss_cooldown:     int    = 12,
    ):
        self.model             = model
        self.feat_cols         = feat_cols
        self.threshold         = threshold
        self.symbol            = symbol
        self.timeframe         = timeframe
        self.initial_capital   = initial_capital
        self.risk_per_trade    = risk_per_trade
        self.base_atr_mult     = base_atr_mult
        self.max_atr_mult      = max_atr_mult
        self.rr                = rr
        self.direction_method  = direction_method
        self.be_trigger_r      = be_trigger_r
        self.be_buffer_r       = be_buffer_r
        self.use_partial       = use_partial
        self.partial_r         = partial_r
        self.partial_pct       = partial_pct
        self.use_trailing      = use_trailing
        self.trail_trigger_r   = trail_trigger_r
        self.trail_distance_r  = trail_distance_r
        self.high_vol_size_mult= high_vol_size_mult
        self.max_open          = max_open
        self.cooldown_bars     = cooldown_bars
        self.loss_cooldown     = loss_cooldown

    def _atr_mult_for_prob(self, prob: float) -> float:
        """
        Scale ATR multiplier with vol probability.
        Higher probability → expect bigger move → use wider stop.
        At threshold (e.g. 0.60): use base_atr_mult (e.g. 1.8)
        At prob=1.0: use max_atr_mult (e.g. 2.5)
        """
        t     = self.threshold
        scale = (prob - t) / (1.0 - t + 1e-9)
        return self.base_atr_mult + scale * (self.max_atr_mult - self.base_atr_mult)

    def _size_position(
        self, portfolio: float, entry: float, atr: float,
        atr_mult: float, regime: str,
    ) -> tuple[float, float, float, float]:
        """Returns (qty, stop_loss, take_profit, risk_distance)."""
        size_mult  = self.high_vol_size_mult if regime == "high_volatility" else 1.0
        dollar_risk= portfolio * self.risk_per_trade * size_mult
        stop_dist  = atr * atr_mult

        if stop_dist <= 0 or entry <= 0:
            return 0.0, 0.0, 0.0, 0.0

        qty        = dollar_risk / stop_dist
        stop_loss  = entry - stop_dist
        take_profit= entry + stop_dist * self.rr

        return round(qty, 6), round(stop_loss, 4), round(take_profit, 4), round(stop_dist, 4)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        df = df.dropna(subset=["atr"]).reset_index(drop=True)

        # Add regime one-hot columns for model features
        for r in REGIMES:
            df[f"regime_{r}"] = df.apply(
                lambda row: 1 if assign_regime(row) == r else 0, axis=1
            )

        # Pre-compute all model probabilities in one batch call (much faster)
        log.info("Computing volatility probabilities for %d bars…", len(df))
        X_all   = np.vstack([
            extract_features(df.iloc[i], self.feat_cols)
            for i in range(len(df))
        ])
        vol_probs = self.model.predict_proba(X_all)[:, 1]
        log.info("Done. Avg prob: %.3f  Above threshold (%.2f): %d bars (%.1f%%)",
                 vol_probs.mean(), self.threshold,
                 (vol_probs >= self.threshold).sum(),
                 (vol_probs >= self.threshold).mean() * 100)

        portfolio       = self.initial_capital
        equity_curve    = [portfolio]
        trades          = []
        open_trades     = []
        last_signal_bar = -999
        last_loss_bar   = -999
        skipped_regime  = 0
        skipped_thresh  = 0
        skipped_dir     = 0

        for i, row in df.iterrows():
            close    = row["close"]
            high     = row["high"]
            low      = row["low"]
            atr      = row.get("atr") or 0
            vol_prob = vol_probs[i]

            # ── Three-tier trade management ───────────────────────────────────
            still_open = []
            for t in open_trades:
                risk_dist = abs(t.entry_price - t.stop_loss)
                if risk_dist <= 0:
                    still_open.append(t)
                    continue

                if t.direction == "long":
                    profit_r_high = (high - t.entry_price) / risk_dist
                else:
                    profit_r_high = (t.entry_price - low) / risk_dist

                # Tier 1: Breakeven stop
                if not t.breakeven_set and profit_r_high >= self.be_trigger_r:
                    buf = risk_dist * self.be_buffer_r
                    if t.direction == "long":
                        new_be = t.entry_price + buf
                        if new_be > t.stop_loss:
                            t.stop_loss    = round(new_be, 4)
                            t.breakeven_set= True
                    else:
                        new_be = t.entry_price - buf
                        if new_be < t.stop_loss:
                            t.stop_loss    = round(new_be, 4)
                            t.breakeven_set= True

                # Tier 2: Partial profit
                if self.use_partial and not t.partial_done and profit_r_high >= self.partial_r:
                    pqty = round(t.qty * self.partial_pct, 6)
                    if pqty > 0:
                        ep = (close * (1 - COST_PER_SIDE) if t.direction == "long"
                              else close * (1 + COST_PER_SIDE))
                        pp = ((ep - t.entry_price) * pqty if t.direction == "long"
                              else (t.entry_price - ep) * pqty)
                        portfolio     += pp
                        t.qty         -= pqty
                        t.partial_done = True

                # Tier 3: Trailing stop
                if self.use_trailing and profit_r_high >= self.trail_trigger_r:
                    td = risk_dist * self.trail_distance_r
                    if t.direction == "long":
                        new_ts = high - td
                        if new_ts > t.trail_stop:
                            t.trail_stop = round(new_ts, 4)
                    else:
                        new_ts = low + td
                        if t.trail_stop == 0 or new_ts < t.trail_stop:
                            t.trail_stop = round(new_ts, 4)

                # Determine active stop
                active_sl = t.trail_stop if t.trail_stop > 0 else t.stop_loss
                reason    = None
                exit_ref  = 0.0

                if t.direction == "long":
                    if low <= active_sl:
                        exit_ref = active_sl
                        reason   = ("trail_stop" if t.trail_stop > 0
                                    else "breakeven" if t.breakeven_set
                                    else "stop_loss")
                    elif high >= t.take_profit:
                        exit_ref = t.take_profit
                        reason   = "take_profit"
                else:
                    if high >= active_sl:
                        exit_ref = active_sl
                        reason   = ("trail_stop" if t.trail_stop > 0
                                    else "breakeven" if t.breakeven_set
                                    else "stop_loss")
                    elif low <= t.take_profit:
                        exit_ref = t.take_profit
                        reason   = "take_profit"

                if reason is None:
                    still_open.append(t)
                    continue

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

            # ── Entry logic ───────────────────────────────────────────────────
            if len(open_trades) >= self.max_open:
                equity_curve.append(portfolio)
                continue

            if (i - last_signal_bar) < self.cooldown_bars:
                equity_curve.append(portfolio)
                continue

            if (i - last_loss_bar) < self.loss_cooldown:
                equity_curve.append(portfolio)
                continue

            # Gate 1: Volatility model must predict a significant move
            if vol_prob < self.threshold:
                skipped_thresh += 1
                equity_curve.append(portfolio)
                continue

            # Gate 2: Get direction from trend logic
            regime            = assign_regime(row)
            direction, reason = get_direction(row, regime, self.direction_method)

            if direction is None:
                skipped_dir += 1
                equity_curve.append(portfolio)
                continue

            # Gate 3: Regime validation (redundant with get_direction but explicit)
            if regime == "ranging":
                skipped_regime += 1
                equity_curve.append(portfolio)
                continue

            # Size position — ATR mult scales with volatility probability
            atr_mult = self._atr_mult_for_prob(vol_prob)
            qty, sl, tp, risk_dist = self._size_position(
                portfolio, close, atr, atr_mult, regime
            )

            if qty <= 0 or risk_dist <= 0:
                equity_curve.append(portfolio)
                continue

            if direction == "short":
                sl = close + (close - sl)
                tp = close - (tp - close)

            entry_net = (close * (1 + COST_PER_SIDE) if direction == "long"
                         else close * (1 - COST_PER_SIDE))

            t = Trade(
                symbol      = self.symbol,
                direction   = direction,
                strategy    = f"vol_model_{reason}",
                regime      = regime,
                entry_time  = row["time"],
                entry_price = round(entry_net, 4),
                qty         = qty,
                stop_loss   = sl,
                take_profit = tp,
                trail_stop  = 0.0,
                breakeven_set   = False,
                partial_done    = False,
                risk_pct    = self.risk_per_trade * 100,
                commission  = round(qty * close * ROUND_TRIP, 4),
                score       = round(vol_prob * 100, 1),
            )
            open_trades.append(t)
            last_signal_bar = i

            log.debug(
                "ENTRY %s %s  prob=%.3f  atr_mult=%.2f  sl=%.4f  tp=%.4f  [%s]",
                direction.upper(), row.get("time", ""), vol_prob,
                atr_mult, sl, tp, reason,
            )

            equity_curve.append(portfolio)

        # Force-close all open trades at last bar
        for t in open_trades:
            lc = df["close"].iloc[-1]
            if t.direction == "long":
                pnl = (lc * (1 - COST_PER_SIDE) - t.entry_price) * t.qty
            else:
                pnl = (t.entry_price - lc * (1 + COST_PER_SIDE)) * t.qty
            cost         = t.qty * t.entry_price
            t.exit_time  = df["time"].iloc[-1]
            t.exit_price = round(lc, 4)
            t.pnl        = round(pnl, 4)
            t.pnl_pct    = round((pnl / cost) * 100, 3) if cost else 0
            t.exit_reason= "end_of_data"
            portfolio   += pnl
            trades.append(t)

        log.info("Signals skipped — below threshold: %d | ranging regime: %d | no direction: %d",
                 skipped_thresh, skipped_regime, skipped_dir)

        return self._compute_stats(trades, equity_curve, portfolio, df)

    def _compute_stats(
        self,
        trades:       list[Trade],
        equity_curve: list[float],
        final_cap:    float,
        df:           pd.DataFrame,
    ) -> BacktestResult:
        equity = np.array(equity_curve, dtype=float)
        total_ret = (final_cap - self.initial_capital) / self.initial_capital * 100

        start = df["time"].iloc[0]
        end   = df["time"].iloc[-1]
        if hasattr(start, "to_pydatetime"):
            start, end = start.to_pydatetime(), end.to_pydatetime()

        years  = max((end - start).days / 365, 0.001)
        cagr   = ((final_cap / max(self.initial_capital, 1)) ** (1 / years) - 1) * 100

        ppy    = {"1m": 525_600, "5m": 105_120, "15m": 35_040,
                  "1h": 8_760,   "4h": 2_190,   "1d": 365}.get(self.timeframe, 2_190)
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

        pcts     = [t.pnl_pct for t in trades]
        hold_h   = [(t.exit_time - t.entry_time).total_seconds() / 3600
                    for t in trades if t.exit_time and t.entry_time]

        return BacktestResult(
            symbol=self.symbol, timeframe=self.timeframe,
            strategy="vol_model",
            start_date=start, end_date=end,
            initial_capital=self.initial_capital,
            final_capital=round(final_cap, 2),
            total_return_pct=round(total_ret, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 3),
            sortino=round(sortino, 3),
            calmar=round(calmar, 3),
            max_drawdown_pct=round(max_dd, 2),
            win_rate_pct=round(wr, 1),
            profit_factor=round(pf, 3),
            total_trades=len(trades),
            avg_trade_pct=round(float(np.mean(pcts)), 3) if pcts else 0,
            avg_win_pct=round(float(np.mean([t.pnl_pct for t in won])), 3) if won else 0,
            avg_loss_pct=round(float(np.mean([t.pnl_pct for t in lost])), 3) if lost else 0,
            best_trade_pct=round(max(pcts, default=0), 3),
            worst_trade_pct=round(min(pcts, default=0), 3),
            avg_hold_hours=round(float(np.mean(hold_h)), 1) if hold_h else 0,
            trades=trades,
            equity_curve=list(equity_curve),
        )


# ── Data loading ───────────────────────────────────────────────────────────────

def load_enriched_csv(csv_path: str) -> pd.DataFrame:
    """
    Load alpha-enriched CSV. If it has OHLCV but no technical features,
    compute them. If it already has features (from alpha pipeline), use as-is
    and only compute missing technical indicators.
    """
    log.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.sort_values("time").reset_index(drop=True)
    log.info("Loaded %d rows", len(df))

    # Always recompute technical features to ensure macd_hist_prev etc. exist
    ohlcv_cols = ["time", "open", "high", "low", "close", "volume"]
    ohlcv_only = df[ohlcv_cols].copy()
    tech_df    = build_features(ohlcv_only)

    # Merge: tech_df columns take priority over anything already in df
    tech_new_cols = [c for c in tech_df.columns if c not in ohlcv_cols]
    for col in tech_new_cols:
        df[col] = tech_df[col].values

    # Add regime one-hot
    for r in ["ranging", "trending_bull", "trending_bear", "high_volatility"]:
        df[f"regime_{r}"] = df.apply(
            lambda row: 1 if assign_regime(row) == r else 0, axis=1
        )

    log.info("Features ready. Columns: %d", len(df.columns))
    return df


# ── Results printer ────────────────────────────────────────────────────────────

def print_result(r: BacktestResult, title: str = "") -> None:
    sep = "═" * 64
    passed = (
        r.total_trades     >= 20 and
        r.sharpe           >= 0.5 and
        r.max_drawdown_pct <= 30.0 and
        r.profit_factor    >= 1.0 and
        r.win_rate_pct     >= 40.0
    )
    verdict = "✅ PASS" if passed else "❌ FAIL"

    print(f"\n{sep}")
    print(f"  {title or 'VOL MODEL BACKTEST'}  —  {r.symbol}  {r.timeframe}")
    print(f"  {r.start_date.date()} → {r.end_date.date()}   {verdict}")
    print(sep)
    print(f"  {'Capital':32}  ${r.initial_capital:>10,.2f} → ${r.final_capital:>10,.2f}")
    print(f"  {'Total return':32}  {r.total_return_pct:>+10.2f}%")
    print(f"  {'CAGR':32}  {r.cagr_pct:>+10.2f}%")
    print()

    def row(lbl, val, thr, op=">=", unit=""):
        ok = val >= thr if op == ">=" else val <= thr
        return f"  {'✓' if ok else '✗'}  {lbl:30}  {val:>8.3f}{unit}   ({op}{thr}{unit})"

    print(row("Sharpe ratio",       r.sharpe,           0.5))
    print(row("Sortino ratio",      r.sortino,          0.7))
    print(row("Calmar ratio",       r.calmar,           0.8))
    print(row("Max drawdown",       r.max_drawdown_pct, 30.0, "<=", "%"))
    print(row("Win rate",           r.win_rate_pct,     40.0, ">=", "%"))
    print(row("Profit factor",      r.profit_factor,    1.0))
    print()
    print(f"  {'Total trades':32}  {r.total_trades:>10}")
    print(f"  {'Avg trade':32}  {r.avg_trade_pct:>+9.3f}%")
    print(f"  {'Avg win':32}  {r.avg_win_pct:>+9.3f}%")
    print(f"  {'Avg loss':32}  {r.avg_loss_pct:>+9.3f}%")
    print(f"  {'Best trade':32}  {r.best_trade_pct:>+9.3f}%")
    print(f"  {'Worst trade':32}  {r.worst_trade_pct:>+9.3f}%")
    print(f"  {'Avg hold':32}  {r.avg_hold_hours:>9.1f} hrs")

    if r.trades:
        from collections import Counter
        rc = Counter(t.regime    for t in r.trades)
        ec = Counter(t.exit_reason.split("_partial")[0] for t in r.trades)
        print(f"\n  Trades by regime:  " +
              "  ".join(f"{k}={v}" for k, v in sorted(rc.items())))
        print(f"  Exit reasons:      " +
              "  ".join(f"{k}={v}" for k, v in sorted(ec.items())))

    print(sep)


def save_trades(r: BacktestResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_time","exit_time","symbol","direction","strategy",
                    "regime","vol_score","entry_price","exit_price",
                    "qty","stop_loss","trail_stop",
                    "pnl","pnl_pct","exit_reason","commission"])
        for t in r.trades:
            w.writerow([t.entry_time, t.exit_time, t.symbol, t.direction,
                        t.strategy, t.regime, t.score,
                        t.entry_price, t.exit_price, t.qty,
                        t.stop_loss, t.trail_stop,
                        t.pnl, t.pnl_pct, t.exit_reason, t.commission])
    log.info("Trades saved → %s", path)


# ── Comparison run (filtered vs unfiltered) ───────────────────────────────────

def run_unfiltered_comparison(
    df: pd.DataFrame,
    feat_cols: list[str],
    args: argparse.Namespace,
) -> None:
    """
    Run an unfiltered version (all trending bars get entered) for comparison.
    Shows exactly how much the volatility filter is adding.
    """
    from backtest.engine import BacktestEngine, STRATEGY_SIGNALS

    # Create a simple trend-follow run with no ML filter
    engine = BacktestEngine(
        symbol=args.symbol, timeframe=args.timeframe,
        strategy="trend_follow",
        initial_capital=args.capital,
        risk_per_trade=args.risk / 100,
        atr_mult=args.base_atr_mult,
        rr=args.rr,
        use_trailing=not args.no_trailing,
        be_trigger_r=0.75, be_buffer_r=0.05,
        use_partial=True, partial_r=1.5, partial_pct=0.4,
        trail_trigger_r=2.0, trail_distance_r=1.5,
        cooldown_bars=args.cooldown,
        loss_cooldown=args.loss_cooldown,
    )
    r_unfilt = engine.run(df.copy())
    print_result(r_unfilt, "Unfiltered Trend Follow (no ML)")

    return r_unfilt


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Volatility-model-driven backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python ml/backtest_with_volatility.py \\
      --csv ml/data/BTC_USDT_4h_alpha.csv \\
      --model ml/models/BTC_USDT_4h_alpha_volatility.json \\
      --meta  ml/models/BTC_USDT_4h_alpha_volatility_meta.json

  # Higher threshold (fewer, higher-quality trades)
  python ml/backtest_with_volatility.py \\
      --csv ml/data/ETH_USDT_4h_alpha.csv \\
      --model ml/models/ETH_USDT_4h_alpha_volatility.json \\
      --meta  ml/models/ETH_USDT_4h_alpha_volatility_meta.json \\
      --threshold 0.68

  # Compare ML-filtered vs unfiltered
  python ml/backtest_with_volatility.py \\
      --csv ml/data/ETH_USDT_4h_alpha.csv \\
      --model ml/models/ETH_USDT_4h_alpha_volatility.json \\
      --meta  ml/models/ETH_USDT_4h_alpha_volatility_meta.json \\
      --compare --save-trades
        """
    )
    p.add_argument("--csv",      required=True,
                   help="Alpha-enriched CSV from fetch_alpha_features.py")
    p.add_argument("--model",    required=True,
                   help="Path to _volatility.json model file")
    p.add_argument("--meta",     required=True,
                   help="Path to _volatility_meta.json file")
    p.add_argument("--symbol",   default="UNKNOWN")
    p.add_argument("--timeframe",default="4h",
                   choices=["1m","5m","15m","1h","4h","1d"])
    p.add_argument("--capital",  type=float, default=10_000.0)
    p.add_argument("--risk",     type=float, default=1.0,
                   help="Risk %% per trade (default 1.0)")
    p.add_argument("--threshold",type=float, default=None,
                   help="Override model threshold (default: use recommended from meta)")
    p.add_argument("--base-atr-mult", type=float, default=1.8,
                   help="ATR multiplier at threshold probability (default 1.8)")
    p.add_argument("--max-atr-mult",  type=float, default=2.5,
                   help="ATR multiplier at prob=1.0 (default 2.5)")
    p.add_argument("--rr",       type=float, default=2.5,
                   help="Take profit R:R ratio (default 2.5)")
    p.add_argument("--direction-method", default="trend",
                   choices=["trend", "regime", "macd"],
                   help="How to determine trade direction (default: trend)")
    p.add_argument("--no-trailing",  action="store_true")
    p.add_argument("--no-partial",   action="store_true")
    p.add_argument("--cooldown",     type=int, default=6)
    p.add_argument("--loss-cooldown",type=int, default=12)
    p.add_argument("--compare",      action="store_true",
                   help="Also run unfiltered comparison and print side-by-side")
    p.add_argument("--save-trades",  action="store_true")
    args = p.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        log.error("Model not found: %s", args.model)
        log.error("Run: python ml/train_volatility_model.py --data <your_vol_training.csv>")
        sys.exit(1)

    log.info("Loading model: %s", args.model)
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    with open(args.meta) as f:
        meta = json.load(f)

    feat_cols = meta.get("feature_cols", [])
    threshold = args.threshold or meta.get("recommended_threshold", 0.60)

    log.info("Model AUC (training): %.4f  |  Threshold: %.2f",
             meta.get("auc_roc", 0), threshold)
    log.info("Top 5 features: %s", meta.get("top5_features", []))

    # ── Load and prepare data ──────────────────────────────────────────────────
    df = load_enriched_csv(args.csv)

    # ── Run volatility-driven backtest ─────────────────────────────────────────
    engine = VolatilityBacktestEngine(
        model            = model,
        feat_cols        = feat_cols,
        threshold        = threshold,
        symbol           = args.symbol,
        timeframe        = args.timeframe,
        initial_capital  = args.capital,
        risk_per_trade   = args.risk / 100,
        base_atr_mult    = args.base_atr_mult,
        max_atr_mult     = args.max_atr_mult,
        rr               = args.rr,
        direction_method = args.direction_method,
        use_trailing     = not args.no_trailing,
        use_partial      = not args.no_partial,
        cooldown_bars    = args.cooldown,
        loss_cooldown    = args.loss_cooldown,
    )

    log.info("Running backtest…")
    result = engine.run(df.copy())
    print_result(result, f"Volatility Model — {args.symbol} {args.timeframe}")

    # ── Optional comparison ────────────────────────────────────────────────────
    if args.compare:
        print("\n  Running unfiltered comparison…")
        run_unfiltered_comparison(df.copy(), feat_cols, args)

        improvement = result.total_return_pct - (
            result.total_return_pct  # placeholder — actual comparison printed separately
        )
        print(f"\n  Summary:")
        print(f"  {'Metric':25}  {'Vol Model':>12}  {'Unfiltered':>12}")
        print(f"  {'─'*52}")
        # Note: comparison metrics printed by print_result above
        print(f"  (See both result blocks above for full comparison)")

    # ── Save trades ────────────────────────────────────────────────────────────
    if args.save_trades and result.trades:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = os.path.splitext(os.path.basename(args.csv))[0]
        path = f"backtest/output/{slug}_vol_bt_{ts}_trades.csv"
        save_trades(result, path)

    # ── Final guidance ─────────────────────────────────────────────────────────
    print(f"\n  Next steps:")
    if result.total_return_pct > 0 and result.profit_factor >= 1.0:
        print(f"  ✅ Positive return with PF >= 1.0")
        print(f"  1. Run walk-forward on this backtest:")
        print(f"     Split the alpha CSV manually into 3 date ranges and run each")
        print(f"  2. Start paper trading with TRADING_MODE=paper in docker-compose")
        print(f"     Update services/strategy/main.py to call the volatility model")
        print(f"  3. Target 8 weeks paper trading before live capital")
    else:
        print(f"  Threshold sensitivity test — try higher threshold:")
        print(f"  python ml/backtest_with_volatility.py \\")
        print(f"    --csv {args.csv} --model {args.model} --meta {args.meta} \\")
        print(f"    --threshold {min(threshold + 0.05, 0.80):.2f}")
        print(f"\n  Or test on altcoins (SOL/USDT recommended next):")
        print(f"  python scripts/load_history.py --symbol SOL/USDT --timeframe 4h --years 3")


if __name__ == "__main__":
    main()
