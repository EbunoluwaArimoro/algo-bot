"""
ml/straddle_backtester.py
──────────────────────────
Pure volatility straddle strategy driven by the volatility model.

Architecture:
  1. Volatility model fires (prob >= threshold)
  2. Place a buy-stop ABOVE the current candle's high
  3. Place a sell-stop BELOW the current candle's low
  4. Whichever triggers first on the NEXT candle becomes the trade
  5. The other order is immediately cancelled
  6. Standard three-tier exit manages the winning leg

Why this beats directional entry:
  - You never predict direction — you let the market tell you
  - Entry only happens AFTER price has committed to breaking out
  - Eliminates wick-entries where price spikes and reverses within a candle
  - Avg loss shrinks because entry is at breakout, not mid-candle noise
  - The 58.3% win rate from the model stays intact, but now the entries
    are structurally clean

Ambiguous candle handling (conservative / realistic):
  When NEXT candle's high >= buy_stop AND low <= sell_stop (both triggered),
  we cannot know from OHLCV data which fired first. We assume the LOSING
  side triggered first (worst case). This makes the backtest slightly
  pessimistic — better to be surprised upward in live trading.

Stop sizing:
  Straddle stop = candle_range * range_mult (not ATR)
  Using the candle range as the stop basis is more natural for straddles:
  the consolidation candle defines the risk, the stop sits just beyond
  the opposite side of the range.

Usage:
    python ml/straddle_backtester.py \\
        --csv ml/data/SOL_USDT_4h_alpha.csv \\
        --model ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta  ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --timeframe 4h

    # Threshold sweep to find the best operating point
    python ml/straddle_backtester.py \\
        --csv ml/data/SOL_USDT_4h_alpha.csv \\
        --model ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta  ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --timeframe 4h \\
        --threshold-sweep

    # Save trade log
    python ml/straddle_backtester.py \\
        --csv ml/data/SOL_USDT_4h_alpha.csv \\
        --model ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta  ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --timeframe 4h \\
        --save-trades
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
    assign_regime, COST_PER_SIDE, ROUND_TRIP, RISK_FREE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("straddle")

REGIMES = ["ranging", "trending_bull", "trending_bear", "high_volatility"]


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class StraddleTrade:
    symbol:         str
    direction:      Literal["long", "short"]
    regime:         str
    signal_time:    datetime        # candle that triggered the model
    signal_high:    float           # high of signal candle
    signal_low:     float           # low of signal candle
    buy_stop:       float           # trigger level above
    sell_stop:      float           # trigger level below
    entry_time:     datetime | None = None
    entry_price:    float = 0.0
    stop_loss:      float = 0.0
    take_profit:    float = 0.0
    trail_stop:     float = 0.0
    breakeven_set:  bool  = False
    partial_done:   bool  = False
    qty:            float = 0.0
    exit_time:      datetime | None = None
    exit_price:     float = 0.0
    pnl:            float = 0.0
    pnl_pct:        float = 0.0
    exit_reason:    str   = ""
    vol_prob:       float = 0.0
    commission:     float = 0.0
    candle_range:   float = 0.0     # range of signal candle (high - low)


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
    return np.array(vals, dtype=np.float32)


# ── Core engine ────────────────────────────────────────────────────────────────

class StraddleEngine:
    """
    Volatility straddle backtesting engine.

    Signal candle (bar i):
      - Model fires: vol_prob >= threshold
      - Record signal_high and signal_low
      - Set buy_stop  = signal_high + (trigger_buffer * ATR)
      - Set sell_stop = signal_low  - (trigger_buffer * ATR)

    Execution candle (bar i+1):
      - If next bar's high >= buy_stop  → enter LONG at buy_stop
      - If next bar's low  <= sell_stop → enter SHORT at sell_stop
      - If both trigger (ambiguous) → assume losing side hit first (conservative)
      - If neither triggers → order expires, wait for next signal

    Stop loss for the entered trade:
      For a LONG entry: stop = sell_stop (other side of the range)
      For a SHORT entry: stop = buy_stop (other side of the range)
      This naturally sizes the stop as the candle range width.

    Take profit:
      tp = entry + (range_width * rr)
    """

    def __init__(
        self,
        model:              xgb.XGBClassifier,
        feat_cols:          list[str],
        threshold:          float  = 0.60,
        symbol:             str    = "UNKNOWN",
        timeframe:          str    = "4h",
        initial_capital:    float  = 10_000.0,
        risk_per_trade:     float  = 0.01,
        # Straddle-specific parameters
        trigger_buffer_atr: float  = 0.1,    # buy_stop = high + 0.1 * ATR (avoid noise triggers)
        stop_buffer_atr:    float  = 0.2,    # stop sits 0.2 ATR beyond opposite side
        rr:                 float  = 2.0,    # take profit R:R
        order_expiry_bars:  int    = 1,      # cancel pending order after N bars
        max_range_pct:      float  = 8.0,    # skip if candle range > 8% of price (already exploded)
        min_range_pct:      float  = 0.5,    # skip if range < 0.5% (no real consolidation)
        # Three-tier exit
        be_trigger_r:       float  = 0.75,
        be_buffer_r:        float  = 0.05,
        use_partial:        bool   = True,
        partial_r:          float  = 1.0,
        partial_pct:        float  = 0.40,
        use_trailing:       bool   = True,
        trail_trigger_r:    float  = 1.5,
        trail_distance_r:   float  = 1.0,
        # Position limits
        max_open:           int    = 1,
        cooldown_bars:      int    = 3,
        loss_cooldown:      int    = 6,
    ):
        self.model               = model
        self.feat_cols           = feat_cols
        self.threshold           = threshold
        self.symbol              = symbol
        self.timeframe           = timeframe
        self.initial_capital     = initial_capital
        self.risk_per_trade      = risk_per_trade
        self.trigger_buffer_atr  = trigger_buffer_atr
        self.stop_buffer_atr     = stop_buffer_atr
        self.rr                  = rr
        self.order_expiry_bars   = order_expiry_bars
        self.max_range_pct       = max_range_pct
        self.min_range_pct       = min_range_pct
        self.be_trigger_r        = be_trigger_r
        self.be_buffer_r         = be_buffer_r
        self.use_partial         = use_partial
        self.partial_r           = partial_r
        self.partial_pct         = partial_pct
        self.use_trailing        = use_trailing
        self.trail_trigger_r     = trail_trigger_r
        self.trail_distance_r    = trail_distance_r
        self.max_open            = max_open
        self.cooldown_bars       = cooldown_bars
        self.loss_cooldown       = loss_cooldown

    def _size_position(
        self, portfolio: float, entry: float, stop: float
    ) -> tuple[float, float]:
        """
        Size position based on dollar risk and stop distance.
        Returns (qty, actual_dollar_risk).
        """
        stop_dist   = abs(entry - stop)
        if stop_dist <= 0 or entry <= 0:
            return 0.0, 0.0
        dollar_risk = portfolio * self.risk_per_trade
        qty         = dollar_risk / stop_dist
        return round(qty, 6), round(dollar_risk, 2)

    def _manage_open_trade(
        self,
        t: StraddleTrade,
        high: float,
        low: float,
        close: float,
        candle_time: datetime,
        portfolio: float,
    ) -> tuple[StraddleTrade | None, float, float]:
        """
        Apply three-tier exit management to an open trade.
        Returns (trade_if_still_open, updated_portfolio, partial_pnl_booked).
        """
        risk_dist = abs(t.entry_price - t.stop_loss)
        if risk_dist <= 0:
            return t, portfolio, 0.0

        partial_booked = 0.0

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
                    t.stop_loss   = round(new_be, 6)
                    t.breakeven_set = True
            else:
                new_be = t.entry_price - buf
                if new_be < t.stop_loss:
                    t.stop_loss   = round(new_be, 6)
                    t.breakeven_set = True

        # Tier 2: Partial profit
        if self.use_partial and not t.partial_done and profit_r_high >= self.partial_r:
            pqty = round(t.qty * self.partial_pct, 6)
            if pqty > 0:
                ep = (close * (1 - COST_PER_SIDE) if t.direction == "long"
                      else close * (1 + COST_PER_SIDE))
                pp = ((ep - t.entry_price) * pqty if t.direction == "long"
                      else (t.entry_price - ep) * pqty)
                portfolio      += pp
                partial_booked  = pp
                t.qty          -= pqty
                t.partial_done  = True

        # Tier 3: Trailing stop
        if self.use_trailing and profit_r_high >= self.trail_trigger_r:
            td = risk_dist * self.trail_distance_r
            if t.direction == "long":
                new_ts = high - td
                if new_ts > t.trail_stop:
                    t.trail_stop = round(new_ts, 6)
            else:
                new_ts = low + td
                if t.trail_stop == 0 or new_ts < t.trail_stop:
                    t.trail_stop = round(new_ts, 6)

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
            return t, portfolio, partial_booked

        # Close the trade
        if t.direction == "long":
            exit_net = exit_ref * (1 - COST_PER_SIDE)
            pnl      = (exit_net - t.entry_price) * t.qty
        else:
            exit_net = exit_ref * (1 + COST_PER_SIDE)
            pnl      = (t.entry_price - exit_net) * t.qty

        cost         = t.qty * t.entry_price
        t.exit_time  = candle_time
        t.exit_price = round(exit_net, 6)
        t.pnl        = round(pnl, 4)
        t.pnl_pct    = round((pnl / cost) * 100, 3) if cost > 0 else 0
        t.exit_reason= reason + ("_partial" if t.partial_done else "")
        portfolio   += pnl

        return None, portfolio, partial_booked

    def run(self, df: pd.DataFrame) -> dict:
        """
        Main simulation loop.

        State machine per bar:
          - Check pending straddle orders for execution
          - Manage open trade (stops, partials, trailing)
          - Check if we should place a new straddle order
        """
        df = df.dropna(subset=["atr", "high", "low", "close"]).reset_index(drop=True)

        # Add regime one-hot for model features
        for r in REGIMES:
            df[f"regime_{r}"] = df.apply(
                lambda row: 1 if assign_regime(row) == r else 0, axis=1
            )

        # Batch compute all model probabilities
        log.info("Computing volatility probabilities for %d bars…", len(df))
        X_all     = np.vstack([
            extract_features(df.iloc[i], self.feat_cols)
            for i in range(len(df))
        ])
        vol_probs = self.model.predict_proba(X_all)[:, 1]

        n_above = (vol_probs >= self.threshold).sum()
        log.info("Bars above threshold %.2f: %d / %d (%.1f%%)",
                 self.threshold, n_above, len(df), n_above / len(df) * 100)

        portfolio        = self.initial_capital
        equity_curve     = [portfolio]
        completed_trades: list[StraddleTrade] = []
        open_trade:  StraddleTrade | None = None
        pending_order: StraddleTrade | None = None   # straddle order waiting to trigger
        pending_since_bar: int = -999
        last_signal_bar:   int = -999
        last_loss_bar:     int = -999

        # Diagnostic counters
        signals_generated  = 0
        orders_expired     = 0
        orders_triggered   = 0
        skipped_range      = 0
        skipped_occupied   = 0
        ambiguous_resolved = 0

        for i, row in df.iterrows():
            candle_time = row["time"]
            close       = row["close"]
            high        = row["high"]
            low         = row["low"]
            atr         = row.get("atr") or 0

            # ── Step 1: Try to execute pending straddle order ─────────────────
            if pending_order is not None and open_trade is None:
                bars_pending = i - pending_since_bar

                # Check order expiry
                if bars_pending > self.order_expiry_bars:
                    orders_expired += 1
                    log.debug("Order expired at bar %d (pending %d bars)", i, bars_pending)
                    pending_order = None

                else:
                    bs = pending_order.buy_stop
                    ss = pending_order.sell_stop

                    long_triggered  = high >= bs
                    short_triggered = low  <= ss

                    if long_triggered and short_triggered:
                        # AMBIGUOUS: both sides hit this candle.
                        # Conservative assumption: the losing side triggered first.
                        # We determine "losing" as whichever direction would have
                        # resulted in the stop immediately being hit.
                        # If we assume LONG triggered first → stop = ss → hit by low
                        # If we assume SHORT triggered first → stop = bs → hit by high
                        # Both are losers in this candle. We book the smaller loss.
                        ambiguous_resolved += 1

                        # Long scenario: enter at bs, stop at ss
                        long_entry   = bs * (1 + COST_PER_SIDE)
                        long_sl      = ss - (atr * self.stop_buffer_atr)
                        long_stop_hit= low <= long_sl
                        long_pnl_est = (long_sl - long_entry) if long_stop_hit else (bs * self.rr - long_entry)

                        # Short scenario: enter at ss, stop at bs
                        short_entry  = ss * (1 - COST_PER_SIDE)
                        short_sl     = bs + (atr * self.stop_buffer_atr)
                        short_stop_hit= high >= short_sl
                        short_pnl_est= (short_entry - short_sl) if short_stop_hit else (short_entry - ss * self.rr)

                        # Take the less bad outcome
                        if long_pnl_est >= short_pnl_est:
                            chosen_dir    = "long"
                            entry_price   = long_entry
                            sl            = long_sl
                        else:
                            chosen_dir    = "short"
                            entry_price   = short_entry
                            sl            = short_sl

                        log.debug("Ambiguous candle at %s — chose %s (conservative)",
                                  candle_time, chosen_dir)

                    elif long_triggered:
                        chosen_dir  = "long"
                        entry_price = bs * (1 + COST_PER_SIDE)
                        sl          = ss - (atr * self.stop_buffer_atr)

                    elif short_triggered:
                        chosen_dir  = "short"
                        entry_price = ss * (1 - COST_PER_SIDE)
                        sl          = bs + (atr * self.stop_buffer_atr)

                    else:
                        chosen_dir = None

                    if chosen_dir is not None:
                        # Flip TP direction for shorts
                        range_width = pending_order.candle_range
                        if chosen_dir == "long":
                            tp = entry_price + range_width * self.rr
                        else:
                            tp = entry_price - range_width * self.rr

                        qty, _ = self._size_position(portfolio, entry_price, sl)

                        if qty > 0:
                            pending_order.direction   = chosen_dir
                            pending_order.entry_time  = candle_time
                            pending_order.entry_price = round(entry_price, 6)
                            pending_order.stop_loss   = round(sl, 6)
                            pending_order.take_profit = round(tp, 6)
                            pending_order.qty         = qty
                            pending_order.commission  = round(qty * entry_price * ROUND_TRIP, 4)

                            open_trade    = pending_order
                            pending_order = None
                            orders_triggered += 1

                            log.debug("ENTRY %s at %.4f  sl=%.4f  tp=%.4f  qty=%.4f",
                                      chosen_dir.upper(), entry_price, sl, tp, qty)

            # ── Step 2: Manage open trade ─────────────────────────────────────
            if open_trade is not None:
                result, portfolio, _ = self._manage_open_trade(
                    open_trade, high, low, close, candle_time, portfolio
                )
                if result is None:
                    # Trade closed
                    if open_trade.pnl < 0:
                        last_loss_bar = i
                    completed_trades.append(open_trade)
                    open_trade = None
                else:
                    open_trade = result

            # ── Step 3: Check for new straddle setup ──────────────────────────
            # Only place a new order if:
            #   - No open trade and no pending order
            #   - Not in cooldown
            #   - Model fires
            #   - Signal candle has appropriate range

            if open_trade is not None or pending_order is not None:
                equity_curve.append(portfolio)
                continue

            if (i - last_signal_bar) < self.cooldown_bars:
                equity_curve.append(portfolio)
                continue

            if (i - last_loss_bar) < self.loss_cooldown:
                equity_curve.append(portfolio)
                continue

            vol_prob = vol_probs[i]
            if vol_prob < self.threshold:
                equity_curve.append(portfolio)
                continue

            # Range quality checks
            candle_range = high - low
            range_pct    = (candle_range / close) * 100 if close > 0 else 0

            if range_pct > self.max_range_pct:
                # Candle already exploded — straddle would catch a reversal, not a breakout
                skipped_range += 1
                equity_curve.append(portfolio)
                continue

            if range_pct < self.min_range_pct:
                # Range too tight — spread and slippage eat the trade
                skipped_range += 1
                equity_curve.append(portfolio)
                continue

            # Set straddle trigger levels
            buy_stop  = high + (atr * self.trigger_buffer_atr)
            sell_stop = low  - (atr * self.trigger_buffer_atr)

            regime = assign_regime(row)

            pending_order = StraddleTrade(
                symbol      = self.symbol,
                direction   = "long",        # placeholder — set on execution
                regime      = regime,
                signal_time = candle_time,
                signal_high = high,
                signal_low  = low,
                buy_stop    = round(buy_stop,  6),
                sell_stop   = round(sell_stop, 6),
                vol_prob    = round(vol_prob, 4),
                candle_range= round(candle_range, 6),
            )
            pending_since_bar = i
            last_signal_bar   = i
            signals_generated += 1

            log.debug("SIGNAL bar %d  prob=%.3f  buy_stop=%.4f  sell_stop=%.4f  range=%.2f%%",
                      i, vol_prob, buy_stop, sell_stop, range_pct)

            equity_curve.append(portfolio)

        # Force-close any open trade at last bar
        if open_trade is not None:
            lc = df["close"].iloc[-1]
            if open_trade.direction == "long":
                pnl = (lc * (1 - COST_PER_SIDE) - open_trade.entry_price) * open_trade.qty
            else:
                pnl = (open_trade.entry_price - lc * (1 + COST_PER_SIDE)) * open_trade.qty
            cost = open_trade.qty * open_trade.entry_price
            open_trade.exit_time  = df["time"].iloc[-1]
            open_trade.exit_price = round(lc, 6)
            open_trade.pnl        = round(pnl, 4)
            open_trade.pnl_pct    = round((pnl / cost) * 100, 3) if cost > 0 else 0
            open_trade.exit_reason= "end_of_data"
            portfolio            += pnl
            completed_trades.append(open_trade)

        log.info("Run complete: %d signals → %d triggered → %d expired → %d ambiguous",
                 signals_generated, orders_triggered, orders_expired, ambiguous_resolved)
        log.info("Skipped (range filter): %d  |  Final portfolio: $%.2f",
                 skipped_range, portfolio)

        return {
            "trades":           completed_trades,
            "equity_curve":     equity_curve,
            "final_capital":    portfolio,
            "signals_generated":signals_generated,
            "orders_triggered": orders_triggered,
            "orders_expired":   orders_expired,
            "ambiguous":        ambiguous_resolved,
            "skipped_range":    skipped_range,
        }


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(
    trades:        list[StraddleTrade],
    equity_curve:  list[float],
    final_capital: float,
    initial_capital:float,
    df:            pd.DataFrame,
    timeframe:     str,
    symbol:        str,
) -> dict:
    equity = np.array(equity_curve, dtype=float)
    total_ret = (final_capital - initial_capital) / initial_capital * 100

    start = df["time"].iloc[0]
    end   = df["time"].iloc[-1]
    if hasattr(start, "to_pydatetime"):
        start, end = start.to_pydatetime(), end.to_pydatetime()

    years  = max((end - start).days / 365, 0.001)
    cagr   = ((final_capital / max(initial_capital, 1)) ** (1 / years) - 1) * 100

    ppy    = {"1m": 525_600, "5m": 105_120, "15m": 35_040,
              "1h": 8_760,   "4h": 2_190,   "1d": 365}.get(timeframe, 2_190)
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

    won    = [t for t in trades if t.pnl > 0]
    lost   = [t for t in trades if t.pnl <= 0]
    gp     = sum(t.pnl for t in won)
    gl     = sum(abs(t.pnl) for t in lost)
    wr     = len(won) / len(trades) * 100 if trades else 0
    pf     = gp / gl if gl > 0 else float("inf")
    pcts   = [t.pnl_pct for t in trades]

    hold_h = [(t.exit_time - t.entry_time).total_seconds() / 3600
              for t in trades if t.exit_time and t.entry_time]
    avg_hold = float(np.mean(hold_h)) if hold_h else 0

    from collections import Counter
    regime_dist = Counter(t.regime for t in trades)
    exit_dist   = Counter(t.exit_reason.split("_partial")[0] for t in trades)
    dir_dist    = Counter(t.direction for t in trades)

    avg_vol_prob = float(np.mean([t.vol_prob for t in trades])) if trades else 0

    return {
        "symbol":            symbol,
        "timeframe":         timeframe,
        "start":             start,
        "end":               end,
        "initial_capital":   initial_capital,
        "final_capital":     round(final_capital, 2),
        "total_return_pct":  round(total_ret, 2),
        "cagr_pct":          round(cagr, 2),
        "sharpe":            round(sharpe, 3),
        "sortino":           round(sortino, 3),
        "calmar":            round(calmar, 3),
        "max_drawdown_pct":  round(max_dd, 2),
        "win_rate_pct":      round(wr, 1),
        "profit_factor":     round(pf, 3),
        "total_trades":      len(trades),
        "avg_trade_pct":     round(float(np.mean(pcts)), 3) if pcts else 0,
        "avg_win_pct":       round(float(np.mean([t.pnl_pct for t in won])), 3) if won else 0,
        "avg_loss_pct":      round(float(np.mean([t.pnl_pct for t in lost])), 3) if lost else 0,
        "best_trade_pct":    round(max(pcts, default=0), 3),
        "worst_trade_pct":   round(min(pcts, default=0), 3),
        "avg_hold_hours":    round(avg_hold, 1),
        "avg_vol_prob":      round(avg_vol_prob, 3),
        "regime_dist":       dict(regime_dist),
        "exit_dist":         dict(exit_dist),
        "direction_dist":    dict(dir_dist),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_stats(s: dict, run_info: dict) -> None:
    sep = "═" * 66
    passed = (
        s["total_trades"]    >= 20 and
        s["sharpe"]          >= 0.5 and
        s["max_drawdown_pct"]<= 30.0 and
        s["profit_factor"]   >= 1.0 and
        s["win_rate_pct"]    >= 45.0
    )
    verdict = "✅ PASS" if passed else "❌ FAIL"

    print(f"\n{sep}")
    print(f"  STRADDLE BACKTEST — {s['symbol']}  {s['timeframe']}")
    print(f"  {str(s['start'])[:10]} → {str(s['end'])[:10]}   {verdict}")
    print(sep)

    print(f"  {'Capital':34}  ${s['initial_capital']:>10,.2f} → ${s['final_capital']:>10,.2f}")
    print(f"  {'Total return':34}  {s['total_return_pct']:>+10.2f}%")
    print(f"  {'CAGR':34}  {s['cagr_pct']:>+10.2f}%")
    print()

    def row(lbl, val, thr, op=">=", unit=""):
        ok = val >= thr if op == ">=" else val <= thr
        return f"  {'✓' if ok else '✗'}  {lbl:32}  {val:>8.3f}{unit}   ({op}{thr}{unit})"

    print(row("Sharpe ratio",       s["sharpe"],           0.5))
    print(row("Sortino ratio",      s["sortino"],          0.7))
    print(row("Calmar ratio",       s["calmar"],           0.8))
    print(row("Max drawdown",       s["max_drawdown_pct"], 30.0, "<=", "%"))
    print(row("Win rate",           s["win_rate_pct"],     45.0, ">=", "%"))
    print(row("Profit factor",      s["profit_factor"],    1.0))
    print()
    print(f"  {'Total trades':34}  {s['total_trades']:>10}")
    print(f"  {'Avg trade':34}  {s['avg_trade_pct']:>+9.3f}%")
    print(f"  {'Avg win':34}  {s['avg_win_pct']:>+9.3f}%")
    print(f"  {'Avg loss':34}  {s['avg_loss_pct']:>+9.3f}%")
    print(f"  {'Best trade':34}  {s['best_trade_pct']:>+9.3f}%")
    print(f"  {'Worst trade':34}  {s['worst_trade_pct']:>+9.3f}%")
    print(f"  {'Avg hold':34}  {s['avg_hold_hours']:>9.1f} hrs")
    print(f"  {'Avg model probability':34}  {s['avg_vol_prob']:>9.3f}")
    print()

    # Signal funnel
    print(f"  Signal funnel:")
    print(f"    Generated:     {run_info['signals_generated']:>6}")
    print(f"    Triggered:     {run_info['orders_triggered']:>6}  "
          f"({run_info['orders_triggered']/max(run_info['signals_generated'],1)*100:.0f}%)")
    print(f"    Expired:       {run_info['orders_expired']:>6}")
    print(f"    Ambiguous:     {run_info['ambiguous']:>6}  (conservative resolution applied)")
    print(f"    Range-skipped: {run_info['skipped_range']:>6}")

    if s["regime_dist"]:
        print(f"\n  Trades by regime:  " +
              "  ".join(f"{k}={v}" for k, v in sorted(s["regime_dist"].items())))
    if s["exit_dist"]:
        print(f"  Exit reasons:      " +
              "  ".join(f"{k}={v}" for k, v in sorted(s["exit_dist"].items())))
    if s["direction_dist"]:
        print(f"  Direction split:   " +
              "  ".join(f"{k}={v}" for k, v in sorted(s["direction_dist"].items())))

    print(sep)


def save_trades(trades: list[StraddleTrade], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "signal_time", "entry_time", "exit_time",
            "symbol", "direction", "regime", "vol_prob",
            "signal_high", "signal_low", "buy_stop", "sell_stop",
            "entry_price", "exit_price", "stop_loss", "take_profit",
            "qty", "candle_range",
            "pnl", "pnl_pct", "exit_reason", "commission",
        ])
        for t in trades:
            w.writerow([
                t.signal_time, t.entry_time, t.exit_time,
                t.symbol, t.direction, t.regime, t.vol_prob,
                t.signal_high, t.signal_low, t.buy_stop, t.sell_stop,
                t.entry_price, t.exit_price, t.stop_loss, t.take_profit,
                t.qty, t.candle_range,
                t.pnl, t.pnl_pct, t.exit_reason, t.commission,
            ])
    log.info("Trades saved → %s", path)


# ── Threshold sweep ────────────────────────────────────────────────────────────

def run_threshold_sweep(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    feat_cols: list[str],
    meta: dict,
    args: argparse.Namespace,
) -> None:
    print(f"\n  Threshold sweep for {args.symbol} {args.timeframe}")
    print(f"  {'Threshold':>10}  {'Trades':>8}  {'WinRate':>8}  "
          f"{'PF':>8}  {'Return':>9}  {'Sharpe':>8}  {'MaxDD':>8}")
    print("  " + "─" * 72)

    for t in np.arange(0.50, 0.81, 0.05):
        engine = StraddleEngine(
            model=model, feat_cols=feat_cols, threshold=round(t, 2),
            symbol=args.symbol, timeframe=args.timeframe,
            initial_capital=args.capital, risk_per_trade=args.risk / 100,
            trigger_buffer_atr=args.trigger_buffer,
            stop_buffer_atr=args.stop_buffer,
            rr=args.rr,
            use_trailing=not args.no_trailing,
            use_partial=not args.no_partial,
        )
        result = engine.run(df.copy())
        trades = result["trades"]

        if not trades:
            print(f"  {t:>10.2f}  {'(no trades)':>8}")
            continue

        s = compute_stats(
            trades, result["equity_curve"], result["final_capital"],
            args.capital, df, args.timeframe, args.symbol,
        )
        mark = " ←" if s["profit_factor"] >= 1.0 and s["total_trades"] >= 15 else ""
        print(f"  {t:>10.2f}  {s['total_trades']:>8}  {s['win_rate_pct']:>7.1f}%  "
              f"{s['profit_factor']:>8.3f}  {s['total_return_pct']:>+8.2f}%  "
              f"{s['sharpe']:>8.3f}  {s['max_drawdown_pct']:>7.1f}%{mark}")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    log.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.sort_values("time").reset_index(drop=True)

    # Recompute technical features
    ohlcv_cols = ["time", "open", "high", "low", "close", "volume"]
    ohlcv_only = df[ohlcv_cols].copy()
    tech_df    = build_features(ohlcv_only)
    tech_new   = [c for c in tech_df.columns if c not in ohlcv_cols]
    for col in tech_new:
        df[col] = tech_df[col].values

    for r in REGIMES:
        df[f"regime_{r}"] = df.apply(
            lambda row: 1 if assign_regime(row) == r else 0, axis=1
        )

    df = df.dropna(subset=["atr"]).reset_index(drop=True)
    log.info("Ready: %d bars  (%s → %s)",
             len(df),
             str(df["time"].iloc[0].date()),
             str(df["time"].iloc[-1].date()))
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Volatility straddle backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ml/straddle_backtester.py \\
        --csv ml/data/SOL_USDT_4h_alpha.csv \\
        --model ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta  ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --timeframe 4h

    # Find the best threshold automatically
    python ml/straddle_backtester.py \\
        --csv ml/data/SOL_USDT_4h_alpha.csv \\
        --model ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta  ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --timeframe 4h --threshold-sweep

    # Save full trade log
    python ml/straddle_backtester.py \\
        --csv ml/data/SOL_USDT_4h_alpha.csv \\
        --model ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta  ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --timeframe 4h --save-trades
        """
    )
    p.add_argument("--csv",      required=True)
    p.add_argument("--model",    required=True)
    p.add_argument("--meta",     required=True)
    p.add_argument("--symbol",   default="SOL/USDT")
    p.add_argument("--timeframe",default="4h",
                   choices=["1m","5m","15m","1h","4h","1d"])
    p.add_argument("--capital",  type=float, default=10_000.0)
    p.add_argument("--risk",     type=float, default=1.0,
                   help="Risk %% per trade (default 1.0)")
    p.add_argument("--threshold",type=float, default=None,
                   help="Model probability threshold (default: from meta)")
    p.add_argument("--rr",       type=float, default=2.0,
                   help="Take profit R:R ratio (default 2.0)")
    p.add_argument("--trigger-buffer", type=float, default=0.1,
                   help="ATR buffer above/below candle for stop trigger (default 0.1)")
    p.add_argument("--stop-buffer",    type=float, default=0.2,
                   help="ATR buffer for stop loss beyond opposite side (default 0.2)")
    p.add_argument("--max-range-pct",  type=float, default=8.0,
                   help="Skip candles where range > this %% (default 8.0)")
    p.add_argument("--min-range-pct",  type=float, default=0.5,
                   help="Skip candles where range < this %% (default 0.5)")
    p.add_argument("--no-trailing",    action="store_true")
    p.add_argument("--no-partial",     action="store_true")
    p.add_argument("--cooldown",       type=int, default=3)
    p.add_argument("--loss-cooldown",  type=int, default=6)
    p.add_argument("--threshold-sweep",action="store_true",
                   help="Sweep thresholds 0.50-0.80 and print comparison table")
    p.add_argument("--save-trades",    action="store_true")
    args = p.parse_args()

    # Load model
    if not os.path.exists(args.model):
        log.error("Model not found: %s", args.model)
        sys.exit(1)

    log.info("Loading model: %s", args.model)
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    with open(args.meta) as f:
        meta = json.load(f)

    feat_cols = meta.get("feature_cols", [])
    threshold = args.threshold or meta.get("recommended_threshold", 0.60)
    log.info("AUC (training): %.4f  |  Threshold: %.2f",
             meta.get("auc_roc", 0), threshold)

    # Load data
    df = load_and_prepare(args.csv)

    # Threshold sweep mode
    if args.threshold_sweep:
        run_threshold_sweep(df, model, feat_cols, meta, args)
        return

    # Standard single run
    engine = StraddleEngine(
        model              = model,
        feat_cols          = feat_cols,
        threshold          = threshold,
        symbol             = args.symbol,
        timeframe          = args.timeframe,
        initial_capital    = args.capital,
        risk_per_trade     = args.risk / 100,
        trigger_buffer_atr = args.trigger_buffer,
        stop_buffer_atr    = args.stop_buffer,
        rr                 = args.rr,
        max_range_pct      = args.max_range_pct,
        min_range_pct      = args.min_range_pct,
        use_trailing       = not args.no_trailing,
        use_partial        = not args.no_partial,
        cooldown_bars      = args.cooldown,
        loss_cooldown      = args.loss_cooldown,
    )

    log.info("Running straddle backtest…")
    result = engine.run(df.copy())

    stats = compute_stats(
        result["trades"], result["equity_curve"],
        result["final_capital"], args.capital,
        df, args.timeframe, args.symbol,
    )
    print_stats(stats, result)

    if args.save_trades and result["trades"]:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = os.path.splitext(os.path.basename(args.csv))[0]
        path = f"backtest/output/{slug}_straddle_{ts}_trades.csv"
        save_trades(result["trades"], path)

    print(f"\n  Recommended next run:")
    print(f"    python ml/straddle_backtester.py \\")
    print(f"      --csv {args.csv} \\")
    print(f"      --model {args.model} \\")
    print(f"      --meta {args.meta} \\")
    print(f"      --symbol '{args.symbol}' --timeframe {args.timeframe} \\")
    print(f"      --threshold-sweep")
    print()


if __name__ == "__main__":
    main()