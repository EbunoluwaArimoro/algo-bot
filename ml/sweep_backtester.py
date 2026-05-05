# """
# ml/sweep_backtester.py
# ───────────────────────
# Liquidity Sweep (Wick Catcher) backtester.

# Concept:
#   The 4h volatility model identifies when a big move is imminent.
#   Instead of chasing the breakout, we become the liquidity.
#   We place limit orders in the wick zones below support and above resistance,
#   catching the stop-hunt before the real move reverses toward the range center.

# Signal flow:
#   1. 4h model fires (vol_prob >= threshold)
#   2. Measure recent support (lowest low of lookback bars) and
#      resistance (highest high of lookback bars)
#   3. Place limit buy at support - (sweep_mult * ATR)   → catch the down-wick
#      Place limit sell at resistance + (sweep_mult * ATR) → catch the up-wick
#   4. Orders rest for window_bars candles (15m or 4h, configurable)
#   5. If filled: stop loss is slightly beyond the wick extreme
#      Target: range midpoint + extension (mean-reversion back to center)
#   6. If both fill in same bar → conservative resolution (take smaller loss)
#   7. If neither fills in window → expire, wait for next signal

# Why this works:
#   - Entry at wick extreme → structurally better price than any other entry method
#   - Mean-reversion target is geometrically defined (not arbitrary R:R)
#   - Stop is tight (just beyond the wick extreme) → small losses when wrong
#   - Large wins when right (price runs from wick to opposite range boundary)
#   - 4h model ensures we only place orders during high-probability volatility windows

# Key parameters:
#   --sweep-mult     How far below support / above resistance to place limit order
#                    (in ATR units, default 0.3 — catches moderate wicks)
#   --stop-mult      How far beyond the limit price to place the stop loss
#                    (in ATR units, default 0.5)
#   --target-mode    'midpoint'  = target is the range midpoint
#                    'opposite'  = target is the opposite wick boundary
#                    'rr'        = static R:R ratio (fallback)
#   --lookback       Bars to look back for support/resistance (default 10)
#   --window-bars    How many bars the limit orders stay open (default 8)
#   --use-15m        Use 15m bars for order execution (recommended)
#                    Without this flag, orders execute on 4h bars

# Usage:
#     # 15m execution (recommended)
#     python ml/sweep_backtester.py \\
#         --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
#         --csv-15m backtest/data/SOL_USDT_15m.csv \\
#         --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
#         --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
#         --symbol  "SOL/USDT" --use-15m --threshold-sweep

#     # 4h-only execution (faster iteration, less precise)
#     python ml/sweep_backtester.py \\
#         --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
#         --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
#         --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
#         --symbol  "SOL/USDT" --threshold-sweep

#     # Save trade log for analysis
#     python ml/sweep_backtester.py \\
#         --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
#         --csv-15m backtest/data/SOL_USDT_15m.csv \\
#         --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
#         --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
#         --symbol  "SOL/USDT" --use-15m --save-trades
# """

# import argparse
# import csv
# import json
# import logging
# import math
# import os
# import sys
# from collections import Counter
# from dataclasses import dataclass
# from datetime import datetime, timedelta
# from typing import Literal

# import numpy as np
# import pandas as pd

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# try:
#     import xgboost as xgb
# except ImportError:
#     print("Run: pip install xgboost")
#     sys.exit(1)

# from backtest.engine import (
#     build_features, assign_regime,
#     COST_PER_SIDE, ROUND_TRIP, RISK_FREE_RATE,
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)-8s  %(message)s",
#     datefmt="%H:%M:%S",
# )
# log = logging.getLogger("sweep")

# REGIMES = ["ranging", "trending_bull", "trending_bear", "high_volatility"]


# # ── Data classes ───────────────────────────────────────────────────────────────

# @dataclass
# class SweepOrder:
#     """Pending limit order waiting to be filled."""
#     signal_time:  datetime
#     vol_prob:     float
#     regime:       str
#     support:      float      # recent low
#     resistance:   float      # recent high
#     range_mid:    float      # (resistance + support) / 2
#     atr:          float      # ATR at signal time
#     # Limit order levels
#     limit_buy:    float      # limit buy below support (catch down-wick)
#     limit_sell:   float      # limit sell above resistance (catch up-wick)
#     # Stop and target levels (computed at fill time)
#     stop_buy:     float = 0.0   # stop for the buy order
#     stop_sell:    float = 0.0   # stop for the sell order
#     window_end:   datetime | None = None


# @dataclass
# class SweepTrade:
#     """A completed or open sweep trade."""
#     symbol:       str
#     direction:    Literal["long", "short"]
#     regime:       str
#     vol_prob:     float
#     signal_time:  datetime
#     support:      float
#     resistance:   float
#     range_mid:    float
#     range_size:   float
#     atr_at_signal:float
#     entry_time:   datetime | None = None
#     entry_price:  float = 0.0
#     stop_loss:    float = 0.0
#     take_profit:  float = 0.0
#     trail_stop:   float = 0.0
#     breakeven_set:bool  = False
#     partial_done: bool  = False
#     qty:          float = 0.0
#     exit_time:    datetime | None = None
#     exit_price:   float = 0.0
#     pnl:          float = 0.0
#     pnl_pct:      float = 0.0
#     exit_reason:  str   = ""
#     commission:   float = 0.0


# # ── Feature extraction ─────────────────────────────────────────────────────────

# def extract_features(row: pd.Series, feat_cols: list[str]) -> np.ndarray:
#     vals = []
#     for col in feat_cols:
#         val = row.get(col)
#         if val is None or (isinstance(val, float) and np.isnan(val)):
#             vals.append(0.0)
#         elif isinstance(val, bool):
#             vals.append(float(val))
#         else:
#             vals.append(float(val))
#     return np.array(vals, dtype=np.float32)


# # ── Data loading ───────────────────────────────────────────────────────────────

# def load_4h(path: str) -> pd.DataFrame:
#     log.info("Loading 4h: %s", path)
#     df = pd.read_csv(path, parse_dates=["time"])
#     if df["time"].dt.tz is None:
#         df["time"] = df["time"].dt.tz_localize("UTC")
#     df = df.sort_values("time").reset_index(drop=True)

#     ohlcv = df[["time","open","high","low","close","volume"]].copy()
#     tech  = build_features(ohlcv)
#     for col in [c for c in tech.columns
#                 if c not in ["time","open","high","low","close","volume"]]:
#         df[col] = tech[col].values

#     for r in REGIMES:
#         df[f"regime_{r}"] = df.apply(
#             lambda row, r=r: 1 if assign_regime(row) == r else 0, axis=1
#         )
#     df = df.dropna(subset=["atr"]).reset_index(drop=True)
#     log.info("4h ready: %d bars (%s → %s)",
#              len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))
#     return df


# def load_15m(path: str) -> pd.DataFrame:
#     log.info("Loading 15m: %s", path)
#     df = pd.read_csv(path, parse_dates=["time"])
#     if df["time"].dt.tz is None:
#         df["time"] = df["time"].dt.tz_localize("UTC")
#     df = df.sort_values("time").reset_index(drop=True)
#     import pandas_ta as ta
#     df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
#     df = df.dropna(subset=["atr"]).reset_index(drop=True)
#     log.info("15m ready: %d bars (%s → %s)",
#              len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))
#     return df


# def align(df_4h: pd.DataFrame, df_15m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
#     start = max(df_4h["time"].iloc[0],  df_15m["time"].iloc[0])
#     end   = min(df_4h["time"].iloc[-1], df_15m["time"].iloc[-1])
#     df_4h  = df_4h[(df_4h["time"]  >= start) & (df_4h["time"]  <= end)].reset_index(drop=True)
#     df_15m = df_15m[(df_15m["time"] >= start) & (df_15m["time"] <= end)].reset_index(drop=True)
#     log.info("Aligned: %s → %s  |  4h:%d  15m:%d",
#              str(start.date()), str(end.date()), len(df_4h), len(df_15m))
#     return df_4h, df_15m


# # ── Core engine ────────────────────────────────────────────────────────────────

# class SweepEngine:
#     """
#     Liquidity Sweep engine.

#     For each 4h signal:
#       - Compute support = min(low, lookback bars)
#       - Compute resistance = max(high, lookback bars)
#       - Place limit_buy  = support    - (sweep_mult * atr)
#       - Place limit_sell = resistance + (sweep_mult * atr)
#       - Stop for long  = limit_buy  - (stop_mult * atr)
#       - Stop for short = limit_sell + (stop_mult * atr)
#       - Target for long (midpoint mode)  = range_mid + (extension_mult * range_size)
#       - Target for short (midpoint mode) = range_mid - (extension_mult * range_size)

#     The limit orders rest for window_bars (15m bars) before expiring.
#     Execution candles check if low <= limit_buy or high >= limit_sell.
#     """

#     def __init__(
#         self,
#         model:              xgb.XGBClassifier,
#         feat_cols:          list[str],
#         threshold:          float = 0.60,
#         symbol:             str   = "SOL/USDT",
#         initial_capital:    float = 10_000.0,
#         risk_per_trade:     float = 0.01,
#         lookback:           int   = 10,       # bars to define support/resistance
#         sweep_mult:         float = 0.3,      # ATR units below support for limit buy
#         stop_mult:          float = 0.5,      # ATR units beyond limit for stop loss
#         target_mode:        str   = "midpoint",
#         extension_mult:     float = 0.5,      # how far past midpoint to target
#         rr_fallback:        float = 2.0,      # R:R if target_mode = 'rr'
#         window_bars:        int   = 8,        # 15m bars limit orders stay active
#         signal_cooldown_4h: int   = 2,
#         loss_cooldown_bars: int   = 48,       # 15m bars cooldown after loss (= 12h)
#         # Three-tier exit
#         be_trigger_r:       float = 0.5,
#         be_buffer_r:        float = 0.05,
#         use_partial:        bool  = True,
#         partial_r:          float = 1.0,
#         partial_pct:        float = 0.40,
#         use_trailing:       bool  = True,
#         trail_trigger_r:    float = 1.5,
#         trail_distance_r:   float = 1.0,
#     ):
#         self.model              = model
#         self.feat_cols          = feat_cols
#         self.threshold          = threshold
#         self.symbol             = symbol
#         self.initial_capital    = initial_capital
#         self.risk_per_trade     = risk_per_trade
#         self.lookback           = lookback
#         self.sweep_mult         = sweep_mult
#         self.stop_mult          = stop_mult
#         self.target_mode        = target_mode
#         self.extension_mult     = extension_mult
#         self.rr_fallback        = rr_fallback
#         self.window_bars        = window_bars
#         self.signal_cooldown_4h = signal_cooldown_4h
#         self.loss_cooldown_bars = loss_cooldown_bars
#         self.be_trigger_r       = be_trigger_r
#         self.be_buffer_r        = be_buffer_r
#         self.use_partial        = use_partial
#         self.partial_r          = partial_r
#         self.partial_pct        = partial_pct
#         self.use_trailing       = use_trailing
#         self.trail_trigger_r    = trail_trigger_r
#         self.trail_distance_r   = trail_distance_r
#         # Candle-close confirmation filter (the key new parameter)
#         # When True: the sweep candle must also CLOSE back inside the range.
#         # A candle that wicks below support but closes above it = genuine sweep.
#         # A candle that closes below support = potential breakdown, skip it.
#         # This single filter is expected to raise win rate by 8-12 points.
#         self.require_close_confirm = True   # on by default — disable with --no-confirm

#     def _compute_target(
#         self,
#         direction:   str,
#         entry_price: float,
#         stop_loss:   float,
#         range_mid:   float,
#         range_size:  float,
#         resistance:  float,
#         support:     float,
#     ) -> float:
#         """Compute take-profit based on target_mode."""
#         risk = abs(entry_price - stop_loss)

#         if self.target_mode == "midpoint":
#             # Target: range midpoint, extended slightly past it
#             if direction == "long":
#                 return range_mid + (range_size * self.extension_mult)
#             else:
#                 return range_mid - (range_size * self.extension_mult)

#         elif self.target_mode == "opposite":
#             # Target: the opposite boundary of the support/resistance range
#             if direction == "long":
#                 return resistance
#             else:
#                 return support

#         else:  # 'rr'
#             if direction == "long":
#                 return entry_price + risk * self.rr_fallback
#             else:
#                 return entry_price - risk * self.rr_fallback

#     def _size(self, portfolio: float, entry: float, sl: float) -> float:
#         dist = abs(entry - sl)
#         if dist <= 0 or entry <= 0:
#             return 0.0
#         return round((portfolio * self.risk_per_trade) / dist, 6)

#     def _manage(
#         self,
#         t: SweepTrade,
#         high: float, low: float, close: float,
#         bar_time: datetime,
#         portfolio: float,
#     ) -> tuple[SweepTrade | None, float]:
#         """Three-tier exit management. Returns (trade_if_open, portfolio)."""
#         risk_dist = abs(t.entry_price - t.stop_loss)
#         if risk_dist <= 0:
#             return t, portfolio

#         profit_r = ((high - t.entry_price) / risk_dist if t.direction == "long"
#                     else (t.entry_price - low) / risk_dist)

#         # Tier 1: Breakeven
#         if not t.breakeven_set and profit_r >= self.be_trigger_r:
#             buf = risk_dist * self.be_buffer_r
#             if t.direction == "long":
#                 new_be = t.entry_price + buf
#                 if new_be > t.stop_loss:
#                     t.stop_loss    = round(new_be, 6)
#                     t.breakeven_set = True
#             else:
#                 new_be = t.entry_price - buf
#                 if new_be < t.stop_loss:
#                     t.stop_loss    = round(new_be, 6)
#                     t.breakeven_set = True

#         # Tier 2: Partial profit
#         if self.use_partial and not t.partial_done and profit_r >= self.partial_r:
#             pqty = round(t.qty * self.partial_pct, 6)
#             if pqty > 0:
#                 ep = (close * (1 - COST_PER_SIDE) if t.direction == "long"
#                       else close * (1 + COST_PER_SIDE))
#                 pp = ((ep - t.entry_price) * pqty if t.direction == "long"
#                       else (t.entry_price - ep) * pqty)
#                 portfolio      += pp
#                 t.qty          -= pqty
#                 t.partial_done  = True

#         # Tier 3: Trail
#         if self.use_trailing and profit_r >= self.trail_trigger_r:
#             td = risk_dist * self.trail_distance_r
#             if t.direction == "long":
#                 new_ts = high - td
#                 if new_ts > t.trail_stop:
#                     t.trail_stop = round(new_ts, 6)
#             else:
#                 new_ts = low + td
#                 if t.trail_stop == 0 or new_ts < t.trail_stop:
#                     t.trail_stop = round(new_ts, 6)

#         active_sl = t.trail_stop if t.trail_stop > 0 else t.stop_loss
#         reason    = None
#         exit_ref  = 0.0

#         if t.direction == "long":
#             if low <= active_sl:
#                 exit_ref = active_sl
#                 reason   = ("trail_stop" if t.trail_stop > 0
#                             else "breakeven" if t.breakeven_set
#                             else "stop_loss")
#             elif high >= t.take_profit:
#                 exit_ref = t.take_profit
#                 reason   = "take_profit"
#         else:
#             if high >= active_sl:
#                 exit_ref = active_sl
#                 reason   = ("trail_stop" if t.trail_stop > 0
#                             else "breakeven" if t.breakeven_set
#                             else "stop_loss")
#             elif low <= t.take_profit:
#                 exit_ref = t.take_profit
#                 reason   = "take_profit"

#         if reason is None:
#             return t, portfolio

#         exit_net = (exit_ref * (1 - COST_PER_SIDE) if t.direction == "long"
#                     else exit_ref * (1 + COST_PER_SIDE))
#         pnl      = ((exit_net - t.entry_price) * t.qty if t.direction == "long"
#                     else (t.entry_price - exit_net) * t.qty)
#         cost     = t.qty * t.entry_price

#         t.exit_time  = bar_time
#         t.exit_price = round(exit_net, 6)
#         t.pnl        = round(pnl, 4)
#         t.pnl_pct    = round((pnl / cost) * 100, 3) if cost > 0 else 0
#         t.exit_reason= reason + ("_partial" if t.partial_done else "")
#         portfolio   += pnl

#         return None, portfolio

#     def _build_signals(self, df_4h: pd.DataFrame) -> list[dict]:
#         """
#         Score every 4h bar and build a list of signal dicts.
#         Each signal is keyed by its CLOSE time (open_time + 4h) to
#         guarantee no look-ahead when merging with the execution loop.
#         """
#         X_all     = np.vstack([
#             extract_features(df_4h.iloc[i], self.feat_cols)
#             for i in range(len(df_4h))
#         ])
#         vol_probs = self.model.predict_proba(X_all)[:, 1]

#         signals      = []
#         last_sig_idx = -999

#         for i in range(self.lookback, len(df_4h)):
#             if vol_probs[i] < self.threshold:
#                 continue
#             if (i - last_sig_idx) < self.signal_cooldown_4h:
#                 continue

#             row = df_4h.iloc[i]
#             atr = row.get("atr") or 0
#             if atr <= 0:
#                 continue

#             # Support and resistance from lookback window (EXCLUDING current bar)
#             window    = df_4h.iloc[i - self.lookback: i]
#             support   = float(window["low"].min())
#             resistance= float(window["high"].max())
#             range_size= resistance - support
#             range_mid = (resistance + support) / 2

#             # Limit order levels
#             limit_buy  = support    - (self.sweep_mult * atr)
#             limit_sell = resistance + (self.sweep_mult * atr)

#             # Stop loss levels
#             stop_buy   = limit_buy  - (self.stop_mult * atr)
#             stop_sell  = limit_sell + (self.stop_mult * atr)

#             close_time = row["time"] + timedelta(hours=4)
#             regime     = assign_regime(row)

#             signals.append({
#                 "close_time": close_time,
#                 "vol_prob":   vol_probs[i],
#                 "regime":     regime,
#                 "support":    support,
#                 "resistance": resistance,
#                 "range_mid":  range_mid,
#                 "range_size": range_size,
#                 "atr":        atr,
#                 "limit_buy":  round(limit_buy,  6),
#                 "limit_sell": round(limit_sell, 6),
#                 "stop_buy":   round(stop_buy,   6),
#                 "stop_sell":  round(stop_sell,  6),
#             })
#             last_sig_idx = i

#         log.info("4h signals: %d above threshold %.2f", len(signals), self.threshold)
#         return signals

#     def run(
#         self,
#         df_4h:  pd.DataFrame,
#         df_exec: pd.DataFrame,    # 15m or 4h depending on --use-15m
#         bars_per_4h: int = 16,    # 16 × 15m = 4h; 1 for 4h-only mode
#     ) -> dict:
#         signals  = self._build_signals(df_4h)
#         tf_delta = timedelta(minutes=15 * bars_per_4h // bars_per_4h) if bars_per_4h > 1 \
#                    else timedelta(hours=4)

#         portfolio         = self.initial_capital
#         equity_curve      = [portfolio]
#         completed_trades: list[SweepTrade] = []
#         open_trade:   SweepTrade | None  = None
#         active_order: SweepOrder | None  = None
#         sig_idx       = 0
#         last_loss_bar = -999

#         # Counters
#         cnt_signals   = 0
#         cnt_triggered = 0
#         cnt_expired   = 0
#         cnt_skip_occ  = 0
#         cnt_confirm_rejected = 0   # filtered by close-confirmation check

#         for i_ex, row_ex in df_exec.iterrows():
#             bar_time = row_ex["time"]
#             high     = row_ex["high"]
#             low      = row_ex["low"]
#             close    = row_ex["close"]

#             # ── Activate new signals whose 4h candle has closed ───────────────
#             while sig_idx < len(signals):
#                 sig = signals[sig_idx]
#                 if sig["close_time"] <= bar_time:
#                     # Only activate if we have no open trade and no pending order
#                     if open_trade is None and active_order is None:
#                         if (i_ex - last_loss_bar) >= self.loss_cooldown_bars:
#                             active_order = SweepOrder(
#                                 signal_time = sig["close_time"],
#                                 vol_prob    = sig["vol_prob"],
#                                 regime      = sig["regime"],
#                                 support     = sig["support"],
#                                 resistance  = sig["resistance"],
#                                 range_mid   = sig["range_mid"],
#                                 atr         = sig["atr"],
#                                 limit_buy   = sig["limit_buy"],
#                                 limit_sell  = sig["limit_sell"],
#                                 stop_buy    = sig["stop_buy"],
#                                 stop_sell   = sig["stop_sell"],
#                                 window_end  = sig["close_time"] + timedelta(
#                                     minutes=15 * self.window_bars
#                                 ),
#                             )
#                             cnt_signals += 1
#                         else:
#                             cnt_skip_occ += 1
#                     else:
#                         cnt_skip_occ += 1
#                     sig_idx += 1
#                 else:
#                     break

#             # ── Manage open trade ─────────────────────────────────────────────
#             if open_trade is not None:
#                 result, portfolio = self._manage(
#                     open_trade, high, low, close, bar_time, portfolio
#                 )
#                 if result is None:
#                     if open_trade.pnl < 0:
#                         last_loss_bar = i_ex
#                     completed_trades.append(open_trade)
#                     open_trade = None
#                 else:
#                     open_trade = result

#             # ── Check active limit order ──────────────────────────────────────
#             if active_order is not None and open_trade is None:

#                 # Expire check
#                 if active_order.window_end and bar_time > active_order.window_end:
#                     cnt_expired += 1
#                     active_order = None

#                 else:
#                     lb  = active_order.limit_buy
#                     ls  = active_order.limit_sell
#                     sl_b= active_order.stop_buy
#                     sl_s= active_order.stop_sell

#                     long_filled  = low  <= lb
#                     short_filled = high >= ls

#                     # ── Close confirmation filter ─────────────────────────────
#                     # This is the key filter that separates genuine sweeps
#                     # (wick through support → close back above it) from
#                     # genuine breakdowns (close below support = trend continuation).
#                     #
#                     # For a long sweep: candle wicks below limit_buy level
#                     # but must CLOSE above the support level (not just lb).
#                     # We use active_order.support as the confirmation line
#                     # because support is the structural level; limit_buy is
#                     # just our entry price which is already below support.
#                     #
#                     # For a short sweep: candle wicks above limit_sell level
#                     # but must CLOSE below the resistance level.
#                     if self.require_close_confirm:
#                         if long_filled and close < active_order.support:
#                             # Closed below support = breakdown, not a sweep
#                             cnt_confirm_rejected += 1
#                             long_filled = False
#                         if short_filled and close > active_order.resistance:
#                             # Closed above resistance = breakout, not a sweep
#                             cnt_confirm_rejected += 1
#                             short_filled = False

#                     if long_filled and short_filled:
#                         # Both filled and confirmed — take the smaller risk
#                         long_risk  = abs(lb - sl_b)
#                         short_risk = abs(ls - sl_s)
#                         chosen = "long" if long_risk <= short_risk else "short"
#                     elif long_filled:
#                         chosen = "long"
#                     elif short_filled:
#                         chosen = "short"
#                     else:
#                         chosen = None

#                     if chosen is not None:
#                         if chosen == "long":
#                             entry_price = lb * (1 + COST_PER_SIDE)
#                             sl          = sl_b
#                         else:
#                             entry_price = ls * (1 - COST_PER_SIDE)
#                             sl          = sl_s

#                         tp = self._compute_target(
#                             chosen, entry_price, sl,
#                             active_order.range_mid,
#                             active_order.resistance - active_order.support,
#                             active_order.resistance,
#                             active_order.support,
#                         )

#                         qty = self._size(portfolio, entry_price, sl)

#                         if qty > 0:
#                             open_trade = SweepTrade(
#                                 symbol        = self.symbol,
#                                 direction     = chosen,
#                                 regime        = active_order.regime,
#                                 vol_prob      = active_order.vol_prob,
#                                 signal_time   = active_order.signal_time,
#                                 support       = active_order.support,
#                                 resistance    = active_order.resistance,
#                                 range_mid     = active_order.range_mid,
#                                 range_size    = active_order.resistance - active_order.support,
#                                 atr_at_signal = active_order.atr,
#                                 entry_time    = bar_time,
#                                 entry_price   = round(entry_price, 6),
#                                 stop_loss     = round(sl, 6),
#                                 take_profit   = round(tp, 6),
#                                 trail_stop    = 0.0,
#                                 qty           = qty,
#                                 commission    = round(qty * entry_price * ROUND_TRIP, 4),
#                             )
#                             cnt_triggered += 1
#                             active_order   = None

#             equity_curve.append(portfolio)

#         # Force-close open trade
#         if open_trade is not None:
#             lc  = df_exec["close"].iloc[-1]
#             ep  = (lc * (1 - COST_PER_SIDE) if open_trade.direction == "long"
#                    else lc * (1 + COST_PER_SIDE))
#             pnl = ((ep - open_trade.entry_price) * open_trade.qty
#                    if open_trade.direction == "long"
#                    else (open_trade.entry_price - ep) * open_trade.qty)
#             cost = open_trade.qty * open_trade.entry_price
#             open_trade.exit_time   = df_exec["time"].iloc[-1]
#             open_trade.exit_price  = round(ep, 6)
#             open_trade.pnl         = round(pnl, 4)
#             open_trade.pnl_pct     = round((pnl / cost) * 100, 3) if cost > 0 else 0
#             open_trade.exit_reason = "end_of_data"
#             portfolio             += pnl
#             completed_trades.append(open_trade)

#         log.info("Signals: %d  |  Triggered: %d  |  Expired: %d  |  Skipped: %d  |  Confirm-rejected: %d",
#                  cnt_signals, cnt_triggered, cnt_expired, cnt_skip_occ, cnt_confirm_rejected)

#         return {
#             "trades":            completed_trades,
#             "equity_curve":      equity_curve,
#             "final_capital":     portfolio,
#             "cnt_signals":       cnt_signals,
#             "cnt_triggered":     cnt_triggered,
#             "cnt_expired":       cnt_expired,
#             "cnt_skipped":       cnt_skip_occ,
#             "cnt_confirm_reject":cnt_confirm_rejected,
#         }


# # ── Statistics ─────────────────────────────────────────────────────────────────

# def compute_stats(
#     trades: list[SweepTrade],
#     equity_curve: list[float],
#     final_capital: float,
#     initial_capital: float,
#     df_exec: pd.DataFrame,
#     symbol: str,
#     timeframe_label: str,
# ) -> dict:
#     equity    = np.array(equity_curve, dtype=float)
#     total_ret = (final_capital - initial_capital) / initial_capital * 100
#     start     = df_exec["time"].iloc[0]
#     end       = df_exec["time"].iloc[-1]
#     if hasattr(start, "to_pydatetime"):
#         start, end = start.to_pydatetime(), end.to_pydatetime()

#     years  = max((end - start).days / 365, 0.001)
#     cagr   = ((final_capital / max(initial_capital, 1)) ** (1 / years) - 1) * 100

#     ppy    = 35_040 if "15m" in timeframe_label else 2_190
#     rets   = np.diff(equity) / np.maximum(equity[:-1], 1e-6)
#     rets   = rets[np.isfinite(rets)]
#     rf     = RISK_FREE_RATE / ppy
#     exc    = rets - rf
#     sharpe = float(np.mean(exc) / (np.std(rets) + 1e-10) * math.sqrt(ppy))
#     dn     = rets[rets < 0]
#     sortino= float(np.mean(exc) / (np.std(dn) + 1e-10) * math.sqrt(ppy))

#     peak   = np.maximum.accumulate(equity)
#     dds    = (peak - equity) / (peak + 1e-10) * 100
#     max_dd = float(np.max(dds))
#     calmar = cagr / max(max_dd, 0.01)

#     won  = [t for t in trades if t.pnl > 0]
#     lost = [t for t in trades if t.pnl <= 0]
#     gp   = sum(t.pnl for t in won)
#     gl   = sum(abs(t.pnl) for t in lost)
#     wr   = len(won) / len(trades) * 100 if trades else 0
#     pf   = gp / gl if gl > 0 else float("inf")
#     pcts = [t.pnl_pct for t in trades]

#     hold_h = [(t.exit_time - t.entry_time).total_seconds() / 3600
#               for t in trades if t.exit_time and t.entry_time]

#     avg_win_loss_ratio = (
#         abs(float(np.mean([t.pnl_pct for t in won])) /
#             (float(np.mean([t.pnl_pct for t in lost])) + 1e-9))
#         if won and lost else 0
#     )

#     def sm(lst): return round(float(np.mean(lst)), 3) if lst else 0.0

#     return {
#         "symbol":            symbol,
#         "timeframe":         timeframe_label,
#         "start":             start,
#         "end":               end,
#         "initial_capital":   initial_capital,
#         "final_capital":     round(final_capital, 2),
#         "total_return_pct":  round(total_ret, 2),
#         "cagr_pct":          round(cagr, 2),
#         "sharpe":            round(sharpe, 3),
#         "sortino":           round(sortino, 3),
#         "calmar":            round(calmar, 3),
#         "max_drawdown_pct":  round(max_dd, 2),
#         "win_rate_pct":      round(wr, 1),
#         "profit_factor":     round(pf, 3),
#         "total_trades":      len(trades),
#         "avg_trade_pct":     sm(pcts),
#         "avg_win_pct":       sm([t.pnl_pct for t in won]),
#         "avg_loss_pct":      sm([t.pnl_pct for t in lost]),
#         "best_trade_pct":    round(max(pcts, default=0), 3),
#         "worst_trade_pct":   round(min(pcts, default=0), 3),
#         "avg_hold_hours":    sm(hold_h),
#         "win_loss_ratio":    round(avg_win_loss_ratio, 3),
#         "avg_vol_prob":      round(float(np.mean([t.vol_prob for t in trades])), 3) if trades else 0,
#         "regime_dist":       dict(Counter(t.regime for t in trades)),
#         "exit_dist":         dict(Counter(t.exit_reason.split("_partial")[0] for t in trades)),
#         "direction_dist":    dict(Counter(t.direction for t in trades)),
#     }


# # ── Reporting ──────────────────────────────────────────────────────────────────

# def print_stats(s: dict, run_info: dict) -> None:
#     sep    = "═" * 66
#     passed = (
#         s["total_trades"]    >= 15 and
#         s["profit_factor"]   >= 1.0 and
#         s["max_drawdown_pct"]<= 35.0 and
#         s["win_rate_pct"]    >= 40.0
#     )
#     print(f"\n{sep}")
#     print(f"  SWEEP BACKTEST — {s['symbol']}  ({s['timeframe']})")
#     print(f"  {str(s['start'])[:10]} → {str(s['end'])[:10]}   "
#           f"{'✅ PASS' if passed else '❌ FAIL'}")
#     print(sep)
#     print(f"  {'Capital':34}  ${s['initial_capital']:>10,.2f} → ${s['final_capital']:>10,.2f}")
#     print(f"  {'Total return':34}  {s['total_return_pct']:>+10.2f}%")
#     print(f"  {'CAGR':34}  {s['cagr_pct']:>+10.2f}%")
#     print()

#     def row(lbl, val, thr, op=">=", unit=""):
#         ok = val >= thr if op == ">=" else val <= thr
#         return f"  {'✓' if ok else '✗'}  {lbl:32}  {val:>8.3f}{unit}   ({op}{thr}{unit})"

#     print(row("Sharpe ratio",       s["sharpe"],          0.5))
#     print(row("Sortino ratio",      s["sortino"],         0.7))
#     print(row("Calmar ratio",       s["calmar"],          0.8))
#     print(row("Max drawdown",       s["max_drawdown_pct"],35.0, "<=", "%"))
#     print(row("Win rate",           s["win_rate_pct"],    40.0, ">=", "%"))
#     print(row("Profit factor",      s["profit_factor"],   1.0))
#     print(row("Win/Loss ratio",     s["win_loss_ratio"],  1.5))
#     print()
#     print(f"  {'Total trades':34}  {s['total_trades']:>10}")
#     print(f"  {'Avg trade':34}  {s['avg_trade_pct']:>+9.3f}%")
#     print(f"  {'Avg win':34}  {s['avg_win_pct']:>+9.3f}%")
#     print(f"  {'Avg loss':34}  {s['avg_loss_pct']:>+9.3f}%")
#     print(f"  {'Best trade':34}  {s['best_trade_pct']:>+9.3f}%")
#     print(f"  {'Worst trade':34}  {s['worst_trade_pct']:>+9.3f}%")
#     print(f"  {'Avg hold':34}  {s['avg_hold_hours']:>9.1f} hrs")
#     print(f"  {'Avg model prob':34}  {s['avg_vol_prob']:>9.3f}")
#     print()
#     print(f"  Signal funnel:")
#     print(f"    Signals activated:  {run_info['cnt_signals']:>6}")
#     print(f"    Limit orders filled:{run_info['cnt_triggered']:>6}  "
#           f"({run_info['cnt_triggered']/max(run_info['cnt_signals'],1)*100:.0f}%)")
#     print(f"    Orders expired:     {run_info['cnt_expired']:>6}")
#     print(f"    Confirm-rejected:   {run_info.get('cnt_confirm_reject',0):>6}  "
#           f"(wicked level but closed beyond it — skipped)")
#     print(f"    Skipped (occupied): {run_info['cnt_skipped']:>6}")
#     if s["regime_dist"]:
#         print(f"\n  Regime dist:  " + "  ".join(f"{k}={v}" for k, v in sorted(s["regime_dist"].items())))
#     if s["exit_dist"]:
#         print(f"  Exit reasons: " + "  ".join(f"{k}={v}" for k, v in sorted(s["exit_dist"].items())))
#     if s["direction_dist"]:
#         print(f"  Direction:    " + "  ".join(f"{k}={v}" for k, v in sorted(s["direction_dist"].items())))
#     print(sep)


# def save_trades(trades: list[SweepTrade], path: str) -> None:
#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#     with open(path, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow([
#             "signal_time","entry_time","exit_time",
#             "symbol","direction","regime","vol_prob",
#             "support","resistance","range_mid","range_size","atr_at_signal",
#             "entry_price","exit_price","stop_loss","take_profit",
#             "qty","pnl","pnl_pct","exit_reason","commission",
#         ])
#         for t in trades:
#             w.writerow([
#                 t.signal_time, t.entry_time, t.exit_time,
#                 t.symbol, t.direction, t.regime, t.vol_prob,
#                 t.support, t.resistance, t.range_mid, t.range_size, t.atr_at_signal,
#                 t.entry_price, t.exit_price, t.stop_loss, t.take_profit,
#                 t.qty, t.pnl, t.pnl_pct, t.exit_reason, t.commission,
#             ])
#     log.info("Trades → %s", path)


# # ── Main ───────────────────────────────────────────────────────────────────────

# def main():
#     p = argparse.ArgumentParser(
#         description="Liquidity sweep (wick catcher) backtester",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     p.add_argument("--csv-4h",   required=True)
#     p.add_argument("--csv-15m",  default=None,  help="15m CSV (required if --use-15m)")
#     p.add_argument("--model",    required=True)
#     p.add_argument("--meta",     required=True)
#     p.add_argument("--symbol",   default="SOL/USDT")
#     p.add_argument("--capital",  type=float, default=10_000.0)
#     p.add_argument("--risk",     type=float, default=1.0)
#     p.add_argument("--threshold",type=float, default=None)
#     p.add_argument("--lookback", type=int,   default=10)
#     p.add_argument("--sweep-mult",   type=float, default=0.3,
#                    help="ATR units below support for limit buy (default 0.3)")
#     p.add_argument("--stop-mult",    type=float, default=0.5,
#                    help="ATR units beyond limit for stop loss (default 0.5)")
#     p.add_argument("--target-mode",  default="midpoint",
#                    choices=["midpoint","opposite","rr"])
#     p.add_argument("--extension-mult",type=float, default=0.5)
#     p.add_argument("--rr",           type=float, default=2.0)
#     p.add_argument("--window-bars",  type=int,   default=8)
#     p.add_argument("--use-15m",      action="store_true",
#                    help="Execute on 15m bars instead of 4h bars")
#     p.add_argument("--no-trailing",  action="store_true")
#     p.add_argument("--no-partial",   action="store_true")
#     p.add_argument("--no-confirm",   action="store_true",
#                    help="Disable close-confirmation filter (fills on any wick, not just reversals)")
#     p.add_argument("--threshold-sweep", action="store_true")
#     p.add_argument("--save-trades",     action="store_true")
#     args = p.parse_args()

#     model = xgb.XGBClassifier()
#     model.load_model(args.model)
#     with open(args.meta) as f:
#         meta = json.load(f)
#     feat_cols = meta.get("feature_cols", [])
#     threshold = args.threshold or meta.get("recommended_threshold", 0.60)
#     log.info("Model AUC: %.4f  |  Threshold: %.2f", meta.get("auc_roc", 0), threshold)

#     df_4h = load_4h(args.csv_4h)

#     if args.use_15m:
#         if not args.csv_15m:
#             log.error("--use-15m requires --csv-15m")
#             sys.exit(1)
#         df_exec = load_15m(args.csv_15m)
#         df_4h, df_exec = align(df_4h, df_exec)
#         bars_per_4h   = 16
#         tf_label      = "4h→15m sweep"
#     else:
#         df_exec   = df_4h.copy()
#         bars_per_4h = 1
#         tf_label  = "4h sweep"

#     def single_run(thresh):
#         engine = SweepEngine(
#             model=model, feat_cols=feat_cols, threshold=thresh,
#             symbol=args.symbol, initial_capital=args.capital,
#             risk_per_trade=args.risk / 100,
#             lookback=args.lookback,
#             sweep_mult=args.sweep_mult,
#             stop_mult=args.stop_mult,
#             target_mode=args.target_mode,
#             extension_mult=args.extension_mult,
#             rr_fallback=args.rr,
#             window_bars=args.window_bars,
#             use_partial=not args.no_partial,
#             use_trailing=not args.no_trailing,
#         )
#         engine.require_close_confirm = not args.no_confirm
#         return engine.run(df_4h.copy(), df_exec.copy(), bars_per_4h=bars_per_4h)

#     if args.threshold_sweep:
#         print(f"\n  Sweep backtester — threshold sweep — {args.symbol}  [{tf_label}]")
#         print(f"  {'Threshold':>10}  {'Trades':>8}  {'WinRate':>8}  "
#               f"{'PF':>8}  {'Return':>9}  {'Sharpe':>8}  {'WL Ratio':>9}")
#         print("  " + "─" * 72)
#         for t in np.arange(0.50, 0.81, 0.05):
#             result = single_run(round(t, 2))
#             trades = result["trades"]
#             if not trades:
#                 print(f"  {t:>10.2f}  {'(no fills)':>8}")
#                 continue
#             s = compute_stats(trades, result["equity_curve"], result["final_capital"],
#                               args.capital, df_exec, args.symbol, tf_label)
#             mark = " ←" if s["profit_factor"] >= 1.0 and s["total_trades"] >= 15 else ""
#             print(f"  {t:>10.2f}  {s['total_trades']:>8}  {s['win_rate_pct']:>7.1f}%  "
#                   f"{s['profit_factor']:>8.3f}  {s['total_return_pct']:>+8.2f}%  "
#                   f"{s['sharpe']:>8.3f}  {s['win_loss_ratio']:>9.3f}{mark}")
#         return

#     result = single_run(threshold)
#     s      = compute_stats(result["trades"], result["equity_curve"],
#                            result["final_capital"], args.capital,
#                            df_exec, args.symbol, tf_label)
#     print_stats(s, result)

#     if args.save_trades and result["trades"]:
#         ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
#         path = f"backtest/output/SOL_USDT_sweep_{ts}_trades.csv"
#         save_trades(result["trades"], path)


# if __name__ == "__main__":
#     main()



"""
ml/sweep_backtester.py  —  v2 (Option B: intrabar fill, no look-ahead)
───────────────────────────────────────────────────────────────────────
Liquidity Sweep (Wick Catcher) — clean, executable mechanics.

What changed from v1:
  REMOVED — close-confirmation filter (was look-ahead bias: used the
             current candle's close to retroactively cancel an intrabar
             fill that had already occurred earlier in the same candle).

  ADDED   — Intrabar fill with immediate tight stop.
             When low <= limit_buy, we fill at limit_buy.
             Stop is placed immediately at limit_buy - (tight_stop_mult * ATR).
             If the same candle's low also hits the stop, we assume the stop
             fired immediately after fill (conservative / no look-ahead).
             If the wick recovers, the trade survives and the trailing
             stop system takes over.

Entry mechanic:
  1. 4h model fires → limit_buy placed below support, limit_sell above resistance
  2. Execution candle's low <= limit_buy → fill at limit_buy (long)
     OR high >= limit_sell → fill at limit_sell (short)
  3. Tight initial stop: entry ± (tight_stop_mult * ATR)
     Default: 0.4 ATR — wide enough to survive normal spread, tight enough
     to exit quickly if the wick is a genuine breakdown
  4. Once in profit by be_trigger_r, stop moves to breakeven
  5. Once in profit by trail_trigger_r, aggressive trailing stop activates

Same-candle ambiguity (conservative):
  If candle hits limit_buy AND then hits the stop in the same bar,
  we book a loss equal to (limit_buy - stop) * qty.
  We never credit a recovery that we cannot verify happened.

Key parameters:
  --sweep-mult       ATR units below support for limit_buy (default 0.4)
  --tight-stop-mult  ATR units below limit_buy for initial stop (default 0.4)
  --trail-mult       ATR units behind price for trail stop (default 1.0)
  --trail-trigger-r  R units in profit before trail activates (default 1.0)
  --target-mode      midpoint | opposite | rr

Usage:
    # 4h-only sweep (fast, good for iteration)
    python ml/sweep_backtester.py \\
        --csv-4h ml/data/SOL_USDT_4h_alpha.csv \\
        --model  ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta   ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --threshold-sweep

    # Full parameter sweep (reproduce v1 Test 2 comparison)
    python ml/sweep_backtester.py \\
        --csv-4h ml/data/SOL_USDT_4h_alpha.csv \\
        --model  ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta   ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" \\
        --sweep-mult 0.6 --tight-stop-mult 0.3 \\
        --threshold-sweep

    # 15m execution mode
    python ml/sweep_backtester.py \\
        --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
        --csv-15m backtest/data/SOL_USDT_15m.csv \\
        --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol  "SOL/USDT" --use-15m --threshold-sweep

    # Save trade log for analysis
    python ml/sweep_backtester.py \\
        --csv-4h ml/data/SOL_USDT_4h_alpha.csv \\
        --model  ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta   ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol "SOL/USDT" --sweep-mult 0.6 --tight-stop-mult 0.3 \\
        --threshold 0.60 --save-trades
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    build_features, assign_regime,
    COST_PER_SIDE, ROUND_TRIP, RISK_FREE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sweep")

REGIMES = ["ranging", "trending_bull", "trending_bear", "high_volatility"]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SweepOrder:
    """Pending limit order waiting to be filled."""
    signal_time:  datetime
    vol_prob:     float
    regime:       str
    support:      float
    resistance:   float
    range_mid:    float
    atr:          float
    limit_buy:    float
    limit_sell:   float
    window_end:   datetime | None = None


@dataclass
class SweepTrade:
    """An open or completed sweep trade."""
    symbol:          str
    direction:       Literal["long", "short"]
    regime:          str
    vol_prob:        float
    signal_time:     datetime
    support:         float
    resistance:      float
    range_mid:       float
    range_size:      float
    atr_at_signal:   float
    entry_time:      datetime | None = None
    entry_price:     float = 0.0
    stop_loss:       float = 0.0      # tight initial stop, may widen to breakeven
    take_profit:     float = 0.0
    trail_stop:      float = 0.0      # 0.0 = not yet active
    breakeven_set:   bool  = False
    partial_done:    bool  = False
    qty:             float = 0.0
    exit_time:       datetime | None = None
    exit_price:      float = 0.0
    pnl:             float = 0.0
    pnl_pct:         float = 0.0
    exit_reason:     str   = ""
    commission:      float = 0.0


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


# ── Data loading ───────────────────────────────────────────────────────────────

def load_4h(path: str) -> pd.DataFrame:
    log.info("Loading 4h: %s", path)
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.sort_values("time").reset_index(drop=True)
    ohlcv = df[["time","open","high","low","close","volume"]].copy()
    tech  = build_features(ohlcv)
    for col in [c for c in tech.columns
                if c not in ["time","open","high","low","close","volume"]]:
        df[col] = tech[col].values
    for r in REGIMES:
        df[f"regime_{r}"] = df.apply(
            lambda row, r=r: 1 if assign_regime(row) == r else 0, axis=1
        )
    df = df.dropna(subset=["atr"]).reset_index(drop=True)
    log.info("4h ready: %d bars (%s → %s)",
             len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))
    return df


def load_15m(path: str) -> pd.DataFrame:
    log.info("Loading 15m: %s", path)
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.sort_values("time").reset_index(drop=True)
    import pandas_ta as ta
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df = df.dropna(subset=["atr"]).reset_index(drop=True)
    log.info("15m ready: %d bars (%s → %s)",
             len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))
    return df


def align(df_4h: pd.DataFrame, df_15m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = max(df_4h["time"].iloc[0],  df_15m["time"].iloc[0])
    end   = min(df_4h["time"].iloc[-1], df_15m["time"].iloc[-1])
    df_4h  = df_4h[(df_4h["time"]  >= start) & (df_4h["time"]  <= end)].reset_index(drop=True)
    df_15m = df_15m[(df_15m["time"] >= start) & (df_15m["time"] <= end)].reset_index(drop=True)
    log.info("Aligned: %s → %s  |  4h:%d  15m:%d",
             str(start.date()), str(end.date()), len(df_4h), len(df_15m))
    return df_4h, df_15m


# ── Core engine ────────────────────────────────────────────────────────────────

class SweepEngine:
    """
    Liquidity sweep engine — Option B: intrabar fill, tight stop, no look-ahead.

    Fill mechanic (no look-ahead):
      When execution bar's low <= limit_buy:
        → Fill at limit_buy (limit order executed)
        → Attach stop at limit_buy - (tight_stop_mult * ATR)
        → Check same candle: if low also <= stop → stop immediately fires
          (we cannot know intrabar which happened first, so we book the loss)
        → If same candle does NOT hit stop → trade survives, trail system takes over

    This is fully executable on a real exchange:
      - Place limit order at limit_buy
      - When filled, immediately place a stop-loss order at the stop level
      - The two orders are entered sequentially with milliseconds between them
      - No future information is used at any point
    """

    def __init__(
        self,
        model:              xgb.XGBClassifier,
        feat_cols:          list[str],
        threshold:          float = 0.60,
        symbol:             str   = "SOL/USDT",
        initial_capital:    float = 10_000.0,
        risk_per_trade:     float = 0.01,
        lookback:           int   = 10,
        sweep_mult:         float = 0.4,       # ATR below support for limit_buy
        tight_stop_mult:    float = 0.4,       # ATR below limit_buy for stop
        target_mode:        str   = "opposite",
        extension_mult:     float = 0.3,       # for midpoint mode
        rr_fallback:        float = 2.0,       # for rr mode
        window_bars:        int   = 4,         # bars limit order stays open
        signal_cooldown:    int   = 2,         # 4h bars between signals
        loss_cooldown_bars: int   = 8,         # bars after a loss before next signal
        # Three-tier exit
        be_trigger_r:       float = 0.5,       # move stop to BE after 0.5R profit
        be_buffer_r:        float = 0.05,
        use_partial:        bool  = True,
        partial_r:          float = 1.0,
        partial_pct:        float = 0.40,
        use_trailing:       bool  = True,
        trail_trigger_r:    float = 1.0,       # trail activates at 1R profit
        trail_mult:         float = 1.0,       # trail sits 1 ATR behind price
    ):
        self.model              = model
        self.feat_cols          = feat_cols
        self.threshold          = threshold
        self.symbol             = symbol
        self.initial_capital    = initial_capital
        self.risk_per_trade     = risk_per_trade
        self.lookback           = lookback
        self.sweep_mult         = sweep_mult
        self.tight_stop_mult    = tight_stop_mult
        self.target_mode        = target_mode
        self.extension_mult     = extension_mult
        self.rr_fallback        = rr_fallback
        self.window_bars        = window_bars
        self.signal_cooldown    = signal_cooldown
        self.loss_cooldown_bars = loss_cooldown_bars
        self.be_trigger_r       = be_trigger_r
        self.be_buffer_r        = be_buffer_r
        self.use_partial        = use_partial
        self.partial_r          = partial_r
        self.partial_pct        = partial_pct
        self.use_trailing       = use_trailing
        self.trail_trigger_r    = trail_trigger_r
        self.trail_mult         = trail_mult

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _compute_target(
        self,
        direction:   str,
        entry_price: float,
        stop_loss:   float,
        range_mid:   float,
        range_size:  float,
        resistance:  float,
        support:     float,
    ) -> float:
        risk = abs(entry_price - stop_loss)
        if self.target_mode == "midpoint":
            if direction == "long":
                return range_mid + range_size * self.extension_mult
            else:
                return range_mid - range_size * self.extension_mult
        elif self.target_mode == "opposite":
            return resistance if direction == "long" else support
        else:  # rr
            if direction == "long":
                return entry_price + risk * self.rr_fallback
            else:
                return entry_price - risk * self.rr_fallback

    def _size(self, portfolio: float, entry: float, sl: float) -> float:
        dist = abs(entry - sl)
        if dist <= 0 or entry <= 0:
            return 0.0
        return round((portfolio * self.risk_per_trade) / dist, 6)

    # ── Trade management ───────────────────────────────────────────────────────

    def _manage(
        self,
        t:         SweepTrade,
        high:      float,
        low:       float,
        close:     float,
        atr_bar:   float,       # ATR of current execution bar (for trail sizing)
        bar_time:  datetime,
        portfolio: float,
    ) -> tuple[SweepTrade | None, float]:
        """
        Apply three-tier exit management to an open trade.
        Returns (trade_if_still_open, updated_portfolio).
        """
        risk_dist = abs(t.entry_price - t.stop_loss)
        if risk_dist <= 0:
            return t, portfolio

        profit_r_high = ((high - t.entry_price) / risk_dist if t.direction == "long"
                         else (t.entry_price - low) / risk_dist)

        # Tier 1: Breakeven stop
        if not t.breakeven_set and profit_r_high >= self.be_trigger_r:
            buf = risk_dist * self.be_buffer_r
            if t.direction == "long":
                new_be = t.entry_price + buf
                if new_be > t.stop_loss:
                    t.stop_loss    = round(new_be, 6)
                    t.breakeven_set = True
            else:
                new_be = t.entry_price - buf
                if new_be < t.stop_loss:
                    t.stop_loss    = round(new_be, 6)
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
                t.qty          -= pqty
                t.partial_done  = True

        # Tier 3: Trailing stop — uses current bar's ATR for distance
        # This means the trail widens during volatile bars and tightens during calm ones
        if self.use_trailing and profit_r_high >= self.trail_trigger_r:
            trail_dist = (atr_bar if atr_bar > 0 else risk_dist) * self.trail_mult
            if t.direction == "long":
                new_ts = high - trail_dist
                if new_ts > t.trail_stop:
                    t.trail_stop = round(new_ts, 6)
            else:
                new_ts = low + trail_dist
                if t.trail_stop == 0 or new_ts < t.trail_stop:
                    t.trail_stop = round(new_ts, 6)

        # Determine active stop level
        active_sl = t.trail_stop if t.trail_stop > 0 else t.stop_loss

        reason   = None
        exit_ref = 0.0

        if t.direction == "long":
            if low <= active_sl:
                exit_ref = active_sl
                reason   = ("trail_stop"   if t.trail_stop > 0
                            else "breakeven" if t.breakeven_set
                            else "stop_loss")
            elif high >= t.take_profit:
                exit_ref = t.take_profit
                reason   = "take_profit"
        else:
            if high >= active_sl:
                exit_ref = active_sl
                reason   = ("trail_stop"   if t.trail_stop > 0
                            else "breakeven" if t.breakeven_set
                            else "stop_loss")
            elif low <= t.take_profit:
                exit_ref = t.take_profit
                reason   = "take_profit"

        if reason is None:
            return t, portfolio

        exit_net = (exit_ref * (1 - COST_PER_SIDE) if t.direction == "long"
                    else exit_ref * (1 + COST_PER_SIDE))
        pnl      = ((exit_net - t.entry_price) * t.qty if t.direction == "long"
                    else (t.entry_price - exit_net) * t.qty)
        cost     = t.qty * t.entry_price

        t.exit_time  = bar_time
        t.exit_price = round(exit_net, 6)
        t.pnl        = round(pnl, 4)
        t.pnl_pct    = round((pnl / cost) * 100, 3) if cost > 0 else 0
        t.exit_reason= reason + ("_partial" if t.partial_done else "")
        portfolio   += pnl

        return None, portfolio

    # ── Signal builder ─────────────────────────────────────────────────────────

    def _build_signals(self, df_4h: pd.DataFrame) -> list[dict]:
        """Score every 4h bar and return list of signal dicts keyed by close_time."""
        X_all     = np.vstack([
            extract_features(df_4h.iloc[i], self.feat_cols)
            for i in range(len(df_4h))
        ])
        vol_probs = self.model.predict_proba(X_all)[:, 1]

        signals      = []
        last_sig_idx = -999

        for i in range(self.lookback, len(df_4h)):
            if vol_probs[i] < self.threshold:
                continue
            if (i - last_sig_idx) < self.signal_cooldown:
                continue

            row = df_4h.iloc[i]
            atr = row.get("atr") or 0
            if atr <= 0:
                continue

            # Support / resistance from lookback window BEFORE current bar
            window     = df_4h.iloc[i - self.lookback: i]
            support    = float(window["low"].min())
            resistance = float(window["high"].max())
            range_mid  = (resistance + support) / 2

            limit_buy  = support    - self.sweep_mult * atr
            limit_sell = resistance + self.sweep_mult * atr

            # Initial tight stops — placed immediately on fill
            stop_buy   = limit_buy  - self.tight_stop_mult * atr
            stop_sell  = limit_sell + self.tight_stop_mult * atr

            signals.append({
                "close_time": row["time"] + timedelta(hours=4),
                "vol_prob":   vol_probs[i],
                "regime":     assign_regime(row),
                "support":    support,
                "resistance": resistance,
                "range_mid":  range_mid,
                "atr":        atr,
                "limit_buy":  round(limit_buy,  6),
                "limit_sell": round(limit_sell, 6),
                "stop_buy":   round(stop_buy,   6),
                "stop_sell":  round(stop_sell,  6),
            })
            last_sig_idx = i

        log.info("Signals above threshold %.2f: %d", self.threshold, len(signals))
        return signals

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(
        self,
        df_4h:      pd.DataFrame,
        df_exec:    pd.DataFrame,
        bars_per_4h:int = 1,
    ) -> dict:
        signals   = self._build_signals(df_4h)
        portfolio = self.initial_capital
        equity    = [portfolio]

        completed_trades: list[SweepTrade] = []
        open_trade:  SweepTrade  | None = None
        active_order:SweepOrder  | None = None
        sig_idx      = 0
        last_loss_bar= -999

        cnt_signals  = 0
        cnt_triggered= 0
        cnt_expired  = 0
        cnt_skipped  = 0
        cnt_same_bar_stop = 0   # times stop hit in same bar as fill

        for i_ex, row_ex in df_exec.iterrows():
            bar_time = row_ex["time"]
            high     = row_ex["high"]
            low      = row_ex["low"]
            close    = row_ex["close"]
            atr_bar  = row_ex.get("atr") or 0

            # ── Advance signal queue ───────────────────────────────────────────
            while sig_idx < len(signals):
                sig = signals[sig_idx]
                if sig["close_time"] <= bar_time:
                    if open_trade is None and active_order is None:
                        if (i_ex - last_loss_bar) >= self.loss_cooldown_bars:
                            active_order = SweepOrder(
                                signal_time = sig["close_time"],
                                vol_prob    = sig["vol_prob"],
                                regime      = sig["regime"],
                                support     = sig["support"],
                                resistance  = sig["resistance"],
                                range_mid   = sig["range_mid"],
                                atr         = sig["atr"],
                                limit_buy   = sig["limit_buy"],
                                limit_sell  = sig["limit_sell"],
                                window_end  = (sig["close_time"] +
                                               timedelta(hours=4 * self.window_bars)),
                            )
                            # Store stop levels on the order object
                            active_order.__dict__["stop_buy"]  = sig["stop_buy"]
                            active_order.__dict__["stop_sell"] = sig["stop_sell"]
                            cnt_signals += 1
                        else:
                            cnt_skipped += 1
                    else:
                        cnt_skipped += 1
                    sig_idx += 1
                else:
                    break

            # ── Manage open trade ──────────────────────────────────────────────
            if open_trade is not None:
                result, portfolio = self._manage(
                    open_trade, high, low, close, atr_bar, bar_time, portfolio
                )
                if result is None:
                    if open_trade.pnl < 0:
                        last_loss_bar = i_ex
                    completed_trades.append(open_trade)
                    open_trade = None
                else:
                    open_trade = result

            # ── Check active limit order ───────────────────────────────────────
            if active_order is not None and open_trade is None:

                if active_order.window_end and bar_time > active_order.window_end:
                    cnt_expired += 1
                    active_order = None

                else:
                    lb   = active_order.limit_buy
                    ls   = active_order.limit_sell
                    sl_b = active_order.__dict__.get("stop_buy",  lb - active_order.atr * 0.4)
                    sl_s = active_order.__dict__.get("stop_sell", ls + active_order.atr * 0.4)

                    long_hit  = low  <= lb
                    short_hit = high >= ls

                    # Conservative ambiguity: if both sides hit, take the worse outcome
                    # (assume the side with the tighter stop filled and stopped out)
                    if long_hit and short_hit:
                        long_risk  = abs(lb - sl_b)
                        short_risk = abs(ls - sl_s)
                        # Take the direction whose stop was closer = more likely to hit
                        chosen = "long" if long_risk <= short_risk else "short"
                    elif long_hit:
                        chosen = "long"
                    elif short_hit:
                        chosen = "short"
                    else:
                        chosen = None

                    if chosen is not None:
                        if chosen == "long":
                            entry_price = lb * (1 + COST_PER_SIDE)
                            tight_stop  = sl_b
                        else:
                            entry_price = ls * (1 - COST_PER_SIDE)
                            tight_stop  = sl_s

                        # ── Conservative same-bar stop check ─────────────────
                        # If the same candle that filled us also breaches our
                        # tight stop, we book the loss immediately.
                        # We cannot know intrabar sequence from OHLCV, so we
                        # assume the worst: stop fired right after fill.
                        same_bar_stopped = (
                            (chosen == "long"  and low  <= tight_stop) or
                            (chosen == "short" and high >= tight_stop)
                        )

                        if same_bar_stopped:
                            # Book the immediate stop-loss without opening trade
                            cnt_same_bar_stop += 1
                            if chosen == "long":
                                exit_net = tight_stop * (1 - COST_PER_SIDE)
                                pnl_est  = (exit_net - entry_price)
                            else:
                                exit_net = tight_stop * (1 + COST_PER_SIDE)
                                pnl_est  = (entry_price - exit_net)

                            # Approximate qty based on portfolio for accounting
                            approx_qty = self._size(portfolio, entry_price, tight_stop)
                            pnl        = pnl_est * approx_qty
                            portfolio += pnl
                            last_loss_bar = i_ex

                            # Record as a completed loss trade
                            loss_trade = SweepTrade(
                                symbol       = self.symbol,
                                direction    = chosen,
                                regime       = active_order.regime,
                                vol_prob     = active_order.vol_prob,
                                signal_time  = active_order.signal_time,
                                support      = active_order.support,
                                resistance   = active_order.resistance,
                                range_mid    = active_order.range_mid,
                                range_size   = active_order.resistance - active_order.support,
                                atr_at_signal= active_order.atr,
                                entry_time   = bar_time,
                                entry_price  = round(entry_price, 6),
                                stop_loss    = round(tight_stop, 6),
                                take_profit  = 0.0,
                                qty          = approx_qty,
                                exit_time    = bar_time,
                                exit_price   = round(exit_net, 6),
                                pnl          = round(pnl, 4),
                                pnl_pct      = round((pnl / max(approx_qty * entry_price, 1e-9)) * 100, 3),
                                exit_reason  = "same_bar_stop",
                                commission   = round(approx_qty * entry_price * ROUND_TRIP, 4),
                            )
                            completed_trades.append(loss_trade)
                            active_order = None
                            cnt_triggered += 1

                        else:
                            # Fill survived the candle — open the trade
                            tp = self._compute_target(
                                chosen, entry_price, tight_stop,
                                active_order.range_mid,
                                active_order.resistance - active_order.support,
                                active_order.resistance,
                                active_order.support,
                            )
                            qty = self._size(portfolio, entry_price, tight_stop)

                            if qty > 0:
                                open_trade = SweepTrade(
                                    symbol       = self.symbol,
                                    direction    = chosen,
                                    regime       = active_order.regime,
                                    vol_prob     = active_order.vol_prob,
                                    signal_time  = active_order.signal_time,
                                    support      = active_order.support,
                                    resistance   = active_order.resistance,
                                    range_mid    = active_order.range_mid,
                                    range_size   = active_order.resistance - active_order.support,
                                    atr_at_signal= active_order.atr,
                                    entry_time   = bar_time,
                                    entry_price  = round(entry_price, 6),
                                    stop_loss    = round(tight_stop, 6),
                                    take_profit  = round(tp, 6),
                                    qty          = qty,
                                    commission   = round(qty * entry_price * ROUND_TRIP, 4),
                                )
                                cnt_triggered += 1
                                active_order   = None

            equity.append(portfolio)

        # Force-close open trade at last bar
        if open_trade is not None:
            lc  = df_exec["close"].iloc[-1]
            ep  = (lc * (1 - COST_PER_SIDE) if open_trade.direction == "long"
                   else lc * (1 + COST_PER_SIDE))
            pnl = ((ep - open_trade.entry_price) * open_trade.qty
                   if open_trade.direction == "long"
                   else (open_trade.entry_price - ep) * open_trade.qty)
            cost = open_trade.qty * open_trade.entry_price
            open_trade.exit_time  = df_exec["time"].iloc[-1]
            open_trade.exit_price = round(ep, 6)
            open_trade.pnl        = round(pnl, 4)
            open_trade.pnl_pct    = round((pnl / cost) * 100, 3) if cost > 0 else 0
            open_trade.exit_reason= "end_of_data"
            portfolio            += pnl
            completed_trades.append(open_trade)

        log.info(
            "Signals:%d  Triggered:%d  Expired:%d  Skipped:%d  SameBarStop:%d",
            cnt_signals, cnt_triggered, cnt_expired, cnt_skipped, cnt_same_bar_stop,
        )

        return {
            "trades":            completed_trades,
            "equity_curve":      equity,
            "final_capital":     portfolio,
            "cnt_signals":       cnt_signals,
            "cnt_triggered":     cnt_triggered,
            "cnt_expired":       cnt_expired,
            "cnt_skipped":       cnt_skipped,
            "cnt_same_bar_stop": cnt_same_bar_stop,
        }


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(
    trades: list[SweepTrade], equity: list[float],
    final_cap: float, init_cap: float,
    df_exec: pd.DataFrame, symbol: str, tf_label: str,
) -> dict:
    eq        = np.array(equity, dtype=float)
    total_ret = (final_cap - init_cap) / init_cap * 100
    start     = df_exec["time"].iloc[0]
    end       = df_exec["time"].iloc[-1]
    if hasattr(start, "to_pydatetime"):
        start, end = start.to_pydatetime(), end.to_pydatetime()

    years  = max((end - start).days / 365, 0.001)
    cagr   = ((final_cap / max(init_cap, 1)) ** (1 / years) - 1) * 100
    ppy    = 35_040 if "15m" in tf_label else 2_190
    rets   = np.diff(eq) / np.maximum(eq[:-1], 1e-6)
    rets   = rets[np.isfinite(rets)]
    rf     = RISK_FREE_RATE / ppy
    exc    = rets - rf
    sharpe = float(np.mean(exc) / (np.std(rets) + 1e-10) * math.sqrt(ppy))
    dn     = rets[rets < 0]
    sortino= float(np.mean(exc) / (np.std(dn) + 1e-10) * math.sqrt(ppy))
    peak   = np.maximum.accumulate(eq)
    dds    = (peak - eq) / (peak + 1e-10) * 100
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
    wl_ratio = (abs(float(np.mean([t.pnl_pct for t in won])) /
                    (float(np.mean([t.pnl_pct for t in lost])) + 1e-9))
                if won and lost else 0)

    def sm(lst): return round(float(np.mean(lst)), 3) if lst else 0.0

    return {
        "symbol":           symbol,   "timeframe":        tf_label,
        "start":            start,    "end":              end,
        "initial_capital":  init_cap, "final_capital":    round(final_cap, 2),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct":         round(cagr, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "calmar":           round(calmar, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct":     round(wr, 1),
        "profit_factor":    round(pf, 3),
        "total_trades":     len(trades),
        "avg_trade_pct":    sm(pcts),
        "avg_win_pct":      sm([t.pnl_pct for t in won]),
        "avg_loss_pct":     sm([t.pnl_pct for t in lost]),
        "best_trade_pct":   round(max(pcts, default=0), 3),
        "worst_trade_pct":  round(min(pcts, default=0), 3),
        "avg_hold_hours":   sm(hold_h),
        "win_loss_ratio":   round(wl_ratio, 3),
        "avg_vol_prob":     round(float(np.mean([t.vol_prob for t in trades])), 3) if trades else 0,
        "regime_dist":      dict(Counter(t.regime for t in trades)),
        "exit_dist":        dict(Counter(t.exit_reason.split("_partial")[0] for t in trades)),
        "direction_dist":   dict(Counter(t.direction for t in trades)),
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_stats(s: dict, run_info: dict) -> None:
    sep    = "═" * 68
    passed = (s["total_trades"] >= 20 and s["profit_factor"] >= 1.0
              and s["max_drawdown_pct"] <= 35.0)
    print(f"\n{sep}")
    print(f"  SWEEP v2 (Option B) — {s['symbol']}  [{s['timeframe']}]")
    print(f"  {str(s['start'])[:10]} → {str(s['end'])[:10]}   "
          f"{'✅ PASS' if passed else '❌ FAIL'}")
    print(sep)
    print(f"  {'Capital':34}  ${s['initial_capital']:>10,.2f} → ${s['final_capital']:>10,.2f}")
    print(f"  {'Total return':34}  {s['total_return_pct']:>+10.2f}%")
    print(f"  {'CAGR':34}  {s['cagr_pct']:>+10.2f}%")
    print()

    def row(lbl, val, thr, op=">=", unit=""):
        ok = val >= thr if op == ">=" else val <= thr
        return f"  {'✓' if ok else '✗'}  {lbl:32}  {val:>8.3f}{unit}   ({op}{thr}{unit})"

    print(row("Sharpe ratio",       s["sharpe"],          0.5))
    print(row("Sortino ratio",      s["sortino"],         0.7))
    print(row("Calmar ratio",       s["calmar"],          0.8))
    print(row("Max drawdown",       s["max_drawdown_pct"],35.0, "<=", "%"))
    print(row("Win rate",           s["win_rate_pct"],    40.0, ">=", "%"))
    print(row("Profit factor",      s["profit_factor"],   1.0))
    print(row("Win/Loss ratio",     s["win_loss_ratio"],  1.5))
    print()
    print(f"  {'Total trades':34}  {s['total_trades']:>10}")
    print(f"  {'Avg trade':34}  {s['avg_trade_pct']:>+9.3f}%")
    print(f"  {'Avg win':34}  {s['avg_win_pct']:>+9.3f}%")
    print(f"  {'Avg loss':34}  {s['avg_loss_pct']:>+9.3f}%")
    print(f"  {'Best trade':34}  {s['best_trade_pct']:>+9.3f}%")
    print(f"  {'Worst trade':34}  {s['worst_trade_pct']:>+9.3f}%")
    print(f"  {'Avg hold':34}  {s['avg_hold_hours']:>9.1f} hrs")
    print(f"  {'Avg model prob':34}  {s['avg_vol_prob']:>9.3f}")
    print()
    print(f"  Signal funnel (no look-ahead):")
    print(f"    Signals activated:     {run_info['cnt_signals']:>6}")
    print(f"    Limit orders triggered:{run_info['cnt_triggered']:>6}  "
          f"({run_info['cnt_triggered']/max(run_info['cnt_signals'],1)*100:.0f}%)")
    print(f"    Same-bar stops:        {run_info['cnt_same_bar_stop']:>6}  "
          f"(fill + immediate stop in same candle)")
    print(f"    Orders expired:        {run_info['cnt_expired']:>6}")
    print(f"    Skipped (occupied):    {run_info['cnt_skipped']:>6}")

    if s["regime_dist"]:
        print(f"\n  Regime:  " + "  ".join(f"{k}={v}" for k,v in sorted(s["regime_dist"].items())))
    if s["exit_dist"]:
        print(f"  Exits:   " + "  ".join(f"{k}={v}" for k,v in sorted(s["exit_dist"].items())))
    if s["direction_dist"]:
        print(f"  Dir:     " + "  ".join(f"{k}={v}" for k,v in sorted(s["direction_dist"].items())))
    print(sep)


def save_trades(trades: list[SweepTrade], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "signal_time","entry_time","exit_time",
            "symbol","direction","regime","vol_prob",
            "support","resistance","range_mid","range_size","atr_at_signal",
            "entry_price","exit_price","stop_loss","take_profit",
            "qty","pnl","pnl_pct","exit_reason","commission",
        ])
        for t in trades:
            w.writerow([
                t.signal_time, t.entry_time, t.exit_time,
                t.symbol, t.direction, t.regime, t.vol_prob,
                t.support, t.resistance, t.range_mid, t.range_size, t.atr_at_signal,
                t.entry_price, t.exit_price, t.stop_loss, t.take_profit,
                t.qty, t.pnl, t.pnl_pct, t.exit_reason, t.commission,
            ])
    log.info("Trades → %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Sweep backtester v2 — intrabar fill, tight stop, no look-ahead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv-4h",           required=True)
    p.add_argument("--csv-15m",          default=None)
    p.add_argument("--model",            required=True)
    p.add_argument("--meta",             required=True)
    p.add_argument("--symbol",           default="SOL/USDT")
    p.add_argument("--capital",          type=float, default=10_000.0)
    p.add_argument("--risk",             type=float, default=1.0)
    p.add_argument("--threshold",        type=float, default=None)
    p.add_argument("--lookback",         type=int,   default=10)
    p.add_argument("--sweep-mult",       type=float, default=0.4,
                   help="ATR units below support for limit buy (default 0.4)")
    p.add_argument("--tight-stop-mult",  type=float, default=0.4,
                   help="ATR units below limit buy for initial stop (default 0.4)")
    p.add_argument("--target-mode",      default="opposite",
                   choices=["midpoint","opposite","rr"])
    p.add_argument("--extension-mult",   type=float, default=0.3)
    p.add_argument("--rr",               type=float, default=2.0)
    p.add_argument("--window-bars",      type=int,   default=4,
                   help="4h bars the limit order stays open (default 4 = 16h)")
    p.add_argument("--trail-mult",       type=float, default=1.0,
                   help="ATR multiplier for trailing stop distance (default 1.0)")
    p.add_argument("--trail-trigger-r",  type=float, default=1.0,
                   help="R profit before trail activates (default 1.0)")
    p.add_argument("--use-15m",          action="store_true")
    p.add_argument("--no-trailing",      action="store_true")
    p.add_argument("--no-partial",       action="store_true")
    p.add_argument("--threshold-sweep",  action="store_true")
    p.add_argument("--param-sweep",      action="store_true",
                   help="Sweep sweep-mult and tight-stop-mult combinations")
    p.add_argument("--save-trades",      action="store_true")
    args = p.parse_args()

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    with open(args.meta) as f:
        meta = json.load(f)
    feat_cols = meta.get("feature_cols", [])
    threshold = args.threshold or meta.get("recommended_threshold", 0.60)
    log.info("Model AUC: %.4f  |  Threshold: %.2f", meta.get("auc_roc", 0), threshold)

    df_4h = load_4h(args.csv_4h)
    if args.use_15m:
        if not args.csv_15m:
            log.error("--use-15m requires --csv-15m"); sys.exit(1)
        df_exec    = load_15m(args.csv_15m)
        df_4h, df_exec = align(df_4h, df_exec)
        bars_per_4h= 16
        tf_label   = "4h→15m sweep"
    else:
        df_exec    = df_4h.copy()
        bars_per_4h= 1
        tf_label   = "4h sweep"

    def make_engine(thresh, sweep_mult=None, stop_mult=None):
        return SweepEngine(
            model=model, feat_cols=feat_cols, threshold=thresh,
            symbol=args.symbol, initial_capital=args.capital,
            risk_per_trade=args.risk / 100,
            lookback=args.lookback,
            sweep_mult=sweep_mult or args.sweep_mult,
            tight_stop_mult=stop_mult or args.tight_stop_mult,
            target_mode=args.target_mode,
            extension_mult=args.extension_mult,
            rr_fallback=args.rr,
            window_bars=args.window_bars,
            use_partial=not args.no_partial,
            use_trailing=not args.no_trailing,
            trail_mult=args.trail_mult,
            trail_trigger_r=args.trail_trigger_r,
        )

    def single_run(thresh, sweep_mult=None, stop_mult=None):
        e = make_engine(thresh, sweep_mult, stop_mult)
        return e.run(df_4h.copy(), df_exec.copy(), bars_per_4h=bars_per_4h)

    # ── Threshold sweep ────────────────────────────────────────────────────────
    if args.threshold_sweep:
        print(f"\n  Sweep v2 threshold sweep — {args.symbol}  [{tf_label}]")
        print(f"  sweep_mult={args.sweep_mult}  tight_stop_mult={args.tight_stop_mult}")
        print(f"  {'Thresh':>7}  {'Trades':>7}  {'SameBar':>8}  {'WR':>7}  "
              f"{'PF':>7}  {'Ret':>9}  {'Sharpe':>8}  {'MaxDD':>8}")
        print("  " + "─" * 74)
        for t in np.arange(0.50, 0.81, 0.05):
            r = single_run(round(t, 2))
            trades = r["trades"]
            if not trades:
                print(f"  {t:>7.2f}  {'(no fills)':>7}")
                continue
            s    = compute_stats(trades, r["equity_curve"], r["final_capital"],
                                 args.capital, df_exec, args.symbol, tf_label)
            mark = " ←" if s["profit_factor"] >= 1.0 and s["total_trades"] >= 15 else ""
            print(f"  {t:>7.2f}  {s['total_trades']:>7}  "
                  f"{r['cnt_same_bar_stop']:>8}  "
                  f"{s['win_rate_pct']:>6.1f}%  "
                  f"{s['profit_factor']:>7.3f}  "
                  f"{s['total_return_pct']:>+8.2f}%  "
                  f"{s['sharpe']:>8.3f}  "
                  f"{s['max_drawdown_pct']:>7.1f}%{mark}")
        return

    # ── Parameter sweep ────────────────────────────────────────────────────────
    if args.param_sweep:
        print(f"\n  Parameter sweep: sweep_mult × tight_stop_mult  [{tf_label}]")
        print(f"  threshold={threshold}")
        print(f"  {'sweep_m':>8}  {'stop_m':>7}  {'Trades':>7}  {'WR':>7}  "
              f"{'PF':>7}  {'Ret':>9}  {'WLR':>7}")
        print("  " + "─" * 68)
        for sm in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for stm in [0.2, 0.3, 0.4, 0.5]:
                r = single_run(threshold, sweep_mult=sm, stop_mult=stm)
                trades = r["trades"]
                if not trades:
                    print(f"  {sm:>8.1f}  {stm:>7.1f}  {'(no fills)':>7}")
                    continue
                s = compute_stats(trades, r["equity_curve"], r["final_capital"],
                                  args.capital, df_exec, args.symbol, tf_label)
                mark = " ←" if s["profit_factor"] >= 1.0 and s["total_trades"] >= 15 else ""
                print(f"  {sm:>8.1f}  {stm:>7.1f}  {s['total_trades']:>7}  "
                      f"{s['win_rate_pct']:>6.1f}%  {s['profit_factor']:>7.3f}  "
                      f"{s['total_return_pct']:>+8.2f}%  {s['win_loss_ratio']:>7.3f}{mark}")
        return

    # ── Single run ─────────────────────────────────────────────────────────────
    r = single_run(threshold)
    s = compute_stats(r["trades"], r["equity_curve"], r["final_capital"],
                      args.capital, df_exec, args.symbol, tf_label)
    print_stats(s, r)

    if args.save_trades and r["trades"]:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"backtest/output/SOL_sweep_v2_{ts}_trades.csv"
        save_trades(r["trades"], path)
        print(f"\n  Trades saved → {path}")
        print(f"  Open in Excel and review exit_reason column.")
        print(f"  'same_bar_stop' = fill + stop in same candle (genuine losses)")
        print(f"  'take_profit'   = target hit (confirming sweep worked)")
        print(f"  'trail_stop'    = trailing stop (means trade moved in our favour)")


if __name__ == "__main__":
    main()