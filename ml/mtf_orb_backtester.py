"""
ml/mtf_orb_backtester.py
─────────────────────────
Multi-Timeframe Opening Range Breakout backtester.

Signal flow:
  4h candle closes
      → Volatility model scores it
      → If vol_prob >= threshold: open a 2-hour execution window
      → First 15m candle in that window = setup candle (defines ORB range)
      → Place buy-stop above setup high, sell-stop below setup low
      → Remaining 15m candles in window = execution candles
      → First stop triggered = entry. Other side cancelled.
      → Trade managed bar-by-bar on 15m candles until SL/TP/trail exit

Why this works where 4h straddle failed:
  - 15m SOL candle range ≈ 0.3-0.8% vs 4h range ≈ 3-5%
  - Stop loss is 5-8x tighter → ambiguous candle problem disappears
  - Entry is still on a confirmed breakout (not market order)
  - 4h model provides the macro timing filter so we only trade
    during windows the model identified as high-volatility

Key parameters:
  --window-bars    How many 15m candles the execution window stays open (default 8 = 2h)
  --threshold      4h model probability threshold (default: from meta)
  --rr             Take profit R:R ratio (default 2.5)
  --max-range-pct  Skip setup candles where 15m range > X% (avoids post-explosion entries)

Usage:
    # Basic run
    python ml/mtf_orb_backtester.py \\
        --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
        --csv-15m backtest/data/SOL_USDT_15m.csv \\
        --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol  "SOL/USDT"

    # Threshold sweep
    python ml/mtf_orb_backtester.py \\
        --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
        --csv-15m backtest/data/SOL_USDT_15m.csv \\
        --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol  "SOL/USDT" --threshold-sweep

    # Save trade log
    python ml/mtf_orb_backtester.py \\
        --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
        --csv-15m backtest/data/SOL_USDT_15m.csv \\
        --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol  "SOL/USDT" --save-trades
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
except ImportError:
    print("Run: pip install xgboost")
    sys.exit(1)

from backtest.engine import build_features, assign_regime, COST_PER_SIDE, ROUND_TRIP, RISK_FREE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mtf_orb")

REGIMES        = ["ranging", "trending_bull", "trending_bear", "high_volatility"]
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


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class OrbSetup:
    """Represents one active ORB setup triggered by the 4h model."""
    signal_4h_time:  datetime    # 4h candle close time that triggered model
    vol_prob:        float       # model probability
    regime:          str
    window_end:      datetime    # latest 15m bar that can still trigger entry
    # Filled when setup candle is identified (first 15m bar in window)
    setup_time:      datetime | None = None
    setup_high:      float = 0.0
    setup_low:       float = 0.0
    buy_stop:        float = 0.0
    sell_stop:       float = 0.0
    range_pct:       float = 0.0
    setup_done:      bool  = False   # True once setup candle has been identified


@dataclass
class OrbTrade:
    """A completed or open ORB trade."""
    symbol:          str
    direction:       Literal["long", "short"]
    regime:          str
    vol_prob:        float
    signal_4h_time:  datetime
    setup_time:      datetime
    setup_high:      float
    setup_low:       float
    range_pct:       float
    entry_time:      datetime | None = None
    entry_price:     float = 0.0
    stop_loss:       float = 0.0
    take_profit:     float = 0.0
    trail_stop:      float = 0.0
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


# ── Data loading and preparation ───────────────────────────────────────────────

def load_4h(csv_path: str) -> pd.DataFrame:
    """Load the alpha-enriched 4h CSV and compute technical features."""
    log.info("Loading 4h data: %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.sort_values("time").reset_index(drop=True)

    # Recompute technical indicators (ensures macd_hist_prev etc. are present)
    ohlcv = df[["time","open","high","low","close","volume"]].copy()
    tech  = build_features(ohlcv)
    for col in [c for c in tech.columns if c not in ["time","open","high","low","close","volume"]]:
        df[col] = tech[col].values

    # Regime one-hot columns for model features
    for r in REGIMES:
        df[f"regime_{r}"] = df.apply(
            lambda row, r=r: 1 if assign_regime(row) == r else 0, axis=1
        )

    df = df.dropna(subset=["atr"]).reset_index(drop=True)
    log.info("4h ready: %d bars  (%s → %s)",
             len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))
    return df


def load_15m(csv_path: str) -> pd.DataFrame:
    """Load 15m OHLCV CSV. Compute ATR for stop sizing on 15m bars."""
    log.info("Loading 15m data: %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.sort_values("time").reset_index(drop=True)

    # Compute ATR on 15m for position sizing
    import pandas_ta as ta
    df["atr_15m"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df = df.dropna(subset=["atr_15m"]).reset_index(drop=True)

    log.info("15m ready: %d bars  (%s → %s)",
             len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))
    return df


def align_timeframes(df_4h: pd.DataFrame, df_15m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trim both dataframes to their overlapping date range.
    This is essential — the backtester needs both timeframes to cover the same period.
    """
    start = max(df_4h["time"].iloc[0],  df_15m["time"].iloc[0])
    end   = min(df_4h["time"].iloc[-1], df_15m["time"].iloc[-1])

    df_4h  = df_4h[(df_4h["time"]  >= start) & (df_4h["time"]  <= end)].reset_index(drop=True)
    df_15m = df_15m[(df_15m["time"] >= start) & (df_15m["time"] <= end)].reset_index(drop=True)

    log.info("Aligned range: %s → %s", str(start.date()), str(end.date()))
    log.info("4h bars: %d  |  15m bars: %d", len(df_4h), len(df_15m))
    return df_4h, df_15m


# ── Core engine ────────────────────────────────────────────────────────────────

class MTFOrbEngine:
    """
    Multi-timeframe ORB engine.

    The engine maintains two parallel loops:
      - 4h loop: scores each 4h candle with the volatility model,
                 creates OrbSetup objects when threshold is crossed
      - 15m loop: for each 15m bar, checks if it falls inside an active
                  setup window, executes the ORB entry, and manages open trades

    The two loops are synchronized by timestamp: for each 15m bar, we
    first apply all 4h signals whose close time is <= the 15m bar time.
    This preserves the no-look-ahead guarantee.
    """

    def __init__(
        self,
        model:             xgb.XGBClassifier,
        feat_cols:         list[str],
        threshold:         float = 0.60,
        symbol:            str   = "SOL/USDT",
        initial_capital:   float = 10_000.0,
        risk_per_trade:    float = 0.01,
        window_bars_15m:   int   = 8,      # 8 × 15m = 2h execution window
        rr:                float = 2.5,
        max_range_pct:     float = 2.0,    # skip 15m setup candles > 2% range
        min_range_pct:     float = 0.15,   # skip 15m setup candles < 0.15% range
        trigger_buffer_pct:float = 0.05,   # buy_stop = high × (1 + 0.0005)
        stop_buffer_pct:   float = 0.10,   # stop sits 0.1% beyond opposite side
        # Three-tier exit
        be_trigger_r:      float = 0.75,
        be_buffer_r:       float = 0.05,
        use_partial:       bool  = True,
        partial_r:         float = 1.0,
        partial_pct:       float = 0.40,
        use_trailing:      bool  = True,
        trail_trigger_r:   float = 1.5,
        trail_distance_r:  float = 1.0,
        # Guards
        max_open:          int   = 1,
        signal_cooldown_4h:int   = 2,      # bars between 4h signals
        loss_cooldown_4h:  int   = 3,      # extra wait after losing trade
    ):
        self.model              = model
        self.feat_cols          = feat_cols
        self.threshold          = threshold
        self.symbol             = symbol
        self.initial_capital    = initial_capital
        self.risk_per_trade     = risk_per_trade
        self.window_bars_15m    = window_bars_15m
        self.rr                 = rr
        self.max_range_pct      = max_range_pct
        self.min_range_pct      = min_range_pct
        self.trigger_buffer_pct = trigger_buffer_pct / 100
        self.stop_buffer_pct    = stop_buffer_pct / 100
        self.be_trigger_r       = be_trigger_r
        self.be_buffer_r        = be_buffer_r
        self.use_partial        = use_partial
        self.partial_r          = partial_r
        self.partial_pct        = partial_pct
        self.use_trailing       = use_trailing
        self.trail_trigger_r    = trail_trigger_r
        self.trail_distance_r   = trail_distance_r
        self.max_open           = max_open
        self.signal_cooldown_4h = signal_cooldown_4h
        self.loss_cooldown_4h   = loss_cooldown_4h

    def _size_position(self, portfolio: float, entry: float, sl: float) -> float:
        stop_dist   = abs(entry - sl)
        dollar_risk = portfolio * self.risk_per_trade
        if stop_dist <= 0 or entry <= 0:
            return 0.0
        return round(dollar_risk / stop_dist, 6)

    def _manage_trade(
        self,
        trade:    OrbTrade,
        high:     float,
        low:      float,
        close:    float,
        bar_time: datetime,
        portfolio:float,
    ) -> tuple[OrbTrade | None, float]:
        """Apply three-tier exit to open trade. Returns (trade|None, portfolio)."""
        risk_dist = abs(trade.entry_price - trade.stop_loss)
        if risk_dist <= 0:
            return trade, portfolio

        profit_r_high = ((high - trade.entry_price) / risk_dist if trade.direction == "long"
                         else (trade.entry_price - low) / risk_dist)

        # Tier 1: Breakeven
        if not trade.breakeven_set and profit_r_high >= self.be_trigger_r:
            buf = risk_dist * self.be_buffer_r
            if trade.direction == "long":
                new_be = trade.entry_price + buf
                if new_be > trade.stop_loss:
                    trade.stop_loss    = round(new_be, 6)
                    trade.breakeven_set = True
            else:
                new_be = trade.entry_price - buf
                if new_be < trade.stop_loss:
                    trade.stop_loss    = round(new_be, 6)
                    trade.breakeven_set = True

        # Tier 2: Partial profit
        if self.use_partial and not trade.partial_done and profit_r_high >= self.partial_r:
            pqty = round(trade.qty * self.partial_pct, 6)
            if pqty > 0:
                ep = (close * (1 - COST_PER_SIDE) if trade.direction == "long"
                      else close * (1 + COST_PER_SIDE))
                pp = ((ep - trade.entry_price) * pqty if trade.direction == "long"
                      else (trade.entry_price - ep) * pqty)
                portfolio       += pp
                trade.qty       -= pqty
                trade.partial_done = True

        # Tier 3: Trailing stop
        if self.use_trailing and profit_r_high >= self.trail_trigger_r:
            td = risk_dist * self.trail_distance_r
            if trade.direction == "long":
                new_ts = high - td
                if new_ts > trade.trail_stop:
                    trade.trail_stop = round(new_ts, 6)
            else:
                new_ts = low + td
                if trade.trail_stop == 0 or new_ts < trade.trail_stop:
                    trade.trail_stop = round(new_ts, 6)

        # Determine active stop and check for exit
        active_sl = trade.trail_stop if trade.trail_stop > 0 else trade.stop_loss
        reason    = None
        exit_ref  = 0.0

        if trade.direction == "long":
            if low <= active_sl:
                exit_ref = active_sl
                reason   = ("trail_stop" if trade.trail_stop > 0
                            else "breakeven" if trade.breakeven_set
                            else "stop_loss")
            elif high >= trade.take_profit:
                exit_ref = trade.take_profit
                reason   = "take_profit"
        else:
            if high >= active_sl:
                exit_ref = active_sl
                reason   = ("trail_stop" if trade.trail_stop > 0
                            else "breakeven" if trade.breakeven_set
                            else "stop_loss")
            elif low <= trade.take_profit:
                exit_ref = trade.take_profit
                reason   = "take_profit"

        if reason is None:
            return trade, portfolio

        # Close trade
        exit_net = (exit_ref * (1 - COST_PER_SIDE) if trade.direction == "long"
                    else exit_ref * (1 + COST_PER_SIDE))
        pnl      = ((exit_net - trade.entry_price) * trade.qty if trade.direction == "long"
                    else (trade.entry_price - exit_net) * trade.qty)
        cost     = trade.qty * trade.entry_price

        trade.exit_time  = bar_time
        trade.exit_price = round(exit_net, 6)
        trade.pnl        = round(pnl, 4)
        trade.pnl_pct    = round((pnl / cost) * 100, 3) if cost > 0 else 0
        trade.exit_reason= reason + ("_partial" if trade.partial_done else "")
        portfolio       += pnl

        return None, portfolio

    def run(
        self,
        df_4h:  pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> dict:
        # ── Pre-compute 4h model probabilities ────────────────────────────────
        log.info("Computing 4h volatility probabilities…")
        X_4h      = np.vstack([
            extract_features(df_4h.iloc[i], self.feat_cols)
            for i in range(len(df_4h))
        ])
        vol_probs = self.model.predict_proba(X_4h)[:, 1]
        n_above   = (vol_probs >= self.threshold).sum()
        log.info("4h bars above threshold %.2f: %d / %d (%.1f%%)",
                 self.threshold, n_above, len(df_4h),
                 n_above / max(len(df_4h), 1) * 100)

        # ── Build 4h signal list ───────────────────────────────────────────────
        # Each 4h candle's close time marks when the signal becomes available.
        # We convert the 4h bar open time to a close time by adding 4 hours.
        # This is critical for no-look-ahead: the signal is only usable AFTER
        # the 4h candle closes.
        tf_delta = timedelta(hours=4)
        signals_4h = []
        last_signal_idx = -999

        for i in range(len(df_4h)):
            if vol_probs[i] < self.threshold:
                continue
            if (i - last_signal_idx) < self.signal_cooldown_4h:
                continue
            bar_close_time = df_4h["time"].iloc[i] + tf_delta
            regime         = assign_regime(df_4h.iloc[i])
            window_end     = bar_close_time + timedelta(minutes=15 * self.window_bars_15m)
            signals_4h.append({
                "close_time": bar_close_time,
                "window_end": window_end,
                "vol_prob":   vol_probs[i],
                "regime":     regime,
            })
            last_signal_idx = i

        log.info("4h signals generated: %d", len(signals_4h))

        # ── Main 15m loop ──────────────────────────────────────────────────────
        portfolio         = self.initial_capital
        equity_curve      = [portfolio]
        completed_trades: list[OrbTrade] = []
        open_trade:  OrbTrade | None  = None
        active_setup: OrbSetup | None = None
        sig_idx       = 0          # pointer into signals_4h list
        last_loss_bar = -999       # 15m bar index of last losing trade
        loss_cooldown_15m = self.loss_cooldown_4h * 16   # convert 4h bars to 15m bars

        # Diagnostic counters
        signals_activated = 0
        setups_formed     = 0
        entries_triggered = 0
        setups_expired    = 0
        skipped_range     = 0

        for i15, row15 in df_15m.iterrows():
            bar_time = row15["time"]
            high     = row15["high"]
            low      = row15["low"]
            close    = row15["close"]
            atr_15m  = row15.get("atr_15m") or 0

            # ── Advance 4h signals: activate any whose close_time <= bar_time ─
            while sig_idx < len(signals_4h):
                sig = signals_4h[sig_idx]
                if sig["close_time"] <= bar_time:
                    # Only activate if no open trade and no active setup
                    if open_trade is None and active_setup is None:
                        if (i15 - last_loss_bar) >= loss_cooldown_15m:
                            active_setup = OrbSetup(
                                signal_4h_time = sig["close_time"],
                                vol_prob       = sig["vol_prob"],
                                regime         = sig["regime"],
                                window_end     = sig["window_end"],
                            )
                            signals_activated += 1
                    sig_idx += 1
                else:
                    break

            # ── Manage open trade ──────────────────────────────────────────────
            if open_trade is not None:
                result, portfolio = self._manage_trade(
                    open_trade, high, low, close, bar_time, portfolio
                )
                if result is None:
                    # Trade closed
                    if open_trade.pnl < 0:
                        last_loss_bar = i15
                    completed_trades.append(open_trade)
                    open_trade = None
                else:
                    open_trade = result

            # ── Process active setup ───────────────────────────────────────────
            if active_setup is not None and open_trade is None:

                # Check if setup window has expired
                if bar_time > active_setup.window_end:
                    setups_expired += 1
                    log.debug("Setup expired at %s (window ended %s)",
                              bar_time, active_setup.window_end)
                    active_setup = None

                elif not active_setup.setup_done:
                    # This is the first 15m bar inside the window — it IS the setup candle
                    range_size = high - low
                    range_pct  = (range_size / close * 100) if close > 0 else 0

                    if range_pct > self.max_range_pct:
                        # Candle already moved too much — entered window mid-explosion
                        skipped_range += 1
                        active_setup = None
                    elif range_pct < self.min_range_pct:
                        # Range too tight — spread eats the trade
                        skipped_range += 1
                        active_setup = None
                    else:
                        # Valid setup candle — define the ORB levels
                        buf = close * self.trigger_buffer_pct
                        active_setup.setup_time  = bar_time
                        active_setup.setup_high  = high
                        active_setup.setup_low   = low
                        active_setup.buy_stop    = round(high + buf, 6)
                        active_setup.sell_stop   = round(low  - buf, 6)
                        active_setup.range_pct   = round(range_pct, 3)
                        active_setup.setup_done  = True
                        setups_formed += 1
                        log.debug("Setup at %s  buy=%.4f  sell=%.4f  range=%.2f%%",
                                  bar_time, active_setup.buy_stop,
                                  active_setup.sell_stop, range_pct)

                else:
                    # Execution candle — check if either stop triggered
                    bs = active_setup.buy_stop
                    ss = active_setup.sell_stop

                    long_hit  = high >= bs
                    short_hit = low  <= ss

                    if long_hit and short_hit:
                        # Ambiguous: conservative assumption — losing side first
                        # Determine which entry has the smaller loss and take it
                        long_entry  = bs * (1 + COST_PER_SIDE)
                        long_sl     = ss * (1 - self.stop_buffer_pct)
                        short_entry = ss * (1 - COST_PER_SIDE)
                        short_sl    = bs * (1 + self.stop_buffer_pct)

                        long_loss_est  = abs(long_entry  - long_sl)
                        short_loss_est = abs(short_entry - short_sl)

                        # Take the direction with the smaller maximum loss
                        chosen = "long" if long_loss_est <= short_loss_est else "short"
                        entry_price = long_entry if chosen == "long" else short_entry
                        sl          = long_sl    if chosen == "long" else short_sl

                    elif long_hit:
                        chosen      = "long"
                        entry_price = bs * (1 + COST_PER_SIDE)
                        sl          = ss * (1 - self.stop_buffer_pct)
                    elif short_hit:
                        chosen      = "short"
                        entry_price = ss * (1 - COST_PER_SIDE)
                        sl          = bs * (1 + self.stop_buffer_pct)
                    else:
                        chosen = None

                    if chosen is not None:
                        range_width = active_setup.setup_high - active_setup.setup_low
                        if chosen == "long":
                            tp = entry_price + range_width * self.rr
                        else:
                            tp = entry_price - range_width * self.rr

                        qty = self._size_position(portfolio, entry_price, sl)

                        if qty > 0:
                            open_trade = OrbTrade(
                                symbol         = self.symbol,
                                direction      = chosen,
                                regime         = active_setup.regime,
                                vol_prob       = active_setup.vol_prob,
                                signal_4h_time = active_setup.signal_4h_time,
                                setup_time     = active_setup.setup_time,
                                setup_high     = active_setup.setup_high,
                                setup_low      = active_setup.setup_low,
                                range_pct      = active_setup.range_pct,
                                entry_time     = bar_time,
                                entry_price    = round(entry_price, 6),
                                stop_loss      = round(sl, 6),
                                take_profit    = round(tp, 6),
                                trail_stop     = 0.0,
                                qty            = qty,
                                commission     = round(qty * entry_price * ROUND_TRIP, 4),
                            )
                            entries_triggered += 1
                            active_setup = None

                            log.debug("ENTRY %s at %.4f  sl=%.4f  tp=%.4f  qty=%.4f",
                                      chosen.upper(), entry_price, sl, tp, qty)

            equity_curve.append(portfolio)

        # Force-close any open trade at last bar
        if open_trade is not None:
            lc = df_15m["close"].iloc[-1]
            ep = (lc * (1 - COST_PER_SIDE) if open_trade.direction == "long"
                  else lc * (1 + COST_PER_SIDE))
            pnl = ((ep - open_trade.entry_price) * open_trade.qty
                   if open_trade.direction == "long"
                   else (open_trade.entry_price - ep) * open_trade.qty)
            cost = open_trade.qty * open_trade.entry_price
            open_trade.exit_time  = df_15m["time"].iloc[-1]
            open_trade.exit_price = round(ep, 6)
            open_trade.pnl        = round(pnl, 4)
            open_trade.pnl_pct    = round((pnl / cost) * 100, 3) if cost > 0 else 0
            open_trade.exit_reason= "end_of_data"
            portfolio            += pnl
            completed_trades.append(open_trade)

        log.info(
            "Run complete: %d signals → %d setups formed → %d entries → "
            "%d expired → %d range-skipped",
            signals_activated, setups_formed, entries_triggered,
            setups_expired, skipped_range,
        )

        return {
            "trades":            completed_trades,
            "equity_curve":      equity_curve,
            "final_capital":     portfolio,
            "signals_activated": signals_activated,
            "setups_formed":     setups_formed,
            "entries_triggered": entries_triggered,
            "setups_expired":    setups_expired,
            "skipped_range":     skipped_range,
        }


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(
    trades:          list[OrbTrade],
    equity_curve:    list[float],
    final_capital:   float,
    initial_capital: float,
    df_15m:          pd.DataFrame,
    symbol:          str,
) -> dict:
    equity    = np.array(equity_curve, dtype=float)
    total_ret = (final_capital - initial_capital) / initial_capital * 100
    start     = df_15m["time"].iloc[0]
    end       = df_15m["time"].iloc[-1]
    if hasattr(start, "to_pydatetime"):
        start, end = start.to_pydatetime(), end.to_pydatetime()

    years  = max((end - start).days / 365, 0.001)
    cagr   = ((final_capital / max(initial_capital, 1)) ** (1 / years) - 1) * 100

    ppy    = 35_040   # 15m bars per year
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

    from collections import Counter
    regime_dist = Counter(t.regime     for t in trades)
    exit_dist   = Counter(t.exit_reason.split("_partial")[0] for t in trades)
    dir_dist    = Counter(t.direction  for t in trades)
    avg_prob    = float(np.mean([t.vol_prob  for t in trades])) if trades else 0
    avg_range   = float(np.mean([t.range_pct for t in trades])) if trades else 0

    def safe_mean(lst): return round(float(np.mean(lst)), 3) if lst else 0.0

    return {
        "symbol":           symbol,
        "timeframe":        "4h→15m",
        "start":            start,
        "end":              end,
        "initial_capital":  initial_capital,
        "final_capital":    round(final_capital, 2),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct":         round(cagr, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "calmar":           round(calmar, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct":     round(wr, 1),
        "profit_factor":    round(pf, 3),
        "total_trades":     len(trades),
        "avg_trade_pct":    safe_mean(pcts),
        "avg_win_pct":      safe_mean([t.pnl_pct for t in won]),
        "avg_loss_pct":     safe_mean([t.pnl_pct for t in lost]),
        "best_trade_pct":   round(max(pcts, default=0), 3),
        "worst_trade_pct":  round(min(pcts, default=0), 3),
        "avg_hold_hours":   safe_mean(hold_h),
        "avg_vol_prob":     round(avg_prob, 3),
        "avg_setup_range":  round(avg_range, 3),
        "regime_dist":      dict(regime_dist),
        "exit_dist":        dict(exit_dist),
        "direction_dist":   dict(dir_dist),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_stats(s: dict, run_info: dict) -> None:
    sep     = "═" * 66
    passed  = (
        s["total_trades"]    >= 20 and
        s["sharpe"]          >= 0.5 and
        s["max_drawdown_pct"]<= 30.0 and
        s["profit_factor"]   >= 1.0 and
        s["win_rate_pct"]    >= 45.0
    )
    verdict = "✅ PASS" if passed else "❌ FAIL"

    print(f"\n{sep}")
    print(f"  MTF ORB — {s['symbol']}  (4h signal → 15m execution)")
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
    print(f"  {'Avg model prob':34}  {s['avg_vol_prob']:>9.3f}")
    print(f"  {'Avg 15m setup range':34}  {s['avg_setup_range']:>8.3f}%")
    print()
    print(f"  Signal funnel (4h→15m):")
    print(f"    4h signals activated:  {run_info['signals_activated']:>6}")
    print(f"    15m setups formed:     {run_info['setups_formed']:>6}  "
          f"({run_info['setups_formed']/max(run_info['signals_activated'],1)*100:.0f}%)")
    print(f"    Entries triggered:     {run_info['entries_triggered']:>6}  "
          f"({run_info['entries_triggered']/max(run_info['setups_formed'],1)*100:.0f}%)")
    print(f"    Setups expired:        {run_info['setups_expired']:>6}")
    print(f"    Range-skipped:         {run_info['skipped_range']:>6}")
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


def save_trades(trades: list[OrbTrade], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "signal_4h_time", "setup_time", "entry_time", "exit_time",
            "symbol", "direction", "regime", "vol_prob",
            "setup_high", "setup_low", "range_pct",
            "entry_price", "exit_price", "stop_loss", "take_profit",
            "qty", "pnl", "pnl_pct", "exit_reason", "commission",
        ])
        for t in trades:
            w.writerow([
                t.signal_4h_time, t.setup_time, t.entry_time, t.exit_time,
                t.symbol, t.direction, t.regime, t.vol_prob,
                t.setup_high, t.setup_low, t.range_pct,
                t.entry_price, t.exit_price, t.stop_loss, t.take_profit,
                t.qty, t.pnl, t.pnl_pct, t.exit_reason, t.commission,
            ])
    log.info("Trades saved → %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Multi-timeframe ORB backtester (4h signal → 15m execution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ml/mtf_orb_backtester.py \\
        --csv-4h  ml/data/SOL_USDT_4h_alpha.csv \\
        --csv-15m backtest/data/SOL_USDT_15m.csv \\
        --model   ml/models/SOL_USDT_4h_alpha_volatility.json \\
        --meta    ml/models/SOL_USDT_4h_alpha_volatility_meta.json \\
        --symbol  "SOL/USDT" --threshold-sweep
        """
    )
    p.add_argument("--csv-4h",   required=True, help="Alpha-enriched 4h CSV")
    p.add_argument("--csv-15m",  required=True, help="15m OHLCV CSV")
    p.add_argument("--model",    required=True, help="_volatility.json model")
    p.add_argument("--meta",     required=True, help="_volatility_meta.json")
    p.add_argument("--symbol",   default="SOL/USDT")
    p.add_argument("--capital",  type=float, default=10_000.0)
    p.add_argument("--risk",     type=float, default=1.0)
    p.add_argument("--threshold",type=float, default=None)
    p.add_argument("--window-bars",    type=int,   default=8,
                   help="15m bars the execution window stays open (default 8 = 2h)")
    p.add_argument("--rr",             type=float, default=2.5)
    p.add_argument("--max-range-pct",  type=float, default=2.0,
                   help="Max 15m setup candle range %% (default 2.0)")
    p.add_argument("--min-range-pct",  type=float, default=0.15)
    p.add_argument("--trigger-buffer", type=float, default=0.05,
                   help="Buy/sell stop buffer above/below as %% of price (default 0.05)")
    p.add_argument("--stop-buffer",    type=float, default=0.10)
    p.add_argument("--no-trailing",    action="store_true")
    p.add_argument("--no-partial",     action="store_true")
    p.add_argument("--threshold-sweep",action="store_true")
    p.add_argument("--save-trades",    action="store_true")
    args = p.parse_args()

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    with open(args.meta) as f:
        meta = json.load(f)
    feat_cols = meta.get("feature_cols", [])
    threshold = args.threshold or meta.get("recommended_threshold", 0.60)
    log.info("Model AUC: %.4f  |  Threshold: %.2f", meta.get("auc_roc", 0), threshold)

    # Load data
    df_4h  = load_4h(args.csv_4h)
    df_15m = load_15m(args.csv_15m)
    df_4h, df_15m = align_timeframes(df_4h, df_15m)

    def single_run(thresh):
        engine = MTFOrbEngine(
            model=model, feat_cols=feat_cols, threshold=thresh,
            symbol=args.symbol, initial_capital=args.capital,
            risk_per_trade=args.risk / 100,
            window_bars_15m=args.window_bars,
            rr=args.rr,
            max_range_pct=args.max_range_pct,
            min_range_pct=args.min_range_pct,
            trigger_buffer_pct=args.trigger_buffer,
            stop_buffer_pct=args.stop_buffer,
            use_trailing=not args.no_trailing,
            use_partial=not args.no_partial,
        )
        return engine.run(df_4h.copy(), df_15m.copy())

    if args.threshold_sweep:
        print(f"\n  MTF ORB threshold sweep — {args.symbol}")
        print(f"  {'Threshold':>10}  {'Trades':>8}  {'WinRate':>8}  "
              f"{'PF':>8}  {'Return':>9}  {'Sharpe':>8}  {'MaxDD':>8}")
        print("  " + "─" * 72)
        for t in np.arange(0.50, 0.81, 0.05):
            result = single_run(round(t, 2))
            trades = result["trades"]
            if not trades:
                print(f"  {t:>10.2f}  {'(no trades)':>8}")
                continue
            s = compute_stats(
                trades, result["equity_curve"], result["final_capital"],
                args.capital, df_15m, args.symbol,
            )
            mark = " ←" if s["profit_factor"] >= 1.0 and s["total_trades"] >= 15 else ""
            print(f"  {t:>10.2f}  {s['total_trades']:>8}  {s['win_rate_pct']:>7.1f}%  "
                  f"{s['profit_factor']:>8.3f}  {s['total_return_pct']:>+8.2f}%  "
                  f"{s['sharpe']:>8.3f}  {s['max_drawdown_pct']:>7.1f}%{mark}")
        return

    # Single run
    result = single_run(threshold)
    s      = compute_stats(
        result["trades"], result["equity_curve"], result["final_capital"],
        args.capital, df_15m, args.symbol,
    )
    print_stats(s, result)

    if args.save_trades and result["trades"]:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"backtest/output/SOL_USDT_mtf_orb_{ts}_trades.csv"
        save_trades(result["trades"], path)


if __name__ == "__main__":
    main()
