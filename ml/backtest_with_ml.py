"""
ml/backtest_with_ml.py
───────────────────────
Runs the backtesting engine but replaces the rules-based SCORE_THRESHOLD
gates with XGBoost model predictions.

Instead of checking RSI < 32 or ADX > 25, the model predicts the
probability that an entry at this bar will hit TP before SL.
We only trade if that probability exceeds the threshold found during training.

This is the correct architecture: ML decides IF we trade,
the engine decides HOW MUCH we risk and WHEN we exit.

Usage:
    python ml/backtest_with_ml.py \
        --csv backtest/data/BTC_USDT_4h.csv \
        --model-dir ml/models \
        --slug BTC_USDT_4h \
        --threshold 0.60
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
except ImportError:
    print("Run: pip install xgboost")
    sys.exit(1)

from backtest.engine import (
    BacktestEngine, Trade, assign_regime,
    build_features, load_ohlcv_from_csv,
    COST_PER_SIDE, ROUND_TRIP,
)
from backtest.run_backtest import print_result, print_monte_carlo, save_trades
from backtest.engine import monte_carlo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.backtest")

REGIMES = ["ranging", "trending_bull", "trending_bear", "high_volatility"]


class MLSignalEngine(BacktestEngine):
    """
    Subclass of BacktestEngine that replaces the strategy signal function
    with ML model predictions.

    The parent class handles all trade management (stops, sizing, exits).
    This class overrides only the signal generation step.
    """

    def __init__(self, model_long, model_short, meta_long, meta_short,
                 threshold_long, threshold_short, feat_cols, **kwargs):
        # Pass a dummy strategy name — signal_fn will be overridden
        super().__init__(strategy="trend_follow", **kwargs)
        self.model_long      = model_long
        self.model_short     = model_short
        self.meta_long       = meta_long
        self.meta_short      = meta_short
        self.threshold_long  = threshold_long
        self.threshold_short = threshold_short
        self.feat_cols       = feat_cols
        self.strategy_name   = "ml_signal"

    def _get_features(self, row: pd.Series) -> np.ndarray | None:
        """Extract feature vector from a dataframe row."""
        vals = []
        for col in self.feat_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                vals.append(0.0)
            elif isinstance(val, bool):
                vals.append(float(val))
            else:
                vals.append(float(val))
        return np.array(vals).reshape(1, -1)

    def _get_regime_features(self, row: pd.Series, df_encoded: pd.DataFrame, i: int) -> np.ndarray:
        """Get feature vector including one-hot regime columns."""
        regime = assign_regime(row)
        feat_row = {}
        for col in self.feat_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                feat_row[col] = 0.0
            elif isinstance(val, bool):
                feat_row[col] = float(val)
            else:
                feat_row[col] = float(val)
        # Add regime one-hot
        for r in REGIMES:
            feat_row[f"regime_{r}"] = 1.0 if regime == r else 0.0
        return np.array([feat_row.get(c, 0.0) for c in self.feat_cols]).reshape(1, -1)

    def run(self, df: pd.DataFrame):
        """Override run to use ML predictions instead of rules-based signals."""
        df = df.dropna(subset=["ema200", "rsi", "atr", "adx"]).reset_index(drop=True)

        # Add regime one-hot columns
        for r in REGIMES:
            df[f"regime_{r}"] = df.apply(
                lambda row: 1 if assign_regime(row) == r else 0, axis=1
            )

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

            # ── Trade management (identical to parent class) ──────────────────
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

                # Breakeven stop
                if not t.breakeven_set and profit_r_high >= self.be_trigger_r:
                    buffer = risk_dist * self.be_buffer_r
                    if t.direction == "long":
                        new_be = t.entry_price + buffer
                        if new_be > t.stop_loss:
                            t.stop_loss = round(new_be, 4)
                            t.breakeven_set = True
                    else:
                        new_be = t.entry_price - buffer
                        if new_be < t.stop_loss:
                            t.stop_loss = round(new_be, 4)
                            t.breakeven_set = True

                # Partial exit
                if self.use_partial and not t.partial_done and profit_r_high >= self.partial_r:
                    partial_qty = round(t.qty * self.partial_pct, 6)
                    if partial_qty > 0:
                        exit_p = close * (1 - COST_PER_SIDE) if t.direction == "long" \
                                 else close * (1 + COST_PER_SIDE)
                        part_pnl = ((exit_p - t.entry_price) * partial_qty if t.direction == "long"
                                    else (t.entry_price - exit_p) * partial_qty)
                        portfolio += part_pnl
                        t.qty -= partial_qty
                        t.partial_done = True

                # Trailing stop
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

                active_sl = t.trail_stop if t.trail_stop > 0 else t.stop_loss
                reason = None
                exit_ref = 0.0

                if t.direction == "long":
                    if low <= active_sl:
                        exit_ref = active_sl
                        reason = "trail_stop" if t.trail_stop > 0 else (
                            "breakeven" if t.breakeven_set else "stop_loss")
                    elif high >= t.take_profit:
                        exit_ref = t.take_profit
                        reason = "take_profit"
                else:
                    if high >= active_sl:
                        exit_ref = active_sl
                        reason = "trail_stop" if t.trail_stop > 0 else (
                            "breakeven" if t.breakeven_set else "stop_loss")
                    elif low <= t.take_profit:
                        exit_ref = t.take_profit
                        reason = "take_profit"

                if reason is None:
                    still_open.append(t)
                    continue

                if t.direction == "long":
                    exit_net = exit_ref * (1 - COST_PER_SIDE)
                    pnl = (exit_net - t.entry_price) * t.qty
                else:
                    exit_net = exit_ref * (1 + COST_PER_SIDE)
                    pnl = (t.entry_price - exit_net) * t.qty

                cost = t.qty * t.entry_price
                t.exit_time  = row["time"]
                t.exit_price = round(exit_net, 4)
                t.pnl        = round(pnl, 4)
                t.pnl_pct    = round((pnl / cost) * 100, 3) if cost else 0
                t.exit_reason= reason
                portfolio   += pnl
                if pnl < 0:
                    last_loss_bar = i
                trades.append(t)

            open_trades = still_open

            # ── ML signal generation ──────────────────────────────────────────
            if len(open_trades) >= self.max_open:
                equity_curve.append(portfolio)
                continue
            if (i - last_signal_bar) < self.cooldown_bars:
                equity_curve.append(portfolio)
                continue
            if (i - last_loss_bar) < self.loss_cooldown:
                equity_curve.append(portfolio)
                continue

            # Get feature vector for this bar
            X = np.array([[
                row.get(c, 0.0) if not (isinstance(row.get(c), float) and
                                        np.isnan(row.get(c, 0.0))) else 0.0
                for c in self.feat_cols
            ]])

            direction = None
            score     = 0.0

            # Check long model
            if self.model_long is not None:
                prob_long = self.model_long.predict_proba(X)[0][1]
                if prob_long >= self.threshold_long:
                    direction = "long"
                    score     = round(prob_long * 100, 1)

            # Check short model (only if no long signal)
            if direction is None and self.model_short is not None:
                prob_short = self.model_short.predict_proba(X)[0][1]
                if prob_short >= self.threshold_short:
                    direction = "short"
                    score     = round(prob_short * 100, 1)

            if direction is None:
                equity_curve.append(portfolio)
                continue

            qty, sl, tp, _ = self._size_position(portfolio, close, atr)
            if qty <= 0:
                equity_curve.append(portfolio)
                continue

            if direction == "short":
                sl = close + (close - sl)
                tp = close - (tp - close)

            entry_net = (close * (1 + COST_PER_SIDE) if direction == "long"
                         else close * (1 - COST_PER_SIDE))

            regime = assign_regime(row)
            t = Trade(
                symbol=self.symbol, direction=direction,
                strategy="ml_signal", regime=regime,
                entry_time=row["time"],
                entry_price=round(entry_net, 4),
                qty=qty, stop_loss=sl, take_profit=tp,
                trail_stop=0.0, breakeven_set=False, partial_done=False,
                risk_pct=self.risk_per_trade * 100,
                commission=round(qty * close * ROUND_TRIP, 4),
                score=score,
            )
            open_trades.append(t)
            last_signal_bar = i
            equity_curve.append(portfolio)

        # Force-close remaining
        for t in open_trades:
            lc = df["close"].iloc[-1]
            pnl = (lc * (1 - COST_PER_SIDE) - t.entry_price) * t.qty if t.direction == "long" \
                  else (t.entry_price - lc * (1 + COST_PER_SIDE)) * t.qty
            t.exit_time  = df["time"].iloc[-1]
            t.exit_price = round(lc, 4)
            cost = t.qty * t.entry_price
            t.pnl = round(pnl, 4)
            t.pnl_pct = round((pnl / cost) * 100, 3) if cost else 0
            t.exit_reason = "end_of_data"
            portfolio += pnl
            trades.append(t)

        return self._compute_stats(trades, equity_curve, portfolio, df)


def load_model_and_meta(model_dir: str, slug: str, direction: str):
    model_path = os.path.join(model_dir, f"{slug}_{direction}.json")
    meta_path  = model_path.replace(".json", "_meta.json")

    if not os.path.exists(model_path):
        log.warning("Model not found: %s", model_path)
        return None, None, None

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    log.info("Loaded %s model — AUC %.3f, threshold %.2f",
             direction, meta.get("auc_roc", 0), meta.get("recommended_threshold", 0.5))
    return model, meta, meta.get("feature_cols", [])


def main():
    p = argparse.ArgumentParser(description="ML-powered backtest")
    p.add_argument("--csv",        required=True)
    p.add_argument("--model-dir",  default="ml/models")
    p.add_argument("--slug",       required=True,
                   help="Model filename slug e.g. BTC_USDT_4h")
    p.add_argument("--symbol",     default="BTC/USDT")
    p.add_argument("--timeframe",  default="4h")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Override model threshold (default: use recommended from training)")
    p.add_argument("--capital",    type=float, default=10_000.0)
    p.add_argument("--risk",       type=float, default=1.0)
    p.add_argument("--atr-mult",   type=float, default=1.5)
    p.add_argument("--rr",         type=float, default=2.0)
    p.add_argument("--save-trades",action="store_true")
    args = p.parse_args()

    log.info("Loading models from %s/%s_*.json", args.model_dir, args.slug)
    model_long,  meta_long,  feat_long  = load_model_and_meta(args.model_dir, args.slug, "long")
    model_short, meta_short, feat_short = load_model_and_meta(args.model_dir, args.slug, "short")

    if model_long is None and model_short is None:
        log.error("No models found. Run train_model.py first.")
        sys.exit(1)

    feat_cols = feat_long or feat_short
    thresh_l  = args.threshold or (meta_long.get("recommended_threshold", 0.55) if meta_long else 0.55)
    thresh_s  = args.threshold or (meta_short.get("recommended_threshold", 0.55) if meta_short else 0.55)

    log.info("Thresholds — long: %.2f  short: %.2f", thresh_l, thresh_s)

    log.info("Loading data: %s", args.csv)
    raw = load_ohlcv_from_csv(args.csv)
    df  = build_features(raw)

    engine = MLSignalEngine(
        model_long=model_long,   model_short=model_short,
        meta_long=meta_long,     meta_short=meta_short,
        threshold_long=thresh_l, threshold_short=thresh_s,
        feat_cols=feat_cols,
        symbol=args.symbol,      timeframe=args.timeframe,
        initial_capital=args.capital,
        risk_per_trade=args.risk / 100,
        atr_mult=args.atr_mult,  rr=args.rr,
        use_trailing=False, use_partial=False,   # start simple
    )

    result = engine.run(df.copy())
    print_result(result, "ML Signal Strategy")

    if args.save_trades and result.trades:
        from datetime import datetime
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"backtest/output/{args.slug}_ml_{ts}_trades.csv"
        save_trades(result, path)

    if result.trades:
        mc = monte_carlo(result.trades, args.capital, n_simulations=2000)
        print_monte_carlo(mc, args.symbol)


if __name__ == "__main__":
    main()