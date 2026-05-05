"""
ml/generate_training_data.py
─────────────────────────────
Generates a labeled ML training dataset from OHLCV CSV data.

Approach: instead of using the rules-based strategy to generate entry signals
(which produces too few trades for ML), this script sweeps EVERY bar and
records the full feature vector. It then looks forward N bars to determine
whether a long or short entry at that bar would have been profitable given
a fixed ATR-based stop and target.

This produces 1000-3000 labeled examples from 3 years of 4h data —
enough to train a robust XGBoost classifier.

The ML model then learns WHICH feature combinations predict profitable
outcomes, replacing our hand-written scoring logic entirely.

Output columns:
  - All technical features at bar time (RSI, ADX, MACD, BB, etc.)
  - regime (ranging / trending_bull / trending_bear / high_volatility)
  - label_long  : 1 if a long entry here would have hit TP before SL
  - label_short : 1 if a short entry here would have hit TP before SL
  - forward_return_pct: actual % return over next N bars (for regression)

Usage:
    python ml/generate_training_data.py --csv backtest/data/BTC_USDT_4h.csv
    python ml/generate_training_data.py --csv backtest/data/ETH_USDT_4h.csv
    python ml/generate_training_data.py --csv backtest/data/BTC_USDT_4h.csv --atr-mult 1.5 --rr 2.0 --forward-bars 10
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import build_features, assign_regime, load_ohlcv_from_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.data_gen")

COST_PER_SIDE = 0.0018   # matches engine.py


def label_bar(
    i: int,
    df: pd.DataFrame,
    atr_mult: float,
    rr: float,
    forward_bars: int,
) -> dict:
    """
    For bar i, simulate both a long and short entry using ATR-based SL/TP.
    Look forward up to `forward_bars` bars to see which level gets hit first.

    Returns a dict with label_long, label_short, and forward_return_pct.
    """
    row   = df.iloc[i]
    entry = row["close"]
    atr   = row.get("atr") or 0

    if atr <= 0 or entry <= 0:
        return {"label_long": np.nan, "label_short": np.nan,
                "forward_return_pct": np.nan, "forward_bars_used": np.nan}

    stop_dist = atr * atr_mult
    long_sl   = entry - stop_dist
    long_tp   = entry + stop_dist * rr
    short_sl  = entry + stop_dist
    short_tp  = entry - stop_dist * rr

    # Apply entry cost
    long_entry  = entry * (1 + COST_PER_SIDE)
    short_entry = entry * (1 - COST_PER_SIDE)

    label_long  = np.nan
    label_short = np.nan
    forward_return = np.nan
    bars_used   = np.nan

    end = min(i + forward_bars + 1, len(df))

    for j in range(i + 1, end):
        future = df.iloc[j]
        h = future["high"]
        l = future["low"]

        # Resolve long first (check SL before TP within same bar)
        if label_long is np.nan:
            if l <= long_sl:
                exit_p = long_sl * (1 - COST_PER_SIDE)
                label_long = 0
            elif h >= long_tp:
                exit_p = long_tp * (1 - COST_PER_SIDE)
                label_long = 1

        # Resolve short
        if label_short is np.nan:
            if h >= short_sl:
                label_short = 0
            elif l <= short_tp:
                label_short = 1

        if label_long is not np.nan and label_short is not np.nan:
            bars_used = j - i
            break

    # Forward return: simple close-to-close over forward_bars
    if i + forward_bars < len(df):
        future_close    = df.iloc[i + forward_bars]["close"]
        forward_return  = (future_close - entry) / entry * 100

    # If still unresolved at end of window, use final close as exit
    if label_long is np.nan:
        if i + forward_bars < len(df):
            fc = df.iloc[i + forward_bars]["close"] * (1 - COST_PER_SIDE)
            label_long = 1 if fc > long_entry else 0
        else:
            label_long = 0

    if label_short is np.nan:
        if i + forward_bars < len(df):
            fc = df.iloc[i + forward_bars]["close"] * (1 + COST_PER_SIDE)
            label_short = 1 if fc < short_entry else 0
        else:
            label_short = 0

    return {
        "label_long":         int(label_long),
        "label_short":        int(label_short),
        "forward_return_pct": round(float(forward_return), 4) if not np.isnan(forward_return) else 0.0,
        "forward_bars_used":  int(bars_used) if not np.isnan(bars_used) else forward_bars,
    }


# Feature columns to include in training data
# These are the model inputs — everything computable at signal time
FEATURE_COLS = [
    "rsi", "macd_hist", "macd_hist_prev", "adx",
    "ema_spread_pct", "atr_pct", "vol_ratio",
    "bb_width", "bb_squeeze",
    "ema_bull", "ema_bear", "ema_bull_4h", "ema_bear_4h", "adx_4h",
]


def generate(
    df: pd.DataFrame,
    atr_mult: float,
    rr: float,
    forward_bars: int,
    min_adx: float,
    skip_bars: int,
) -> pd.DataFrame:
    """
    Sweep every `skip_bars`-th bar in df and generate labels.
    skip_bars=1 means every bar (maximum data, more correlation between rows).
    skip_bars=2 means every other bar (less correlation, still large dataset).

    min_adx: skip bars where there is essentially no market structure at all.
    Even for ML data gen we don't want to label bars during total dead markets.
    """
    records = []
    total   = len(df)

    log.info("Sweeping %d bars (every %d, skip ADX < %.0f)…",
             total, skip_bars, min_adx)

    for i in range(50, total - forward_bars - 1, skip_bars):
        row = df.iloc[i]

        # Skip bars with missing core features
        if pd.isna(row.get("rsi")) or pd.isna(row.get("atr")) or pd.isna(row.get("adx")):
            continue

        # Skip total dead-market bars (very low ADX AND very low volume)
        # These produce un-labelable noise even for ML
        adx = row.get("adx") or 0
        vol = row.get("vol_ratio") or 1.0
        if adx < min_adx and vol < 0.5:
            continue

        regime = assign_regime(row)
        labels = label_bar(i, df, atr_mult, rr, forward_bars)

        record = {
            "time":    row["time"],
            "close":   row["close"],
            "regime":  regime,
        }

        # Add all feature columns (with safe fallback to 0)
        exclude = ["time", "open", "high", "low", "close", "volume", "regime"]
        feature_cols = [c for c in df.columns if c not in exclude]
        for col in feature_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                record[col] = 0.0
            elif isinstance(val, bool):
                record[col] = int(val)
            else:
                record[col] = float(val)

        record.update(labels)
        records.append(record)

    df_out = pd.DataFrame(records)
    log.info("Generated %d labeled bars", len(df_out))

    # Label distribution
    long_wr  = df_out["label_long"].mean()  * 100
    short_wr = df_out["label_short"].mean() * 100
    log.info("Long win rate across all bars:  %.1f%%", long_wr)
    log.info("Short win rate across all bars: %.1f%%", short_wr)
    log.info("Class balance note: ~%.0f%% of bars are naturally unprofitable",
             100 - (long_wr + short_wr) / 2)

    return df_out


def main():
    p = argparse.ArgumentParser(description="Generate ML training data from OHLCV CSV")
    p.add_argument("--csv",          required=True,  help="Path to OHLCV CSV")
    p.add_argument("--output",       default=None,   help="Output CSV path (auto-named if omitted)")
    p.add_argument("--atr-mult",     type=float, default=1.5)
    p.add_argument("--rr",           type=float, default=2.0)
    p.add_argument("--forward-bars", type=int,   default=10,
                   help="Max bars to look forward for SL/TP resolution (default 10 = ~40h on 4h chart)")
    p.add_argument("--skip-bars",    type=int,   default=2,
                   help="Label every Nth bar. 1=all, 2=every other (default 2)")
    p.add_argument("--min-adx",      type=float, default=10.0,
                   help="Minimum ADX to include a bar (default 10 — very loose)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        log.error("CSV not found: %s", args.csv)
        sys.exit(1)

    log.info("Loading %s …", args.csv)
    raw = load_ohlcv_from_csv(args.csv)

    log.info("Computing features…")
    df = build_features(raw)
    df = df.dropna(subset=["ema200", "atr"]).reset_index(drop=True)
    log.info("%d bars after feature computation", len(df))

    out_df = generate(
        df,
        atr_mult=args.atr_mult,
        rr=args.rr,
        forward_bars=args.forward_bars,
        min_adx=args.min_adx,
        skip_bars=args.skip_bars,
    )

    # Output path
    if args.output:
        out_path = args.output
    else:
        slug = os.path.splitext(os.path.basename(args.csv))[0]
        os.makedirs("ml/data", exist_ok=True)
        out_path = f"ml/data/{slug}_training.csv"

    out_df.to_csv(out_path, index=False)
    log.info("Training data saved → %s  (%d rows, %d columns)",
             out_path, len(out_df), len(out_df.columns))

    # Print a readable summary
    print("\n" + "═" * 55)
    print(f"  Training data summary: {os.path.basename(out_path)}")
    print("═" * 55)
    print(f"  Total labeled bars:    {len(out_df):>8,}")
    print(f"  Date range:            {str(out_df['time'].min())[:10]} → {str(out_df['time'].max())[:10]}")
    print(f"  Long TP hit rate:      {out_df['label_long'].mean()*100:>7.1f}%  (base rate)")
    print(f"  Short TP hit rate:     {out_df['label_short'].mean()*100:>7.1f}%  (base rate)")
    print(f"  ATR mult / RR:         {args.atr_mult} / {args.rr}")
    print(f"  Forward window:        {args.forward_bars} bars")
    print()
    print("  Regime distribution:")
    for regime, count in out_df["regime"].value_counts().items():
        pct = count / len(out_df) * 100
        print(f"    {regime:20}  {count:>5}  ({pct:.1f}%)")
    print("═" * 55)
    print(f"\n  Next step:")
    print(f"    python ml/train_model.py --data {out_path}")
    print()


if __name__ == "__main__":
    main()