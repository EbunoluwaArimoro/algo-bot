"""
ml/generate_volatility_labels.py
──────────────────────────────────
Replaces generate_training_data.py for volatility-based prediction.

Instead of predicting direction (will price go up or down?), we predict
whether price will make a significant move in EITHER direction within N bars.

Why this works better than directional prediction:
  - Volatility clusters: high vol follows high vol (GARCH effect)
  - The features we have (funding extremes, OI spikes, volume delta)
    are better predictors of "something big is about to happen" than
    "it will be up or down"
  - This is asset-class agnostic — works on BTC, ETH, and altcoins

Label definition:
  label_volatile = 1 if max(|high - entry|, |low - entry|) > threshold
                       within forward_bars candles
  label_volatile = 0 otherwise

Threshold options:
  --vol-mult 1.5   price moves > 1.5x current ATR = volatile (default)
  --vol-pct 2.0    price moves > 2.0% = volatile (alternative)

Strategy implication:
  When model predicts volatile=1 with probability > 0.60:
    → Enter a straddle-like position (long AND short with tight correlated stops)
    → Or enter in the direction of the current trend with wider stops
    → Or simply sit out if volatility prediction is 0 (avoid choppy markets)

Usage:
    python ml/generate_volatility_labels.py \
        --csv ml/data/BTC_USDT_4h_alpha.csv

    python ml/generate_volatility_labels.py \
        --csv backtest/data/SOL_USDT_4h.csv \
        --vol-mult 1.5 --forward-bars 6
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import build_features, load_ohlcv_from_csv, assign_regime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.vol_labels")

# All features available after build_features() + alpha merge
BASE_FEATURE_COLS = [
    # Standard technical
    "rsi", "macd_hist", "macd_hist_prev", "adx",
    "ema_spread_pct", "atr_pct", "vol_ratio",
    "bb_width", "bb_squeeze",
    "ema_bull", "ema_bear", "ema_bull_4h", "ema_bear_4h", "adx_4h",
]

ALPHA_FEATURE_COLS = [
    # Funding
    "funding_rate_8h", "funding_rate_24h_sum", "funding_rate_3d_sum",
    "funding_zscore", "funding_extreme_long", "funding_extreme_short",
    # Open interest
    "oi_change_pct", "oi_change_3bar", "oi_zscore", "oi_acceleration",
    # Volume delta
    "volume_delta", "buy_ratio", "buy_ratio_ma10",
    "buy_ratio_zscore", "cum_delta_3bar", "delta_ma",
    # L/S ratio
    "ls_ratio", "ls_ratio_ma10", "ls_ratio_zscore",
    "ls_ratio_change", "crowd_extreme_long", "crowd_extreme_short",
    # Interactions
    "funding_oi_diverge", "delta_price_agree", "delta_price_diverge",
    "sentiment_agreement", "oi_delta_confirm",
    "funding_reversal_long", "funding_reversal_short",
]

REGIMES = ["ranging", "trending_bull", "trending_bear", "high_volatility"]


def label_volatility(
    i: int,
    df: pd.DataFrame,
    atr_mult: float,
    forward_bars: int,
    vol_pct: float | None,
) -> dict:
    """
    Look forward `forward_bars` bars from bar i.
    Label = 1 if price makes a significant excursion in EITHER direction.
    Also compute:
      - direction_if_volatile: which direction the move went (for strategy use)
      - max_excursion_pct: largest move seen in either direction
      - vol_magnitude: ratio of actual move to ATR threshold
    """
    row   = df.iloc[i]
    entry = row["close"]
    atr   = row.get("atr") or 0

    if atr <= 0 or entry <= 0:
        return {
            "label_volatile":       np.nan,
            "direction_if_volatile":0,
            "max_excursion_pct":    np.nan,
            "vol_magnitude":        np.nan,
        }

    # Threshold: move must exceed this to be labelled volatile
    if vol_pct is not None:
        threshold = entry * (vol_pct / 100)
    else:
        threshold = atr * atr_mult

    end = min(i + forward_bars + 1, len(df))

    max_up   = 0.0
    max_down = 0.0

    for j in range(i + 1, end):
        future    = df.iloc[j]
        up_move   = future["high"]  - entry
        down_move = entry - future["low"]
        max_up    = max(max_up,   up_move)
        max_down  = max(max_down, down_move)

    max_excursion    = max(max_up, max_down)
    max_excursion_pct= max_excursion / entry * 100
    label_volatile   = 1 if max_excursion >= threshold else 0
    vol_magnitude    = max_excursion / threshold

    # Which direction was the dominant move?
    direction_if_volatile = 1 if max_up >= max_down else -1

    return {
        "label_volatile":        label_volatile,
        "direction_if_volatile": direction_if_volatile,
        "max_excursion_pct":     round(max_excursion_pct, 4),
        "vol_magnitude":         round(vol_magnitude, 4),
    }


def generate(
    df: pd.DataFrame,
    atr_mult: float,
    forward_bars: int,
    skip_bars: int,
    vol_pct: float | None,
    has_alpha: bool,
) -> pd.DataFrame:

    feature_cols = BASE_FEATURE_COLS.copy()
    if has_alpha:
        feature_cols += [c for c in ALPHA_FEATURE_COLS if c in df.columns]

    log.info("Using %d features (%d base + %d alpha)",
             len(feature_cols),
             len(BASE_FEATURE_COLS),
             len(feature_cols) - len(BASE_FEATURE_COLS))

    records = []
    total   = len(df)

    log.info("Sweeping %d bars (every %d)…", total, skip_bars)

    for i in range(50, total - forward_bars - 1, skip_bars):
        row = df.iloc[i]

        if pd.isna(row.get("rsi")) or pd.isna(row.get("atr")):
            continue

        regime = assign_regime(row)
        labels = label_volatility(i, df, atr_mult, forward_bars, vol_pct)

        if np.isnan(labels["label_volatile"]):
            continue

        record = {
            "time":   row["time"],
            "close":  row["close"],
            "atr":    row.get("atr", 0),
            "regime": regime,
        }

        # Add regime as one-hot
        for r in REGIMES:
            record[f"regime_{r}"] = 1 if regime == r else 0

        # Add all feature values
        for col in feature_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                record[col] = 0.0
            elif isinstance(val, bool):
                record[col] = float(val)
            else:
                record[col] = float(val)

        record.update(labels)
        records.append(record)

    df_out = pd.DataFrame(records)
    log.info("Generated %d labeled bars", len(df_out))

    vol_rate = df_out["label_volatile"].mean() * 100
    log.info("Volatile bar rate: %.1f%% (base rate the model must beat)", vol_rate)

    if 35 < vol_rate < 65:
        log.info("  ✓ Good class balance for training")
    elif vol_rate < 20 or vol_rate > 80:
        log.warning("  ⚠ Imbalanced labels (%.1f%%). Try adjusting --vol-mult or --forward-bars", vol_rate)
        log.warning("  Target: 35-65%% volatile bars for best model performance")

    return df_out


def main():
    p = argparse.ArgumentParser(
        description="Generate volatility-prediction training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tuning guide for label balance (aim for 40-60% volatile bars):
  Too many volatile (>65%): increase --vol-mult or decrease --forward-bars
  Too few volatile (<35%):  decrease --vol-mult or increase --forward-bars

Recommended starting configs:
  BTC/ETH 4h:   --vol-mult 1.5 --forward-bars 6   (target: ~50% volatile)
  Altcoins 4h:  --vol-mult 1.2 --forward-bars 8   (altcoins move more)
  Any 1h:       --vol-mult 1.0 --forward-bars 12
        """
    )
    p.add_argument("--csv",          required=True,
                   help="Path to OHLCV or alpha-enriched CSV")
    p.add_argument("--output",       default=None)
    p.add_argument("--vol-mult",     type=float, default=1.5,
                   help="Volatility threshold as multiple of ATR (default 1.5)")
    p.add_argument("--vol-pct",      type=float, default=None,
                   help="Alternative: threshold as price %% move (overrides --vol-mult)")
    p.add_argument("--forward-bars", type=int,   default=6,
                   help="Bars to look forward (default 6 = 24h on 4h chart)")
    p.add_argument("--skip-bars",    type=int,   default=1,
                   help="Label every Nth bar (default 1 = all bars)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        log.error("File not found: %s", args.csv)
        sys.exit(1)

    log.info("Loading %s …", args.csv)

    # Detect whether this is an alpha CSV or plain OHLCV
    raw      = pd.read_csv(args.csv, nrows=5)
    has_alpha= "funding_rate_8h" in raw.columns
    log.info("Alpha features detected: %s", has_alpha)

    if has_alpha:
        # Alpha CSV: already has OHLCV + alpha columns merged
        df = pd.read_csv(args.csv, parse_dates=["time"])
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        df = df.sort_values("time").reset_index(drop=True)
        # Still need to compute technical indicators
        log.info("Computing technical features on alpha CSV…")
        # Extract OHLCV columns for build_features
        ohlcv_cols = ["time","open","high","low","close","volume"]
        ohlcv_only = df[ohlcv_cols].copy()
        tech_df    = build_features(ohlcv_only)
        # Merge technical features back (overwrite to get macd_hist_prev etc.)
        tech_cols  = [c for c in tech_df.columns if c not in df.columns or c == "time"]
        df = df.merge(tech_df[tech_cols], on="time", how="left")
    else:
        # Plain OHLCV: compute all technical features
        raw_df = load_ohlcv_from_csv(args.csv)
        df     = build_features(raw_df)

    df = df.dropna(subset=["atr"]).reset_index(drop=True)
    log.info("%d bars loaded", len(df))

    out_df = generate(
        df,
        atr_mult    = args.vol_mult,
        forward_bars= args.forward_bars,
        skip_bars   = args.skip_bars,
        vol_pct     = args.vol_pct,
        has_alpha   = has_alpha,
    )

    # Output path
    os.makedirs("ml/data", exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        slug     = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = f"ml/data/{slug}_vol_training.csv"

    out_df.to_csv(out_path, index=False)

    vol_rate  = out_df["label_volatile"].mean() * 100
    up_rate   = out_df.loc[out_df["label_volatile"]==1, "direction_if_volatile"].eq(1).mean() * 100

    print(f"\n{'═'*58}")
    print(f"  Volatility training data: {os.path.basename(out_path)}")
    print(f"{'═'*58}")
    print(f"  Total bars labeled:    {len(out_df):>8,}")
    print(f"  Volatile bars (1):     {out_df['label_volatile'].sum():>8,}  ({vol_rate:.1f}%)")
    print(f"  Quiet bars (0):        {(out_df['label_volatile']==0).sum():>8,}  ({100-vol_rate:.1f}%)")
    print(f"  Of volatile bars, up:  {up_rate:>7.1f}%  down: {100-up_rate:.1f}%")
    print(f"  Avg excursion (vol=1): {out_df.loc[out_df['label_volatile']==1,'max_excursion_pct'].mean():.2f}%")
    print(f"  Alpha features used:   {'yes' if has_alpha else 'no'}")
    print()
    print(f"  Label balance assessment:")
    if 35 <= vol_rate <= 65:
        print(f"  ✅ Good balance ({vol_rate:.1f}%) — ready to train")
    else:
        print(f"  ⚠  Imbalanced ({vol_rate:.1f}%) — adjust --vol-mult or --forward-bars")
        if vol_rate > 65:
            print(f"     Try: --vol-mult {args.vol_mult + 0.3:.1f} or --forward-bars {args.forward_bars - 2}")
        else:
            print(f"     Try: --vol-mult {args.vol_mult - 0.3:.1f} or --forward-bars {args.forward_bars + 2}")
    print(f"\n  Next step:")
    print(f"    python ml/train_volatility_model.py --data {out_path}")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()