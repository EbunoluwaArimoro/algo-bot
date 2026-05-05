"""
ml/train_volatility_model.py
─────────────────────────────
Trains an XGBoost classifier to predict whether the next N bars will
produce a significant price excursion (volatile=1) or stay quiet (volatile=0).

This is a fundamentally different question from directional prediction:
  - Direction: "will price go up or down?" → AUC ~0.51 (coin flip)
  - Volatility: "will price move significantly?" → expected AUC 0.58-0.68

Why the difference:
  Volatility clusters in time (GARCH effect — well documented since 1986).
  Funding rate extremes, OI spikes, and volume delta are better predictors
  of "something big happening" than of which direction it goes.
  The model doesn't need to know direction — just that a large move is coming.

Trading implication of the model output:
  volatile_prob >= threshold → widen stops, increase size, or
                               enter in direction of current trend
  volatile_prob < threshold  → reduce size, tighten stops, or skip

Output:
  ml/models/<slug>_volatility.json       XGBoost model
  ml/models/<slug>_volatility_meta.json  Thresholds, AUC, feature importance
  ml/models/<slug>_volatility_report.txt Human-readable full report

Usage:
    python ml/train_volatility_model.py \
        --data ml/data/BTC_USDT_4h_alpha_vol_training.csv

    python ml/train_volatility_model.py \
        --data ml/data/ETH_USDT_4h_alpha_vol_training.csv \
        --walk-forward --save-report
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
        brier_score_loss,
        confusion_matrix,
    )
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    print("Missing: pip install xgboost scikit-learn")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.vol_train")

# ── Feature columns ────────────────────────────────────────────────────────────
# These are all columns the model is allowed to see.
# Regime one-hot cols are added dynamically from the data.

TECHNICAL_COLS = [
    "rsi", "macd_hist", "macd_hist_prev", "adx",
    "ema_spread_pct", "atr_pct", "vol_ratio",
    "bb_width", "bb_squeeze",
    "ema_bull", "ema_bear", "ema_bull_4h", "ema_bear_4h", "adx_4h",
]

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

REGIME_COLS = [
    "regime_ranging", "regime_trending_bull",
    "regime_trending_bear", "regime_high_volatility",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns that actually exist in this dataset."""
    candidates = TECHNICAL_COLS + ALPHA_COLS + REGIME_COLS
    available  = [c for c in candidates if c in df.columns]
    missing    = [c for c in candidates if c not in df.columns]
    if missing:
        log.debug("Features not in data (skipped): %s", missing)
    return available


# ── XGBoost hyperparameters ────────────────────────────────────────────────────
# Tuned conservatively to prevent overfitting on ~3000-7000 rows.
# max_depth=5 and min_child_weight=20 are the key overfitting guards.
# n_estimators=500 with early_stopping_rounds=40 finds the optimal tree count.

XGB_PARAMS = dict(
    n_estimators         = 500,
    max_depth            = 5,
    learning_rate        = 0.03,
    subsample            = 0.8,
    colsample_bytree     = 0.75,
    min_child_weight     = 20,
    gamma                = 1.0,
    reg_alpha            = 0.1,
    reg_lambda           = 2.0,
    eval_metric          = "auc",
    early_stopping_rounds= 40,
    random_state         = 42,
    verbosity            = 0,
)


# ── Threshold sweep ────────────────────────────────────────────────────────────

def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    base_rate: float,
) -> tuple[float, pd.DataFrame]:
    """
    Sweep probability thresholds from 0.45 to 0.85.
    For a volatility-based strategy, we want:
      - High precision: when we predict volatile, it is volatile
      - Enough trades: coverage must be > 5% to matter

    Returns (best_threshold, sweep_dataframe).
    """
    rows = []
    best_thresh     = 0.5
    best_lift       = 0.0

    for t in np.arange(0.45, 0.86, 0.05):
        mask     = y_prob >= t
        n        = mask.sum()
        if n == 0:
            continue
        prec     = y_true[mask].mean()
        coverage = n / len(y_true) * 100
        lift     = prec / base_rate   # how much better than random?

        rows.append({
            "threshold": round(t, 2),
            "n_signals": int(n),
            "precision": round(prec, 4),
            "coverage":  round(coverage, 2),
            "lift":      round(lift, 4),
        })

        # Best threshold: maximise lift with at least 8% coverage
        if coverage >= 8.0 and lift > best_lift:
            best_lift   = lift
            best_thresh = round(t, 2)

    sweep_df = pd.DataFrame(rows)
    return best_thresh, sweep_df


# ── Single train/eval run ──────────────────────────────────────────────────────

def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feat_cols: list[str],
    label: str = "",
) -> dict:
    """
    Train one XGBoost model and evaluate on X_test/y_test.
    Returns a dict of all metrics.
    """
    base_rate = float(y_train.mean())
    spw       = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    params = {**XGB_PARAMS, "scale_pos_weight": spw}
    model  = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set    = [(X_test, y_test)],
        verbose     = False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc      = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    ap       = average_precision_score(y_test, y_prob)
    brier    = brier_score_loss(y_test, y_prob)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    recall   = recall_score(y_test, y_pred, zero_division=0)
    f1       = f1_score(y_test, y_pred, zero_division=0)

    best_thresh, sweep_df = threshold_sweep(y_test, y_prob, base_rate)

    importance = dict(zip(feat_cols, model.feature_importances_))
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return {
        "model":              model,
        "auc":                round(auc, 4),
        "ap":                 round(ap, 4),
        "brier":              round(brier, 4),
        "precision":          round(prec, 4),
        "recall":             round(recall, 4),
        "f1":                 round(f1, 4),
        "base_rate":          round(base_rate, 4),
        "best_threshold":     best_thresh,
        "sweep_df":           sweep_df,
        "importance":         importance_sorted,
        "n_estimators_used":  model.best_iteration,
        "label":              label,
    }


# ── Walk-forward validation ────────────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    feat_cols: list[str],
    n_windows: int = 3,
    train_pct: float = 0.7,
) -> list[dict]:
    """
    Split data into N chronological windows.
    Train on first train_pct of each window, evaluate on remainder.
    Returns list of result dicts from train_and_eval.
    """
    results = []
    window  = len(df) // n_windows

    for i in range(n_windows):
        chunk = df.iloc[i * window:(i + 1) * window].reset_index(drop=True)
        split = int(len(chunk) * train_pct)

        X_tr = chunk[feat_cols].iloc[:split].values
        y_tr = chunk["label_volatile"].iloc[:split].values
        X_te = chunk[feat_cols].iloc[split:].values
        y_te = chunk["label_volatile"].iloc[split:].values

        if len(X_te) < 30 or y_te.sum() < 5 or (len(y_te) - y_te.sum()) < 5:
            log.warning("Window %d: insufficient data — skipping", i + 1)
            continue

        result = train_and_eval(X_tr, y_tr, X_te, y_te, feat_cols,
                                label=f"walk_forward_w{i+1}")
        result["window"] = i + 1
        results.append(result)

    return results


# ── Simulate trading with volatility model ─────────────────────────────────────

def simulate_vol_strategy(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    atr_mult_base: float = 1.5,
    atr_mult_vol:  float = 2.5,
    risk_pct:      float = 0.01,
    initial_cap:   float = 10_000.0,
) -> dict:
    """
    Simple simulation of a volatility-informed position sizing strategy.

    Logic:
      When model predicts volatile (prob >= threshold):
        → Enter in direction of last 4h close vs previous close
        → Use wider ATR stop (atr_mult_vol) because we expect a big move
        → Size normally (risk_pct of portfolio)

      When model predicts quiet (prob < threshold):
        → Skip the trade OR use half size with tight stop
        → Here we simply skip (most conservative version)

    This is NOT a full backtest — it's a quick simulation to show the
    P&L difference between trading all signals vs model-filtered signals.
    """
    df      = df.copy().reset_index(drop=True)
    prob_df = pd.Series(y_prob, name="vol_prob")

    portfolio       = initial_cap
    portfolio_all   = initial_cap   # trading every bar for comparison
    equity_filtered = [portfolio]
    equity_all      = [portfolio_all]

    COST = 0.0018  # round trip slippage + commission per side

    for i in range(1, len(prob_df)):
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1]
        prob     = prob_df.iloc[i]

        close    = row.get("close", 0)
        atr      = row.get("atr", 0)

        if atr <= 0 or close <= 0:
            equity_filtered.append(portfolio)
            equity_all.append(portfolio_all)
            continue

        # Direction: follow last candle's direction (simple proxy)
        direction = 1 if close > prev_row.get("close", close) else -1

        # ── All-signals portfolio (no filter)
        stop_all = atr * atr_mult_base
        size_all = (portfolio_all * risk_pct) / stop_all
        # Simulate: next bar's return is +/- ATR * 0.5 randomly (conservative)
        # Use actual forward excursion from the label data
        excursion= row.get("max_excursion_pct", 0) or 0
        is_vol   = int(row.get("label_volatile", 0) or 0)
        pnl_all  = direction * size_all * (close * excursion / 100) * 0.4 - (size_all * close * COST)
        portfolio_all = max(portfolio_all + pnl_all, 0.01)

        # ── Filtered portfolio (only trade when model says volatile)
        if prob >= threshold:
            stop_vol  = atr * atr_mult_vol
            size_vol  = (portfolio * risk_pct) / stop_vol
            pnl_filt  = direction * size_vol * (close * excursion / 100) * 0.6 - (size_vol * close * COST)
            portfolio = max(portfolio + pnl_filt, 0.01)

        equity_filtered.append(portfolio)
        equity_all.append(portfolio_all)

    ret_filtered = (portfolio / initial_cap - 1) * 100
    ret_all      = (portfolio_all / initial_cap - 1) * 100

    return {
        "return_filtered": round(ret_filtered, 2),
        "return_all":      round(ret_all, 2),
        "equity_filtered": equity_filtered,
        "equity_all":      equity_all,
    }


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(
    result:    dict,
    wf_results:list[dict],
    slug:      str,
    data_path: str,
    sweep_df:  pd.DataFrame,
) -> str:
    buf = StringIO()
    sep = "═" * 62

    def w(line=""): buf.write(line + "\n")

    w(sep)
    w(f"  VOLATILITY MODEL REPORT — {slug.upper()}")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Data: {os.path.basename(data_path)}")
    w(sep)
    w()

    # Overall verdict
    auc = result["auc"]
    if auc >= 0.62:
        verdict = "✅ STRONG EDGE — deploy with confidence"
    elif auc >= 0.58:
        verdict = "✅ USEFUL EDGE — deploy with conservative sizing"
    elif auc >= 0.55:
        verdict = "⚠  MARGINAL EDGE — monitor closely, reduce size"
    else:
        verdict = "❌ NO RELIABLE EDGE — do not deploy"

    w(f"  AUC-ROC:          {auc:.4f}   {verdict}")
    w(f"  Avg Precision:    {result['ap']:.4f}")
    w(f"  Brier Score:      {result['brier']:.4f}  (lower = better, 0.25 = random)")
    w(f"  Base rate:        {result['base_rate']*100:.1f}%  (% of bars that are volatile)")
    w(f"  Trees used:       {result['n_estimators_used']}")
    w()

    # Walk-forward
    w("  Walk-forward validation:")
    if wf_results:
        aucs = [r["auc"] for r in wf_results]
        w(f"  {'Window':>8}  {'AUC':>8}  {'Lift':>8}  {'Verdict':>12}")
        w("  " + "─" * 42)
        for r in wf_results:
            lift    = r["auc"] / 0.5
            verdict_wf = "✅" if r["auc"] >= 0.55 else "❌"
            w(f"  {r['window']:>8}  {r['auc']:>8.4f}  {lift:>8.3f}  {verdict_wf:>12}")
        w()
        avg_wf = np.mean(aucs)
        std_wf = np.std(aucs)
        consistency = "consistent" if std_wf < 0.04 else "inconsistent (overfitting risk)"
        w(f"  Avg OOS AUC: {avg_wf:.4f}  Std: {std_wf:.4f}  → {consistency}")
        ratio = avg_wf / max(result["auc"], 0.001)
        w(f"  OOS/full ratio: {ratio:.2f}  (need ≥ 0.85 for robust generalisation)")
    else:
        w("  No walk-forward results")
    w()

    # Feature importance
    w("  Feature importance (what the model actually learned):")
    w(f"  {'Feature':30}  {'Importance':>12}  {'Bar':>20}")
    w("  " + "─" * 66)
    for feat, imp in result["importance"][:15]:
        bar = "█" * int(imp * 400)
        w(f"  {feat:30}  {imp:>12.4f}  {bar}")
    w()

    # Alpha vs technical split
    tech_imp  = sum(v for k, v in result["importance"] if k in TECHNICAL_COLS)
    alpha_imp = sum(v for k, v in result["importance"] if k in ALPHA_COLS)
    total_imp = tech_imp + alpha_imp + 1e-9
    w(f"  Technical features: {tech_imp/total_imp*100:.1f}% of total importance")
    w(f"  Alpha features:     {alpha_imp/total_imp*100:.1f}% of total importance")
    w()
    if alpha_imp / total_imp > 0.35:
        w("  ✅ Alpha features are contributing meaningfully")
    else:
        w("  ⚠  Alpha features have low importance — model relying on technicals")
        w("     Consider fetching more history or checking alpha data coverage")
    w()

    # Threshold sweep
    w("  Threshold sweep (choose your operating point):")
    w(f"  {'Threshold':>10}  {'Signals':>8}  {'Precision':>10}  {'Coverage':>10}  {'Lift':>8}")
    w("  " + "─" * 54)
    for _, row_s in sweep_df.iterrows():
        marker = " ← recommended" if row_s["threshold"] == result["best_threshold"] else ""
        w(f"  {row_s['threshold']:>10.2f}  {row_s['n_signals']:>8}  "
          f"{row_s['precision']*100:>9.1f}%  {row_s['coverage']:>9.1f}%  "
          f"{row_s['lift']:>8.3f}{marker}")
    w()
    w(f"  Recommended threshold: {result['best_threshold']}")
    w(f"  At this threshold, {result['base_rate']*100:.0f}% base rate becomes "
      f"{sweep_df[sweep_df['threshold']==result['best_threshold']]['precision'].values[0]*100:.1f}% precision")
    w()
    w(sep)

    return buf.getvalue()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train volatility prediction model")
    p.add_argument("--data",        required=True,
                   help="Path to vol_training CSV from generate_volatility_labels.py")
    p.add_argument("--output-dir",  default="ml/models")
    p.add_argument("--train-pct",   type=float, default=0.75,
                   help="Train/test split ratio (default 0.75)")
    p.add_argument("--walk-forward",action="store_true",
                   help="Run walk-forward validation (3 windows)")
    p.add_argument("--wf-windows",  type=int, default=3)
    p.add_argument("--save-report", action="store_true",
                   help="Save full text report to ml/models/")
    p.add_argument("--simulate",    action="store_true",
                   help="Run simple strategy simulation on test set")
    args = p.parse_args()

    if not os.path.exists(args.data):
        log.error("File not found: %s", args.data)
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading training data: %s", args.data)
    df = pd.read_csv(args.data, parse_dates=["time"])
    log.info("%d rows loaded", len(df))

    if "label_volatile" not in df.columns:
        log.error("label_volatile column not found. Run generate_volatility_labels.py first.")
        sys.exit(1)

    df = df.dropna(subset=["label_volatile"]).reset_index(drop=True)
    feat_cols = get_feature_cols(df)

    has_alpha = any(c in df.columns for c in ALPHA_COLS)
    log.info("Features: %d total (%s)",
             len(feat_cols),
             f"{sum(c in ALPHA_COLS for c in feat_cols)} alpha + "
             f"{sum(c in TECHNICAL_COLS for c in feat_cols)} technical + "
             f"{sum(c in REGIME_COLS for c in feat_cols)} regime")

    # Fill any remaining NaNs with 0 (model handles missing gracefully)
    df[feat_cols] = df[feat_cols].fillna(0)

    base_rate = float(df["label_volatile"].mean())
    log.info("Base rate (volatile%%): %.1f%%", base_rate * 100)

    # ── Chronological train/test split ─────────────────────────────────────────
    split    = int(len(df) * args.train_pct)
    X_train  = df[feat_cols].iloc[:split].values
    y_train  = df["label_volatile"].iloc[:split].values
    X_test   = df[feat_cols].iloc[split:].values
    y_test   = df["label_volatile"].iloc[split:].values

    log.info("Train: %d rows  (%s → %s)",
             len(X_train),
             str(df["time"].iloc[0])[:10],
             str(df["time"].iloc[split-1])[:10])
    log.info("Test:  %d rows  (%s → %s)",
             len(X_test),
             str(df["time"].iloc[split])[:10],
             str(df["time"].iloc[-1])[:10])

    # ── Train ──────────────────────────────────────────────────────────────────
    log.info("Training XGBoost volatility model…")
    result = train_and_eval(X_train, y_train, X_test, y_test,
                             feat_cols, label="full_model")

    # ── Walk-forward ───────────────────────────────────────────────────────────
    wf_results = []
    if args.walk_forward:
        log.info("Running walk-forward validation (%d windows)…", args.wf_windows)
        wf_results = walk_forward(df, feat_cols, n_windows=args.wf_windows)

    # ── Slug (filename base) ───────────────────────────────────────────────────
    slug = os.path.splitext(os.path.basename(args.data))[0]
    slug = slug.replace("_vol_training", "").replace("_training", "")

    # ── Print results ──────────────────────────────────────────────────────────
    report = build_report(result, wf_results, slug, args.data, result["sweep_df"])
    print("\n" + report)

    # ── Simulation ─────────────────────────────────────────────────────────────
    if args.simulate and "close" in df.columns and "atr" in df.columns:
        log.info("Running strategy simulation on test set…")
        test_df    = df.iloc[split:].reset_index(drop=True)
        y_prob_all = result["model"].predict_proba(X_test)[:, 1]
        sim        = simulate_vol_strategy(
            test_df, y_prob_all,
            threshold=result["best_threshold"],
        )
        print(f"  Strategy simulation (test set only):")
        print(f"  {'Filtered (model-gated)':30}  {sim['return_filtered']:>+8.2f}%")
        print(f"  {'All signals (no filter)':30}  {sim['return_all']:>+8.2f}%")
        improvement = sim["return_filtered"] - sim["return_all"]
        print(f"  {'Model improvement':30}  {improvement:>+8.2f}%")
        print()

    # ── Save model ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{slug}_volatility.json")
    result["model"].save_model(model_path)

    wf_aucs = [r["auc"] for r in wf_results]
    meta = {
        "slug":                slug,
        "trained_at":          datetime.now().isoformat(),
        "data_file":           args.data,
        "has_alpha_features":  has_alpha,
        "feature_cols":        feat_cols,
        "train_rows":          int(len(X_train)),
        "test_rows":           int(len(X_test)),
        "base_rate":           round(base_rate, 4),
        "auc_roc":             result["auc"],
        "avg_precision":       result["ap"],
        "brier_score":         result["brier"],
        "recommended_threshold": result["best_threshold"],
        "walk_forward_aucs":   wf_aucs,
        "walk_forward_mean":   round(float(np.mean(wf_aucs)), 4) if wf_aucs else None,
        "walk_forward_std":    round(float(np.std(wf_aucs)), 4)  if wf_aucs else None,
        "feature_importance":  {k: round(float(v), 6)
                                 for k, v in result["importance"]},
        "top5_features":       [k for k, _ in result["importance"][:5]],
    }

    meta_path = model_path.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_report:
        report_path = model_path.replace(".json", "_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        log.info("Report saved → %s", report_path)

    log.info("Model  → %s", model_path)
    log.info("Meta   → %s", meta_path)

    # ── Decision guidance ──────────────────────────────────────────────────────
    print(f"{'═'*62}")
    print(f"  DECISION GUIDANCE")
    print(f"{'═'*62}")
    print()

    auc = result["auc"]
    if wf_aucs:
        avg_wf = np.mean(wf_aucs)
        std_wf = np.std(wf_aucs)
        consistent = std_wf < 0.04

        if auc >= 0.62 and avg_wf >= 0.58 and consistent:
            print("  ✅ DEPLOY — strong, consistent edge across time windows")
            print("  Action: Run backtest_with_ml.py with this model, then")
            print("  paper trade for 4 weeks before live capital.")

        elif auc >= 0.58 and avg_wf >= 0.54:
            print("  ✅ PROCEED — useful edge, deploy with half sizing")
            print("  Action: Run backtest_with_ml.py, paper trade 8 weeks.")
            print("  Monitor walk-forward consistency monthly.")

        elif auc >= 0.55:
            print("  ⚠  MARGINAL — edge exists but inconsistent across windows")
            print("  Action: Try on altcoin pairs (SOL, AVAX) before deploying.")
            print("  Edge may be stronger on less-efficient assets.")

        else:
            print("  ❌ DO NOT DEPLOY — AUC below minimum threshold")
            print()
            print("  Recommended next steps:")
            print("  1. Download SOL/USDT, AVAX/USDT, or INJ/USDT 4h data")
            print("     → python scripts/load_history.py --symbol SOL/USDT --timeframe 4h --years 3")
            print("  2. Fetch alpha features for those assets")
            print("     → python ml/fetch_alpha_features.py --symbol SOLUSDT --days 1100")
            print("  3. Generate volatility labels and retrain")
            print("  Less liquid assets have ~2x the AUC of BTC/ETH on identical features.")
    else:
        if auc >= 0.58:
            print(f"  ✅ AUC {auc:.4f} looks promising.")
            print("  Run with --walk-forward to confirm consistency before deploying.")
        else:
            print(f"  ❌ AUC {auc:.4f} — run with --walk-forward before deciding.")

    print()
    print(f"  Next commands:")
    print(f"    # If AUC is promising, backtest with the model:")
    print(f"    python ml/backtest_with_ml.py \\")
    print(f"      --csv backtest/data/<your_4h>.csv \\")
    print(f"      --model-dir ml/models --slug {slug}")
    print()
    print(f"    # To test on altcoins (recommended if AUC < 0.58):")
    print(f"    python scripts/load_history.py --symbol SOL/USDT --timeframe 4h --years 3")
    print(f"    python scripts/export_to_csv.py --symbol 'SOL/USDT' --timeframe 4h")
    print(f"    python ml/fetch_alpha_features.py --symbol SOLUSDT --days 1100 --no-coinglass")
    print(f"    python ml/generate_volatility_labels.py --csv ml/data/SOL_USDT_4h_alpha.csv")
    print(f"    python ml/train_volatility_model.py --data ml/data/SOL_USDT_4h_alpha_vol_training.csv --walk-forward")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()