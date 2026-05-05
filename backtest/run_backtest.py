"""
backtest/run_backtest.py  —  v2
────────────────────────────────
Reads from local CSV (no DB connection during strategy loops).

Usage:
    python backtest/run_backtest.py --csv data/BTCUSDT_1h.csv
    python backtest/run_backtest.py --csv data/BTCUSDT_1h.csv --compare-all
    python backtest/run_backtest.py --csv data/BTCUSDT_1h.csv --full
    python backtest/run_backtest.py --csv data/BTCUSDT_1h.csv --strategy mean_reversion --no-trailing
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import (
    BacktestEngine, BacktestResult, build_features,
    load_ohlcv_from_csv, monte_carlo, walk_forward, STRATEGY_SIGNALS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest.runner")

PASS = {
    "min_trades": 30, "min_sharpe": 1.0,
    "max_drawdown": 25.0, "min_pf": 1.3, "min_wr": 45.0,
}


# ── Printers ───────────────────────────────────────────────────────────────────

def verdict(r: BacktestResult) -> str:
    ok = (r.total_trades >= PASS["min_trades"] and
          r.sharpe >= PASS["min_sharpe"] and
          r.max_drawdown_pct <= PASS["max_drawdown"] and
          r.profit_factor >= PASS["min_pf"] and
          r.win_rate_pct >= PASS["min_wr"])
    return "✅ PASS" if ok else "❌ FAIL"


def print_result(r: BacktestResult, title: str = "") -> None:
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  {title or r.strategy.upper()}  —  {r.symbol}  {r.timeframe}")
    print(f"  {r.start_date.date()} → {r.end_date.date()}   {verdict(r)}")
    print(sep)
    print(f"  {'Capital':30}  ${r.initial_capital:>10,.2f} → ${r.final_capital:>10,.2f}")
    print(f"  {'Total return':30}  {r.total_return_pct:>+10.2f}%")
    print(f"  {'CAGR':30}  {r.cagr_pct:>+10.2f}%")
    print()

    def row(label, val, thr, op=">=", unit=""):
        ok = val >= thr if op == ">=" else val <= thr
        return f"  {'✓' if ok else '✗'}  {label:28}  {val:>8.3f}{unit}   ({op}{thr}{unit})"

    print(row("Sharpe ratio",       r.sharpe,           PASS["min_sharpe"]))
    print(row("Sortino ratio",      r.sortino,          1.0))
    print(row("Calmar ratio",       r.calmar,           1.5))
    print(row("Max drawdown",       r.max_drawdown_pct, PASS["max_drawdown"], "<=", "%"))
    print(row("Win rate",           r.win_rate_pct,     PASS["min_wr"],       ">=", "%"))
    print(row("Profit factor",      r.profit_factor,    PASS["min_pf"]))
    print()
    print(f"  {'Total trades':30}  {r.total_trades:>10}")
    print(f"  {'Avg trade':30}  {r.avg_trade_pct:>+9.3f}%")
    print(f"  {'Avg win':30}  {r.avg_win_pct:>+9.3f}%")
    print(f"  {'Avg loss':30}  {r.avg_loss_pct:>+9.3f}%")
    print(f"  {'Best trade':30}  {r.best_trade_pct:>+9.3f}%")
    print(f"  {'Worst trade':30}  {r.worst_trade_pct:>+9.3f}%")
    print(f"  {'Avg hold':30}  {r.avg_hold_hours:>9.1f} hrs")

    # Regime breakdown
    if r.trades:
        from collections import Counter
        rc = Counter(t.regime for t in r.trades)
        print(f"\n  Trades by regime: " +
              "  ".join(f"{k}={v}" for k, v in sorted(rc.items())))

    print(sep)


def print_comparison(results: list[BacktestResult]) -> None:
    sep = "═" * 82
    print(f"\n{sep}")
    print(f"  STRATEGY COMPARISON  —  {results[0].symbol}  {results[0].timeframe}")
    print(sep)
    hdr = (f"  {'':2} {'Strategy':20}  {'Return':>8}  {'CAGR':>7}  {'Sharpe':>7}  "
           f"{'MaxDD':>7}  {'WinRate':>8}  {'PF':>6}  {'Trades':>7}")
    print(hdr)
    print("  " + "─" * 78)
    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        vd = "✅" if r.sharpe >= 1.0 and r.max_drawdown_pct <= 25 else "❌"
        print(f"  {vd} {r.strategy:20}  "
              f"{r.total_return_pct:>+7.1f}%  {r.cagr_pct:>+6.1f}%  "
              f"{r.sharpe:>7.3f}  {r.max_drawdown_pct:>6.1f}%  "
              f"{r.win_rate_pct:>7.1f}%  {r.profit_factor:>6.2f}  {r.total_trades:>7}")
    print(sep)


def print_monte_carlo(mc: dict, symbol: str) -> None:
    print(f"\n  Monte Carlo — {mc['n_simulations']:,} simulations — {symbol}")
    print("  " + "─" * 52)
    print(f"  Profitable scenarios:  {mc['pct_profitable']}%")
    print(f"  Worst case  (p5):      ${mc['final_value'][5]:>12,.2f}")
    print(f"  Median      (p50):     ${mc['final_value'][50]:>12,.2f}")
    print(f"  Best case   (p95):     ${mc['final_value'][95]:>12,.2f}")
    print(f"  Max DD p50:            {mc['max_drawdown'][50]:>9.1f}%")
    print(f"  Max DD p95:            {mc['max_drawdown'][95]:>9.1f}%")


def save_trades(r: BacktestResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_time","exit_time","symbol","direction","strategy",
                    "regime","score","entry_price","exit_price","qty",
                    "stop_loss","trail_stop","pnl","pnl_pct","exit_reason"])
        for t in r.trades:
            w.writerow([t.entry_time, t.exit_time, t.symbol, t.direction,
                        t.strategy, t.regime, t.score, t.entry_price, t.exit_price,
                        t.qty, t.stop_loss, t.trail_stop, t.pnl, t.pnl_pct, t.exit_reason])
    log.info("Trades → %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    csv_path = args.csv
    if not os.path.exists(csv_path):
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Loading CSV: %s", csv_path)
    raw_df = load_ohlcv_from_csv(csv_path)

    log.info("Computing features (includes 4h MTF resampling)…")
    df = build_features(raw_df)
    valid = df.dropna(subset=["ema200", "rsi", "atr", "adx"])
    log.info("Ready — %d valid candles out of %d", len(valid), len(df))

    symbol    = args.symbol
    timeframe = args.timeframe
    capital   = args.capital

    engine_kwargs = dict(
        symbol=symbol, timeframe=timeframe, strategy=args.strategy,
        initial_capital=capital, risk_per_trade=args.risk / 100,
        atr_mult=args.atr_mult, rr=args.rr,
        # Tier 1 — breakeven stop
        be_trigger_r=args.be_trigger,
        be_buffer_r=args.be_buffer,
        # Tier 2 — partial profit lock
        use_partial=not args.no_partial,
        partial_r=args.partial_r,
        partial_pct=args.partial_pct,
        # Tier 3 — trailing stop on remainder
        use_trailing=not args.no_trailing,
        trail_trigger_r=args.trail_trigger,
        trail_distance_r=args.trail_distance,
        cooldown_bars=args.cooldown, loss_cooldown=args.loss_cooldown,
    )

    # Compare all strategies
    if args.compare_all or args.full:
        results = []
        for strat in STRATEGY_SIGNALS:
            kw  = {**engine_kwargs, "strategy": strat}
            res = BacktestEngine(**kw).run(df.copy())
            results.append(res)
        print_comparison(results)
        if not args.full:
            return

    # Single strategy detail
    result = BacktestEngine(**engine_kwargs).run(df.copy())
    print_result(result, args.strategy.replace("_", " ").title())

    # Walk-forward
    if args.walk_forward or args.full:
        log.info("Walk-forward validation (3 windows)…")
        wf = walk_forward(df.copy(), engine_kwargs, n_windows=3)
        for r in wf:
            print_result(r, f"OOS window {r.strategy[-1]}")
        if wf:
            avg_oos = sum(r.sharpe for r in wf) / len(wf)
            ratio   = avg_oos / max(result.sharpe, 0.001)
            mark    = "✅" if ratio >= 0.70 else "❌"
            print(f"\n  {mark}  OOS/full Sharpe ratio: {ratio:.2f}  "
                  f"({avg_oos:.3f} / {result.sharpe:.3f})  (need ≥ 0.70)")

    # Monte Carlo
    if args.monte_carlo or args.full:
        log.info("Monte Carlo (5,000 sims)…")
        mc = monte_carlo(result.trades, capital)
        print_monte_carlo(mc, symbol)

    # Save outputs
    if args.save_trades or args.full:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = f"{symbol.replace('/', '')}_{args.strategy}_{ts}"
        save_trades(result, f"backtest/output/{slug}_trades.csv")


def parse_args():
    p = argparse.ArgumentParser(
        description="Algo-bot backtester v3 — three-tier exit system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit system parameter guide:
  Tier 1 — Breakeven stop (eliminates winners turning into losers):
    --be-trigger 0.75   Move stop to entry after 0.75R profit (default)
    --be-buffer  0.05   Stop sits 0.05R above entry to cover fees

  Tier 2 — Partial profit lock (guarantees a win on part of the trade):
    --partial-r   1.0   Take partial at 1R profit (default)
    --partial-pct 0.4   Close 40% of position at partial target (default)
    --no-partial        Disable partial exits entirely

  Tier 3 — Trailing stop on remainder (lets winners run):
    --trail-trigger  1.5  Trail activates at 1.5R profit (default)
    --trail-distance 1.0  Trail sits 1R behind the high/low (default)
    --no-trailing         Use static take-profit target instead

Recommended starting configs by strategy:
  Mean Reversion:  --atr-mult 1.5 --rr 1.8 --be-trigger 0.5 --partial-r 0.8 --no-trailing
  Trend Follow:    --atr-mult 1.5 --rr 3.0 --be-trigger 1.0 --partial-r 1.5 --trail-trigger 2.0
  Breakout:        --atr-mult 1.8 --rr 2.5 --be-trigger 0.75 --partial-r 1.2 --trail-trigger 1.5
        """
    )
    p.add_argument("--csv",       required=True,  help="Path to OHLCV CSV file")
    p.add_argument("--symbol",    default="BTC/USDT")
    p.add_argument("--timeframe", default="4h",
                   choices=["1m","5m","15m","1h","4h","1d"])
    p.add_argument("--strategy",  default="trend_follow",
                   choices=list(STRATEGY_SIGNALS.keys()))
    p.add_argument("--capital",   type=float, default=10_000.0)
    p.add_argument("--risk",      type=float, default=1.0,
                   help="Risk %% of portfolio per trade (default 1.0)")
    p.add_argument("--atr-mult",  type=float, default=1.5,
                   help="ATR multiplier for stop distance (default 1.5)")
    p.add_argument("--rr",        type=float, default=2.0,
                   help="Final target R:R ratio (default 2.0)")

    # Tier 1 — breakeven
    p.add_argument("--be-trigger", type=float, default=0.75,
                   help="R profit to trigger breakeven stop move (default 0.75)")
    p.add_argument("--be-buffer",  type=float, default=0.05,
                   help="R buffer above entry for breakeven stop (default 0.05)")

    # Tier 2 — partial
    p.add_argument("--no-partial",  action="store_true",
                   help="Disable partial profit exits")
    p.add_argument("--partial-r",   type=float, default=1.0,
                   help="R profit to trigger partial exit (default 1.0)")
    p.add_argument("--partial-pct", type=float, default=0.40,
                   help="Fraction of position to close at partial target (default 0.40)")

    # Tier 3 — trailing
    p.add_argument("--no-trailing",    action="store_true",
                   help="Disable trailing stop (use static RR target)")
    p.add_argument("--trail-trigger",  type=float, default=1.5,
                   help="R profit to activate trailing stop (default 1.5)")
    p.add_argument("--trail-distance", type=float, default=1.0,
                   help="R distance for trailing stop behind high/low (default 1.0)")

    p.add_argument("--cooldown",      type=int, default=12,
                   help="Bars to wait between signals (default 12)")
    p.add_argument("--loss-cooldown", type=int, default=24,
                   help="Extra bars to wait after a losing trade (default 24)")
    p.add_argument("--compare-all",   action="store_true",
                   help="Compare all strategies side by side")
    p.add_argument("--walk-forward",  action="store_true")
    p.add_argument("--monte-carlo",   action="store_true")
    p.add_argument("--full",          action="store_true",
                   help="Run all validations and save outputs")
    p.add_argument("--save-trades",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())