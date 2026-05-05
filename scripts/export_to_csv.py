"""
scripts/export_to_csv.py
─────────────────────────
Exports OHLCV data from TimescaleDB to a local CSV file.
Uses synchronous psycopg2 to avoid the asyncpg Windows asyncio issue.

Usage:
    python scripts/export_to_csv.py --symbol "BTC/USDT" --timeframe 4h
    python scripts/export_to_csv.py --symbol "ETH/USDT" --timeframe 4h
    python scripts/export_to_csv.py --symbol "SOL/USDT" --timeframe 1d
    python scripts/export_to_csv.py --all --timeframe 4h
"""

import argparse
import logging
import os
import sys

import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_csv")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5433")),
    "dbname":   os.getenv("DB_NAME",     "botdb"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

OUTPUT_DIR = "backtest/data"


def export_symbol(symbol: str, timeframe: str) -> str:
    """Export one symbol/timeframe to CSV. Returns the output path."""
    log.info("Exporting %s / %s …", symbol, timeframe)

    conn = psycopg2.connect(**DB_CONFIG)
    sql  = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = %s AND timeframe = %s
        ORDER BY time ASC
    """
    df = pd.read_sql(sql, conn, params=(symbol, timeframe))
    conn.close()

    if df.empty:
        log.warning("No data found for %s / %s — did you run load_history.py?",
                    symbol, timeframe)
        return ""

    # Ensure UTC timezone column
    df["time"] = pd.to_datetime(df["time"], utc=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = symbol.replace("/", "_").replace(" ", "")
    path = os.path.join(OUTPUT_DIR, f"{slug}_{timeframe}.csv")
    df.to_csv(path, index=False)

    log.info("  ✓ %d rows → %s  (%s → %s)",
             len(df),
             path,
             str(df["time"].iloc[0].date()),
             str(df["time"].iloc[-1].date()))
    return path


def list_available(conn) -> list[tuple]:
    """List all symbol/timeframe combinations in the DB."""
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, timeframe, COUNT(*) as rows,
               MIN(time)::DATE as earliest, MAX(time)::DATE as latest
        FROM ohlcv
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """)
    return cur.fetchall()


def main():
    p = argparse.ArgumentParser(description="Export OHLCV from TimescaleDB to CSV")
    p.add_argument("--symbol",    default=None,  help="e.g. BTC/USDT")
    p.add_argument("--timeframe", default="4h",  help="e.g. 1h 4h 1d")
    p.add_argument("--all",       action="store_true",
                   help="Export all symbols available in DB for this timeframe")
    p.add_argument("--list",      action="store_true",
                   help="List all available symbol/timeframe pairs and exit")
    args = p.parse_args()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as exc:
        log.error("Cannot connect to DB: %s", exc)
        log.error("Config: %s", {k: v for k, v in DB_CONFIG.items() if k != "password"})
        sys.exit(1)

    if args.list:
        rows = list_available(conn)
        conn.close()
        if not rows:
            print("  DB is empty — run load_history.py first")
            return
        print(f"\n  {'Symbol':14}  {'TF':6}  {'Rows':>8}  {'Earliest':>11}  {'Latest':>11}")
        print("  " + "─" * 58)
        for sym, tf, n, earliest, latest in rows:
            print(f"  {sym:14}  {tf:6}  {n:>8,}  {str(earliest):>11}  {str(latest):>11}")
        print()
        return

    if args.all:
        rows = list_available(conn)
        conn.close()
        symbols = [r[0] for r in rows if r[1] == args.timeframe]
        if not symbols:
            log.error("No data found for timeframe '%s'. Available: %s",
                      args.timeframe, list(set(r[1] for r in rows)))
            sys.exit(1)
        for sym in symbols:
            export_symbol(sym, args.timeframe)
    elif args.symbol:
        conn.close()
        export_symbol(args.symbol, args.timeframe)
    else:
        conn.close()
        log.error("Provide --symbol 'BTC/USDT' or --all")
        sys.exit(1)

    log.info("Done. Files in %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()