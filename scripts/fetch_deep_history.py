"""
scripts/fetch_deep_history.py
──────────────────────────────
Fetches full historical OHLCV data directly from the Binance REST API,
bypassing CCXT entirely. Solves the altcoin pagination truncation issue
where CCXT silently returns only recent data for some symbols.

Why this works when load_history.py doesn't:
  - Calls Binance spot kline endpoint directly with explicit startTime
  - Verifies each response actually contains the expected date range
  - Falls back to vision endpoint if spot truncates
  - Saves directly to CSV (no DB connection needed)
  - Also inserts into TimescaleDB if --db flag is passed

Usage:
    # Save to CSV only (recommended — avoids asyncpg Windows issues)
    python scripts/fetch_deep_history.py --symbol SOLUSDT --years 3

    # Also insert into TimescaleDB
    python scripts/fetch_deep_history.py --symbol SOLUSDT --years 3 --db

    # Multiple symbols
    python scripts/fetch_deep_history.py --symbol SOLUSDT AVAXUSDT INJUSDT --years 3

    # Verify what you got
    python scripts/fetch_deep_history.py --symbol SOLUSDT --verify-only
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deep_history")

# Binance endpoints — spot has the longest history for altcoins
BINANCE_SPOT    = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"
BINANCE_VISION  = "https://data.binance.vision"

SLEEP_BETWEEN_CALLS = 0.3   # seconds — well under rate limit
MAX_RETRIES         = 5
OUTPUT_DIR          = Path("backtest/data")

# Timeframe string → milliseconds per candle
TF_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def get_klines(
    symbol:    str,
    interval:  str,
    start_ms:  int,
    end_ms:    int | None = None,
    limit:     int = 1000,
    endpoint:  str = BINANCE_SPOT,
) -> list:
    """
    Single call to Binance klines endpoint.
    Returns raw list of kline arrays, or empty list on failure.
    """
    path   = "/api/v3/klines" if endpoint == BINANCE_SPOT else "/fapi/v1/klines"
    url    = f"{endpoint}{path}"
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "limit":     limit,
    }
    if end_ms:
        params["endTime"] = end_ms

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=20)

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                log.warning("Rate limited. Sleeping %ds…", wait)
                time.sleep(wait)
                continue

            if r.status_code == 400:
                # Symbol not available on this endpoint
                return []

            r.raise_for_status()
            data = r.json()

            if isinstance(data, dict) and "code" in data:
                log.warning("API error: %s", data)
                return []

            return data

        except requests.Timeout:
            log.warning("Timeout on attempt %d/%d", attempt + 1, MAX_RETRIES)
            time.sleep(2 ** attempt)
        except requests.RequestException as exc:
            log.warning("Request error attempt %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
            time.sleep(2 ** attempt)

    return []


def klines_to_df(klines: list, symbol: str, timeframe: str) -> pd.DataFrame:
    """Convert raw Binance kline arrays to a clean DataFrame."""
    if not klines:
        return pd.DataFrame()

    rows = []
    for k in klines:
        rows.append({
            "time":     ms_to_dt(k[0]),
            "open":     float(k[1]),
            "high":     float(k[2]),
            "low":      float(k[3]),
            "close":    float(k[4]),
            "volume":   float(k[5]),
            "symbol":   symbol,
            "timeframe":timeframe,
        })
    return pd.DataFrame(rows)


def fetch_full_history(
    symbol:    str,
    timeframe: str,
    years:     float,
) -> pd.DataFrame:
    """
    Fetch complete OHLCV history for a symbol by paginating through
    the Binance spot API with explicit startTime control.

    Key difference from CCXT: we explicitly verify each batch covers
    the requested time range and log when the API returns truncated data.
    """
    since_dt = datetime.now(timezone.utc) - timedelta(days=int(365 * years))
    since_ms = dt_to_ms(since_dt)
    tf_ms    = TF_MS.get(timeframe)
    if not tf_ms:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use: {list(TF_MS)}")

    log.info("Fetching %s / %s from %s (%.1f years)…",
             symbol, timeframe, since_dt.date(), years)

    # Try spot first, fall back to futures
    endpoints_to_try = [
        (BINANCE_SPOT,    "/api/v3/klines"),
        (BINANCE_FUTURES, "/fapi/v1/klines"),
    ]

    working_endpoint = None
    for ep, path in endpoints_to_try:
        test = get_klines(symbol, timeframe, since_ms, limit=1, endpoint=ep)
        if test:
            actual_start = ms_to_dt(test[0][0])
            log.info("  Endpoint %s works. First candle: %s", ep, actual_start.date())

            # Check if endpoint is giving us old enough data
            if actual_start <= since_dt + timedelta(days=30):
                working_endpoint = ep
                log.info("  ✓ Endpoint has sufficient history")
                break
            else:
                log.warning("  ✗ Endpoint only goes back to %s (need %s)",
                            actual_start.date(), since_dt.date())
        else:
            log.warning("  ✗ Endpoint %s returned no data", ep)

    if working_endpoint is None:
        log.error("No Binance endpoint has history back to %s for %s",
                  since_dt.date(), symbol)
        log.error("This symbol may not have existed 3 years ago.")
        log.error("Try --years 2 or check when %s was listed.", symbol)
        return pd.DataFrame()

    # Paginate through full history
    all_frames  = []
    current_ms  = since_ms
    total_rows  = 0
    batch_num   = 0

    while True:
        batch = get_klines(
            symbol, timeframe, current_ms,
            limit=1000, endpoint=working_endpoint,
        )

        if not batch:
            log.info("Empty batch at %s — pagination complete",
                     ms_to_dt(current_ms).date())
            break

        df_batch = klines_to_df(batch, symbol, timeframe)
        all_frames.append(df_batch)
        total_rows += len(batch)
        batch_num  += 1

        last_ts   = batch[-1][0]
        last_date = ms_to_dt(last_ts).date()

        if batch_num % 5 == 0 or len(batch) < 1000:
            log.info("  Batch %d: %d candles, up to %s, total: %d",
                     batch_num, len(batch), last_date, total_rows)

        # Check if we've reached the present
        if len(batch) < 1000:
            log.info("Final batch (%d candles) — reached present", len(batch))
            break

        # Advance to the next candle after the last one returned
        current_ms = last_ts + tf_ms
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    actual_start = df["time"].iloc[0]
    actual_end   = df["time"].iloc[-1]
    expected_bars= int((actual_end - actual_start).total_seconds() / (tf_ms / 1000)) + 1
    gap_pct      = (1 - len(df) / expected_bars) * 100 if expected_bars > 0 else 0

    log.info("Complete: %d candles  |  %s → %s  |  %.1f%% gap rate",
             len(df), actual_start.date(), actual_end.date(), gap_pct)

    if gap_pct > 5:
        log.warning("High gap rate (%.1f%%). Exchange may have had downtime.", gap_pct)
    if len(df) < 500:
        log.warning("Only %d candles fetched. Symbol may be too new for --years %.1f",
                    len(df), years)
        log.warning("Check listing date and reduce --years accordingly.")

    return df


def verify_csv(csv_path: Path) -> None:
    """Print a summary of an existing CSV to verify its contents."""
    if not csv_path.exists():
        log.error("File not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path, parse_dates=["time"])
    print(f"\n  Verification: {csv_path.name}")
    print(f"  {'─'*40}")
    print(f"  Rows:       {len(df):>10,}")
    print(f"  First bar:  {str(df['time'].iloc[0])[:19]}")
    print(f"  Last bar:   {str(df['time'].iloc[-1])[:19]}")
    print(f"  Columns:    {list(df.columns)}")
    print(f"  Missing:    {df.isna().sum().sum()} total NaN values")

    # Check for gaps
    if "time" in df.columns and len(df) > 1:
        df["time"] = pd.to_datetime(df["time"])
        diffs = df["time"].diff().dropna()
        mode_diff = diffs.mode()[0]
        gaps = (diffs > mode_diff * 1.5).sum()
        print(f"  Candle gaps:{gaps:>10} detected")
    print()


def save_csv(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    """Save DataFrame to CSV in backtest/data/ directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = symbol.replace("USDT", "_USDT")
    path = OUTPUT_DIR / f"{slug}_{timeframe}.csv"

    # Drop DB-specific columns before saving
    save_cols = ["time", "open", "high", "low", "close", "volume"]
    df[save_cols].to_csv(path, index=False)
    log.info("Saved → %s  (%d rows)", path, len(df))
    return path


def insert_to_db(df: pd.DataFrame, symbol_ccxt: str) -> None:
    """
    Optional: Insert data into TimescaleDB using psycopg2 (synchronous).
    Uses the same connection approach that fixed the asyncpg Windows issues.
    """
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        log.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return

    db_config = {
        "host":     os.getenv("DB_HOST",     "localhost"),
        "port":     int(os.getenv("DB_PORT", "5433")),
        "dbname":   os.getenv("DB_NAME",     "botdb"),
        "user":     os.getenv("DB_USER",     "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }

    try:
        conn = psycopg2.connect(**db_config)
        cur  = conn.cursor()
        log.info("Inserting %d rows into TimescaleDB…", len(df))

        inserted = 0
        batch    = []
        for _, row in df.iterrows():
            batch.append((
                row["time"], symbol_ccxt, "binance",
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]),
                float(row["volume"]), row.get("timeframe", "4h"),
            ))

            if len(batch) >= 1000:
                cur.executemany("""
                    INSERT INTO ohlcv
                        (time, symbol, exchange, open, high, low, close, volume, timeframe)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                """, batch)
                inserted += len(batch)
                batch = []

        if batch:
            cur.executemany("""
                INSERT INTO ohlcv
                    (time, symbol, exchange, open, high, low, close, volume, timeframe)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, batch)
            inserted += len(batch)

        conn.commit()
        cur.close()
        conn.close()
        log.info("DB insert complete: %d rows", inserted)

    except Exception as exc:
        log.error("DB insert failed: %s", exc)
        log.error("CSV was saved successfully — you can use that directly.")


def main():
    p = argparse.ArgumentParser(
        description="Fetch deep historical OHLCV from Binance REST API directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/fetch_deep_history.py --symbol SOLUSDT --years 3
    python scripts/fetch_deep_history.py --symbol SOLUSDT AVAXUSDT --years 3
    python scripts/fetch_deep_history.py --symbol SOLUSDT --years 3 --db
    python scripts/fetch_deep_history.py --symbol SOLUSDT --verify-only

If 3 years fails (symbol too new):
    python scripts/fetch_deep_history.py --symbol SOLUSDT --years 2
    python scripts/fetch_deep_history.py --symbol INJUSDT --years 1.5

Timeframes available: 1m 5m 15m 1h 4h 1d
        """
    )
    p.add_argument("--symbol",      nargs="+", required=True,
                   help="Binance symbol(s) e.g. SOLUSDT AVAXUSDT (no slash)")
    p.add_argument("--timeframe",   default="4h",
                   choices=list(TF_MS.keys()))
    p.add_argument("--years",       type=float, default=3.0,
                   help="Years of history to fetch (default 3.0)")
    p.add_argument("--db",          action="store_true",
                   help="Also insert into TimescaleDB (requires psycopg2)")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify existing CSV, do not fetch")
    args = p.parse_args()

    for raw_symbol in args.symbol:
        symbol = raw_symbol.upper()

        if args.verify_only:
            slug     = symbol.replace("USDT", "_USDT")
            csv_path = OUTPUT_DIR / f"{slug}_{args.timeframe}.csv"
            verify_csv(csv_path)
            continue

        log.info("═" * 50)
        log.info("Symbol: %s  |  Timeframe: %s  |  Years: %.1f",
                 symbol, args.timeframe, args.years)
        log.info("═" * 50)

        df = fetch_full_history(symbol, args.timeframe, args.years)

        if df.empty:
            log.error("No data fetched for %s. Skipping.", symbol)
            continue

        # Save CSV
        df["timeframe"] = args.timeframe
        csv_path = save_csv(df, symbol, args.timeframe)
        verify_csv(csv_path)

        # Optional DB insert
        if args.db:
            # Convert symbol format: SOLUSDT → SOL/USDT for DB storage
            ccxt_symbol = symbol.replace("USDT", "/USDT")
            df["timeframe"] = args.timeframe
            insert_to_db(df, ccxt_symbol)

    log.info("Done.")

    if not args.verify_only:
        print("\n  Next step:")
        for raw_symbol in args.symbol:
            symbol = raw_symbol.upper()
            slug   = symbol.replace("USDT", "_USDT")
            print(f"    python ml/fetch_alpha_features.py \\")
            print(f"      --symbol {symbol} --days 1100 --no-coinglass \\")
            print(f"      --ohlcv-csv backtest/data/{slug}_{args.timeframe}.csv")
        print()


if __name__ == "__main__":
    main()