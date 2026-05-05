"""
scripts/load_history.py
────────────────────────
Downloads historical OHLCV candles from Binance (no API key needed for
public market data) and bulk-inserts into TimescaleDB.

Usage:
    python scripts/load_history.py --symbol BTC/USDT --timeframe 1h --years 5
    python scripts/load_history.py --symbol ETH/USDT --timeframe 1m --years 1
    python scripts/load_history.py --all  # loads all symbols in .env

Binance rate limits for unauthenticated requests:
    - 1,200 request-weight per minute
    - fetch_ohlcv returns up to 1,000 candles per call
    - We sleep 0.25s between calls → ~240 calls/min → safe
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import asyncpg
import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("load_history")

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER', 'bot')}:"
    f"{os.getenv('DB_PASSWORD', 'botpass')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:5433/"   # localhost when run outside Docker
    f"{os.getenv('DB_NAME', 'botdb')}"
)

BATCH_INSERT = 1000     # rows per DB insert
SLEEP_S      = 0.25     # pause between API calls (rate-limit safe)


# ── DB helpers ────────────────────────────────────────────────────────────────

async def get_pool() -> asyncpg.Pool:
    for attempt in range(10):
        try:
            pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=5)
            log.info("Connected to TimescaleDB")
            return pool
        except Exception as exc:
            log.warning("DB not ready (%d/10): %s", attempt + 1, exc)
            await asyncio.sleep(3)
    raise RuntimeError("Cannot connect to DB")


async def bulk_insert(pool: asyncpg.Pool, rows: list[tuple]) -> int:
    """
    Bulk upsert OHLCV rows.
    Returns number of rows inserted (not counting duplicates).
    """
    sql = """
        INSERT INTO ohlcv (time, symbol, exchange, open, high, low, close, volume, timeframe)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT DO NOTHING
    """
    async with pool.acquire() as conn:
        result = await conn.executemany(sql, rows)
    # executemany returns e.g. "INSERT 0 850" — parse the count
    try:
        return int(str(result).split()[-1])
    except Exception:
        return len(rows)


async def get_earliest_stored(
    pool: asyncpg.Pool, symbol: str, timeframe: str
) -> datetime | None:
    """Return the oldest candle timestamp we have for this pair."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT MIN(time) FROM ohlcv WHERE symbol=$1 AND timeframe=$2",
            symbol, timeframe,
        )
    return row[0] if row and row[0] else None


# ── Downloader ────────────────────────────────────────────────────────────────

async def download_symbol(
    exchange: ccxt.Exchange,
    pool: asyncpg.Pool,
    symbol: str,       # e.g. 'BTC/USDT'
    timeframe: str,    # e.g. '1h'
    years: float,
) -> None:
    db_symbol = symbol          # stored as 'BTC/USDT'
    log.info("Starting download: %s  %s  %.1f years", symbol, timeframe, years)

    # Start from `years` ago OR from just before the earliest we have stored
    earliest = await get_earliest_stored(pool, db_symbol, timeframe)
    since_dt = datetime.now(timezone.utc) - timedelta(days=365 * years)

    if earliest and earliest <= since_dt:
        log.info("%s/%s already has data back to %s — skipping",
                 symbol, timeframe, earliest.date())
        return

    since_ms = int(since_dt.timestamp() * 1000)

    total_inserted = 0
    total_fetched  = 0
    batch: list[tuple] = []

    while True:
        try:
            candles = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=1000
            )
        except ccxt.NetworkError as exc:
            log.warning("Network error, retrying in 5s: %s", exc)
            await asyncio.sleep(5)
            continue
        except ccxt.RateLimitExceeded:
            log.warning("Rate limited — sleeping 10s")
            await asyncio.sleep(10)
            continue

        if not candles:
            break

        for c in candles:
            ts = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
            batch.append((ts, db_symbol, "binance",
                          float(c[1]), float(c[2]), float(c[3]),
                          float(c[4]), float(c[5]), timeframe))

        total_fetched += len(candles)

        # Insert in batches
        if len(batch) >= BATCH_INSERT:
            inserted = await bulk_insert(pool, batch)
            total_inserted += inserted
            batch = []
            log.info("  %s/%s  fetched=%-7d  inserted=%-7d  up_to=%s",
                     symbol, timeframe, total_fetched, total_inserted,
                     datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc).date())

        # Advance window past last returned candle
        last_ts = candles[-1][0]
        since_ms = last_ts + 1  # +1ms to avoid re-fetching last candle

        # If the exchange returned fewer than 1000 candles we've hit present
        if len(candles) < 1000:
            break

        await asyncio.sleep(SLEEP_S)

    # Flush remaining
    if batch:
        inserted = await bulk_insert(pool, batch)
        total_inserted += inserted

    log.info("✓ %s/%s done — fetched %d candles, inserted %d",
             symbol, timeframe, total_fetched, total_inserted)


# ── Stats reporter ────────────────────────────────────────────────────────────

async def print_stats(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT symbol, timeframe,
                   COUNT(*)          AS rows,
                   MIN(time)::DATE   AS earliest,
                   MAX(time)::DATE   AS latest
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
    if not rows:
        log.info("ohlcv table is empty")
        return
    log.info("\n%s", "─" * 62)
    log.info("  %-12s %-6s %8s   %11s  %11s", "symbol","tf","rows","earliest","latest")
    log.info("─" * 62)
    for r in rows:
        log.info("  %-12s %-6s %8d   %11s  %11s",
                 r["symbol"], r["timeframe"], r["rows"],
                 str(r["earliest"]), str(r["latest"]))
    log.info("─" * 62)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Download historical OHLCV from Binance")
    p.add_argument("--symbol",    default=None,  help="e.g. BTC/USDT")
    p.add_argument("--timeframe", default="1h",  help="e.g. 1m 5m 1h 4h 1d")
    p.add_argument("--years",     type=float, default=3.0)
    p.add_argument("--all",       action="store_true",
                   help="Download all symbols from SYMBOLS env var")
    p.add_argument("--stats",     action="store_true",
                   help="Print table stats and exit")
    return p.parse_args()


async def run() -> None:
    args = parse_args()

    pool = await get_pool()

    if args.stats:
        await print_stats(pool)
        await pool.close()
        return

    # Determine which symbols to download
    if args.all:
        raw = os.getenv("SYMBOLS", "btcusdt,ethusdt,solusdt").split(",")
        # Convert 'btcusdt' → 'BTC/USDT' for CCXT
        symbols = [f"{s[:-4].upper()}/{s[-4:].upper()}" for s in raw]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        log.error("Provide --symbol BTC/USDT or --all")
        sys.exit(1)

    timeframes = (
        args.timeframe.split(",") if "," in args.timeframe else [args.timeframe]
    )

    exchange = ccxt.kraken({
        "enableRateLimit": True,   # built-in rate limiter
        "options": {"defaultType": "spot"},
    })

    try:
        for symbol in symbols:
            for tf in timeframes:
                await download_symbol(exchange, pool, symbol, tf, args.years)
                await asyncio.sleep(0.5)
    finally:
        await exchange.close()
        await pool.close()

    log.info("\n═══ Download complete ═══")
    await print_stats(await get_pool())


if __name__ == "__main__":
    asyncio.run(run())