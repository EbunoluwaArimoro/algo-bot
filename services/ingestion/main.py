"""
Ingestion Service — algo-bot
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import asyncpg
import pandas as pd
# --- PURE PYTHON TA LIBRARY ---
from ta.trend import EMAIndicator, ADXIndicator, MACD, sma_indicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
# ------------------------------
import redis.asyncio as aioredis
import websockets
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingestion")

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS: list[str] = os.getenv("SYMBOLS", "btcusdt,ethusdt,solusdt").split(",")
TIMEFRAMES: list[str] = os.getenv("TIMEFRAMES", "1m,5m,1h").split(",")
EXCHANGE = "binance"

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER', 'bot')}:"
    f"{os.getenv('DB_PASSWORD', 'botpass')}@"
    f"{os.getenv('DB_HOST', 'timescaledb')}:5432/"
    f"{os.getenv('DB_NAME', 'botdb')}"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"

CANDLE_BUFFER = 200

FEATURE_TTL_S = 120   
REGIME_TTL_S  = 3600  

# ── Shared state ──────────────────────────────────────────────────────────────
candle_buffers: dict[str, pd.DataFrame] = {}

# ── DB helpers ────────────────────────────────────────────────────────────────

async def get_db_pool() -> asyncpg.Pool:
    for attempt in range(30):
        try:
            pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)
            log.info("TimescaleDB connected")
            return pool
        except Exception as exc:
            log.warning("DB not ready (attempt %d/30): %s", attempt + 1, exc)
            await asyncio.sleep(3)
    raise RuntimeError("Could not connect to TimescaleDB after 30 attempts")

async def insert_candle(pool: asyncpg.Pool, row: dict) -> None:
    sql = """
        INSERT INTO ohlcv (time, symbol, exchange, open, high, low, close, volume, timeframe)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.execute(
            sql,
            row["time"], row["symbol"], EXCHANGE,
            row["open"], row["high"], row["low"], row["close"],
            row["volume"], row["timeframe"],
        )

# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 50:
        return None

    # --- USING PURE PYTHON TA ---
    # Trend
    ema20 = EMAIndicator(close=df["close"], window=20).ema_indicator()
    ema50 = EMAIndicator(close=df["close"], window=50).ema_indicator()
    ema200 = EMAIndicator(close=df["close"], window=200).ema_indicator()

    # Momentum
    rsi = RSIIndicator(close=df["close"], window=14).rsi()
    macd_obj = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    macd = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()
    macd_hist = macd_obj.macd_diff()

    # Volatility
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    bb_obj = BollingerBands(close=df["close"], window=20, window_dev=2)
    bb_upper = bb_obj.bollinger_hband()
    bb_lower = bb_obj.bollinger_lband()
    bb_mid = bb_obj.bollinger_mavg()

    # Volume
    vol_sma20 = sma_indicator(close=df["volume"], window=20)
    
    # Handle NaN in volume ratio gracefully
    last_vol_sma = vol_sma20.iloc[-1]
    if pd.isna(last_vol_sma) or last_vol_sma == 0:
        volume_ratio = 1.0
    else:
        volume_ratio = float(df["volume"].iloc[-1] / last_vol_sma)

    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx()

    def last(series) -> float | None:
        try:
            v = series.iloc[-1]
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    current_close = float(df["close"].iloc[-1])
    
    # --- ADDED: Recreate the exact MTF features from backtest/engine.py ---
    ema_bull = (last(ema20) is not None and last(ema50) is not None and last(ema200) is not None and last(ema20) > last(ema50) > last(ema200))
    ema_bear = (last(ema20) is not None and last(ema50) is not None and last(ema200) is not None and last(ema20) < last(ema50) < last(ema200))
    
    ema_spread_pct = 0.0
    if last(ema20) and last(ema50):
        ema_spread_pct = (abs(last(ema20) - last(ema50)) / current_close) * 100
        
    bb_width = 0.0
    if last(bb_upper) and last(bb_lower) and last(bb_mid):
        bb_width = (last(bb_upper) - last(bb_lower)) / (last(bb_mid) + 1e-9)
        
    # We can use the 1h data as a proxy for the 4h MTF indicators for the ML model 
    # to ensure it doesn't receive missing values (0.0) which crashes the probability score
    ema_bull_4h = ema_bull
    ema_bear_4h = ema_bear
    adx_4h = last(adx)
    # ----------------------------------------------------------------------

    features = {
        "close":        current_close,
        "open":         float(df["open"].iloc[-1]),
        "high":         float(df["high"].iloc[-1]),
        "low":          float(df["low"].iloc[-1]),
        "volume":       float(df["volume"].iloc[-1]),
        "volume_ratio": volume_ratio,
        # Trend
        "ema20":        last(ema20),
        "ema50":        last(ema50),
        "ema200":       last(ema200),
        # Momentum
        "rsi":          last(rsi),
        "macd":         last(macd),
        "macd_signal":  last(macd_signal),
        "macd_hist":    last(macd_hist),
        # Volatility
        "atr":          last(atr),
        "bb_upper":     last(bb_upper),
        "bb_lower":     last(bb_lower),
        "bb_mid":       last(bb_mid),
        # Regime helper
        "adx":          last(adx),
        
        # --- ADDED: Missing ML Features ---
        "ema_bull":         float(ema_bull),
        "ema_bear":         float(ema_bear),
        "ema_spread_pct":   ema_spread_pct,
        "bb_width":         bb_width,
        "ema_bull_4h":      float(ema_bull_4h),
        "ema_bear_4h":      float(ema_bear_4h),
        "adx_4h":           adx_4h,
        "atr_pct":          (last(atr) / current_close) * 100 if last(atr) else 0.0,
        # -----------------------------------
        
        # Derived
        "ema_aligned_bull": ema_bull,
        "ema_aligned_bear": ema_bear,
        "timestamp": int(time.time()),
    }

    return features


def detect_regime(features: dict) -> str:
    adx   = features.get("adx") or 0
    rsi   = features.get("rsi") or 50
    atr   = features.get("atr") or 0
    close = features.get("close") or 1
    atr_pct = (atr / close) * 100 

    if atr_pct > 4.0:                                          
        return "high_volatility"
    elif features.get("ema_aligned_bull") and adx > 22:
        return "trending_bull"
    elif features.get("ema_aligned_bear") and adx > 22:
        return "trending_bear"
    else:
        return "ranging"

# ── Redis feature store writer ────────────────────────────────────────────────

async def write_features_to_redis(
    redis: aioredis.Redis,
    symbol: str,
    timeframe: str,
    features: dict,
) -> None:
    pipe = redis.pipeline()
    key  = f"features:{symbol}:{timeframe}:latest"
    pipe.set(key, json.dumps(features), ex=FEATURE_TTL_S)

    if timeframe == "1h":
        regime = detect_regime(features)
        pipe.set(f"regime:{symbol}", regime, ex=REGIME_TTL_S)
        log.info("Regime %-12s → %s", symbol, regime)

    await pipe.execute()


async def load_initial_buffer(
    pool: asyncpg.Pool,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    sql = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = $1 AND timeframe = $2
        ORDER BY time DESC
        LIMIT $3
    """
    
    # This ensures "solusdt" becomes "SOL/USDT"
    clean_symbol = symbol.upper()
    # Change from f"{symbol[:-4].upper()}/{symbol[-4:].upper()}" to:
    db_symbol = symbol.upper()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, db_symbol, timeframe, CANDLE_BUFFER)

    if not rows:
        # If this log appears, your database is either empty or the symbol/timeframe is wrong
        log.warning("No data found in DB for %s %s. Check your 'ohlcv' table.", db_symbol, timeframe)
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.sort_values("time").reset_index(drop=True)
    log.info("Pre-loaded %d candles for %s/%s from DB", len(df), db_symbol, timeframe)
    return df

# ── WebSocket stream ──────────────────────────────────────────────────────────

def build_stream_url(symbols: list[str], timeframes: list[str]) -> str:
    streams = [f"{s}@kline_{tf}" for s in symbols for tf in timeframes]
    return f"{BINANCE_WS_BASE}?streams={'/'.join(streams)}"

def parse_kline(msg: dict) -> dict | None:
    try:
        k = msg["data"]["k"]
        # if not k["x"]:          
        #     return None
        return {
            "symbol":    k["s"],               
            "timeframe": k["i"],               
            "time":      datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
            "open":      float(k["o"]),
            "high":      float(k["h"]),
            "low":       float(k["l"]),
            "close":     float(k["c"]),
            "volume":    float(k["v"]),
        }
    except (KeyError, ValueError) as exc:
        log.warning("Kline parse error: %s | raw: %s", exc, msg)
        return None

async def process_candle_update(
    candle: dict,
    is_closed: bool,
    pool: asyncpg.Pool,
    redis: aioredis.Redis,
) -> None:
    symbol_raw = candle["symbol"].lower()         
    timeframe  = candle["timeframe"]             
    buf_key    = f"{symbol_raw}_{timeframe}"

    if is_closed:
        await insert_candle(pool, candle)

    new_row = pd.DataFrame([{
        "time":   candle["time"],
        "open":   candle["open"],
        "high":   candle["high"],
        "low":    candle["low"],
        "close":  candle["close"],
        "volume": candle["volume"],
    }])

    if buf_key not in candle_buffers:
        candle_buffers[buf_key] = new_row
    else:
        candle_buffers[buf_key] = pd.concat(
            [candle_buffers[buf_key], new_row], ignore_index=True
        ).tail(CANDLE_BUFFER)

    features = compute_features(candle_buffers[buf_key])
    if features is None:
        log.debug("%s/%s — not enough candles yet (%d)",
                  symbol_raw, timeframe, len(candle_buffers[buf_key]))
        return

    await write_features_to_redis(redis, symbol_raw, timeframe, features)

    log.info(
        "%-9s %-4s close=%-10.4f rsi=%-6.1f ema_bull=%s vol_ratio=%.2f",
        symbol_raw, timeframe,
        features["close"], features.get("rsi") or 0,
        features.get("ema_aligned_bull"), features["volume_ratio"],
    )

async def stream_loop(
    pool: asyncpg.Pool,
    redis: aioredis.Redis,
    symbols: list[str],
    timeframes: list[str],
) -> None:
    url     = build_stream_url(symbols, timeframes)
    backoff = 1  

    log.info("Connecting to Binance WebSocket…")
    log.info("Symbols: %s | Timeframes: %s", symbols, timeframes)

    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                log.info("WebSocket connected ✓")
                backoff = 1  

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    candle = parse_kline(msg)
                    if candle:
                        is_closed = msg["data"]["k"]["x"]
                        await process_candle_update(candle, is_closed, pool, redis)

        except websockets.exceptions.ConnectionClosed as exc:
            log.warning("WebSocket closed (%s). Reconnecting in %ds…", exc, backoff)
        except OSError as exc:
            log.error("Network error: %s. Reconnecting in %ds…", exc, backoff)
        except Exception as exc:
            log.exception("Unexpected error: %s. Reconnecting in %ds…", exc, backoff)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)  

# ── Startup: pre-fill buffers from DB ────────────────────────────────────────

async def prefill_buffers(pool: asyncpg.Pool) -> None:
    tasks = [
        load_initial_buffer(pool, s, tf)
        for s in SYMBOLS
        for tf in TIMEFRAMES
    ]
    results = await asyncio.gather(*tasks)
    for i, (s, tf) in enumerate(
        [(s, tf) for s in SYMBOLS for tf in TIMEFRAMES]
    ):
        buf_key = f"{s}_{tf}"
        candle_buffers[buf_key] = results[i]

# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("═══ Ingestion Service starting ═══")

    pool  = await get_db_pool()
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await prefill_buffers(pool)

    stream_task = asyncio.create_task(
        stream_loop(pool, redis, SYMBOLS, TIMEFRAMES)
    )
    await stop.wait()

    log.info("Shutdown signal received — cleaning up…")
    stream_task.cancel()
    await pool.close()
    await redis.aclose()
    log.info("Ingestion Service stopped.")

if __name__ == "__main__":
    asyncio.run(main())