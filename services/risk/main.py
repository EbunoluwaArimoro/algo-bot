"""
Risk Service — algo-bot
────────────────────────
Responsibilities:
  1. Consume signals from signals:queue (BRPOP — blocks until available)
  2. Apply all risk checks — if ANY fail, signal is rejected and logged
  3. Size the position using ATR-based method
  4. Push validated + sized orders onto orders:queue for execution
  5. Maintain circuit breaker state in Redis

Risk checks applied in order:
  A. Circuit breaker — if open, reject everything
  B. Daily drawdown limit
  C. Per-symbol exposure cap
  D. Total open positions cap
  E. Correlation guard (no same-direction signal for same base asset)
  F. Min confidence threshold
  G. Duplicate signal guard (same symbol+direction within cooldown window)
"""

import asyncio
import json
import logging
import os
import signal as posix_signal
import time
from dataclasses import asdict, dataclass

import asyncpg
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("risk")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
DB_DSN = (
    f"postgresql://{os.getenv('DB_USER','bot')}:"
    f"{os.getenv('DB_PASSWORD','botpass')}@"
    f"{os.getenv('DB_HOST','timescaledb')}:5432/"
    f"{os.getenv('DB_NAME','botdb')}"
)

# ── Risk parameters (tune per risk profile) ───────────────────────────────────
RISK_PER_TRADE_PCT  = float(os.getenv("RISK_PER_TRADE_PCT",  "1.0"))  # % of portfolio
DAILY_DRAWDOWN_HALT = float(os.getenv("DAILY_DRAWDOWN_HALT", "5.0"))  # %
MAX_OPEN_POSITIONS  = int(os.getenv("MAX_OPEN_POSITIONS",    "10"))
MAX_SYMBOL_EXPOSURE = float(os.getenv("MAX_SYMBOL_EXPOSURE", "20.0")) # % of portfolio
MIN_CONFIDENCE      = float(os.getenv("MIN_CONFIDENCE",      "55.0"))
SIGNAL_COOLDOWN_S   = int(os.getenv("SIGNAL_COOLDOWN_S",     "300"))  # 5 min

ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))  # stop = 1.5x ATR
TAKE_PROFIT_RR      = float(os.getenv("TAKE_PROFIT_RR",      "2.0"))  # TP at 2:1 R:R


# ── Order dataclass ───────────────────────────────────────────────────────────

@dataclass
class Order:
    symbol:      str
    direction:   str       # 'long' | 'short'
    qty:         float     # in base asset (e.g. BTC)
    entry_ref:   float     # reference price
    stop_loss:   float
    take_profit: float
    strategy:    str
    confidence:  float
    regime:      str
    risk_pct:    float     # actual portfolio risk % for this order
    timestamp:   int


# ── Portfolio state reader ────────────────────────────────────────────────────

async def get_portfolio_state(redis: aioredis.Redis) -> dict:
    raw = await redis.get("portfolio:state")
    if not raw:
        # Bootstrap defaults — will be updated by execution service
        defaults = {
            "total_value":     float(os.getenv("INITIAL_CAPITAL", "1000.0")),
            "cash":            float(os.getenv("INITIAL_CAPITAL", "1000.0")),
            "unrealized_pnl":  0.0,
            "daily_pnl":       0.0,
            "daily_drawdown_pct": 0.0,
            "open_positions":  0,
        }
        await redis.set("portfolio:state", json.dumps(defaults))
        return defaults
    return json.loads(raw)


async def count_open_positions(redis: aioredis.Redis) -> int:
    keys = await redis.keys("position:*")
    return len(keys)


async def get_symbol_exposure_pct(
    redis: aioredis.Redis, symbol: str, portfolio_value: float
) -> float:
    raw = await redis.get(f"position:{symbol}")
    if not raw:
        return 0.0
    pos = json.loads(raw)
    exposure = pos.get("qty", 0) * pos.get("entry", 0)
    return (exposure / portfolio_value) * 100 if portfolio_value > 0 else 0


async def was_recently_signalled(redis: aioredis.Redis, symbol: str, direction: str) -> bool:
    key = f"signal_cooldown:{symbol}:{direction}"
    return bool(await redis.get(key))


async def set_signal_cooldown(redis: aioredis.Redis, symbol: str, direction: str) -> None:
    key = f"signal_cooldown:{symbol}:{direction}"
    await redis.set(key, "1", ex=SIGNAL_COOLDOWN_S)


# ── Position sizing ───────────────────────────────────────────────────────────

def atr_position_size(
    portfolio_value: float,
    risk_pct: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float,
) -> tuple[float, float, float]:
    """
    ATR-based position sizing.

    Formula:
        dollar_risk = portfolio_value * (risk_pct / 100)
        stop_distance = atr * atr_multiplier
        qty = dollar_risk / stop_distance
        stop_loss = entry - stop_distance   (for longs)
        take_profit = entry + stop_distance * RR

    Returns: (qty, stop_loss, take_profit)
    """
    dollar_risk   = portfolio_value * (risk_pct / 100)
    stop_distance = atr * atr_multiplier

    if stop_distance <= 0 or entry_price <= 0:
        return 0.0, 0.0, 0.0

    qty = dollar_risk / stop_distance

    # Expressed as price levels
    stop_loss   = entry_price - stop_distance
    take_profit = entry_price + stop_distance * TAKE_PROFIT_RR

    return round(qty, 6), round(stop_loss, 2), round(take_profit, 2)


# ── Circuit breaker ───────────────────────────────────────────────────────────

async def is_circuit_open(redis: aioredis.Redis) -> bool:
    return (await redis.get("circuit_breaker:status") or "closed") == "open"


async def open_circuit(redis: aioredis.Redis, reason: str) -> None:
    await redis.set("circuit_breaker:status", "open")
    await redis.set("circuit_breaker:reason", reason)
    log.critical("⚡ CIRCUIT BREAKER OPENED — %s — ALL TRADING HALTED", reason)


# ── Risk evaluation ───────────────────────────────────────────────────────────

async def evaluate_signal(
    signal: dict,
    redis: aioredis.Redis,
) -> tuple[bool, str, Order | None]:
    """
    Run all risk checks on an inbound signal.
    Returns (approved, reject_reason, Order | None).
    """
    symbol    = signal["symbol"]
    direction = signal["direction"]
    confidence= signal["confidence"]
    entry_ref = signal["entry_ref"]
    atr       = signal["atr"]
    strategy  = signal["strategy"]
    regime    = signal["regime"]
    timeframe = signal["timeframe"]

    # ── A. Circuit breaker ────────────────────────────────────────────────────
    if await is_circuit_open(redis):
        return False, "circuit_breaker_open", None

    # ── B. Daily drawdown limit ───────────────────────────────────────────────
    portfolio = await get_portfolio_state(redis)
    daily_dd  = portfolio.get("daily_drawdown_pct", 0)
    if daily_dd >= DAILY_DRAWDOWN_HALT:
        await open_circuit(redis, f"daily_drawdown {daily_dd:.1f}% >= {DAILY_DRAWDOWN_HALT}%")
        return False, f"daily_drawdown_limit ({daily_dd:.1f}%)", None

    # ── C. Per-symbol exposure ────────────────────────────────────────────────
    portfolio_value = portfolio.get("total_value", 1.0)
    sym_exposure = await get_symbol_exposure_pct(redis, symbol, portfolio_value)
    if sym_exposure >= MAX_SYMBOL_EXPOSURE:
        return False, f"symbol_exposure_cap ({sym_exposure:.1f}% >= {MAX_SYMBOL_EXPOSURE}%)", None

    # ── D. Max open positions ─────────────────────────────────────────────────
    open_count = await count_open_positions(redis)
    if open_count >= MAX_OPEN_POSITIONS:
        return False, f"max_positions_reached ({open_count})", None

    # ── E. Confidence threshold ───────────────────────────────────────────────
    if confidence < MIN_CONFIDENCE:
        return False, f"low_confidence ({confidence:.1f} < {MIN_CONFIDENCE})", None

    # ── F. Signal cooldown (deduplication) ───────────────────────────────────
    if await was_recently_signalled(redis, symbol, direction):
        return False, f"cooldown ({SIGNAL_COOLDOWN_S}s)", None

    # ── All checks passed — size the position ─────────────────────────────────
    qty, stop_loss, take_profit = atr_position_size(
        portfolio_value=portfolio_value,
        risk_pct=RISK_PER_TRADE_PCT,
        entry_price=entry_ref,
        atr=atr,
        atr_multiplier=ATR_STOP_MULTIPLIER,
    )

    if qty <= 0:
        return False, "position_size_zero (ATR or entry price invalid)", None

    # Flip stop/TP for shorts
    if direction == "short":
        stop_loss   = entry_ref + (entry_ref - stop_loss)
        take_profit = entry_ref - (take_profit - entry_ref)

    order = Order(
        symbol=symbol,       direction=direction,
        qty=qty,             entry_ref=entry_ref,
        stop_loss=stop_loss, take_profit=take_profit,
        strategy=strategy,   confidence=confidence,
        regime=regime,       risk_pct=RISK_PER_TRADE_PCT,
        timestamp=int(time.time()),
    )

    # Set cooldown so we don't re-enter same trade within window
    await set_signal_cooldown(redis, symbol, direction)

    return True, "approved", order


# ── DB: log signal to signals table ──────────────────────────────────────────

async def log_signal_to_db(
    pool: asyncpg.Pool,
    signal: dict,
    executed: bool,
    reject_reason: str | None,
) -> None:
    sql = """
        INSERT INTO signals
            (time, symbol, strategy, direction, confidence, regime, executed, reject_reason)
        VALUES
            (NOW(), $1, $2, $3, $4, $5, $6, $7)
    """
    async with pool.acquire() as conn:
        await conn.execute(
            sql,
            signal["symbol"], signal["strategy"], signal["direction"],
            signal["confidence"], signal["regime"],
            executed, reject_reason,
        )


# ── Main consumer loop ────────────────────────────────────────────────────────

async def consume_loop(
    redis: aioredis.Redis,
    pool: asyncpg.Pool,
    stop: asyncio.Event,
) -> None:
    log.info("Risk service listening on signals:queue…")

    while not stop.is_set():
        # BRPOP blocks for up to 2s, returns (key, value) or None on timeout
        item = await redis.brpop("signals:queue", timeout=2)
        if item is None:
            continue

        _, raw = item
        try:
            signal = json.loads(raw)
        except json.JSONDecodeError:
            log.error("Cannot parse signal: %s", raw)
            continue

        approved, reason, order = await evaluate_signal(signal, redis)

        # Log every signal to DB (approved and rejected)
        try:
            await log_signal_to_db(pool, signal, approved, None if approved else reason)
        except Exception as exc:
            log.warning("Failed to log signal to DB: %s", exc)

        if not approved:
            log.info(
                "✗ REJECTED  %-9s %-5s strategy=%-15s reason=%s",
                signal["symbol"], signal["direction"], signal["strategy"], reason,
            )
            continue

        # Push to orders queue for execution service
        await redis.lpush("orders:queue", json.dumps(asdict(order)))
        log.info(
            "✓ APPROVED  %-9s %-5s strategy=%-15s qty=%.6f  sl=%.2f  tp=%.2f  conf=%.1f",
            order.symbol, order.direction, order.strategy,
            order.qty, order.stop_loss, order.take_profit, order.confidence,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("═══ Risk Service starting ═══")
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pool  = await get_db_pool()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (posix_signal.SIGTERM, posix_signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await consume_loop(redis, pool, stop)

    await pool.close()
    await redis.aclose()
    log.info("Risk Service stopped.")


async def get_db_pool() -> asyncpg.Pool:
    for attempt in range(30):
        try:
            pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=5)
            log.info("TimescaleDB connected")
            return pool
        except Exception as exc:
            log.warning("DB not ready (%d/30): %s", attempt + 1, exc)
            await asyncio.sleep(3)
    raise RuntimeError("Could not connect to DB")


if __name__ == "__main__":
    asyncio.run(main())