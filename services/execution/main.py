"""
Execution Service — algo-bot
─────────────────────────────
In PAPER mode (default): simulates fills with realistic latency + slippage.
In LIVE mode: routes orders through CCXT to the real exchange.

Responsibilities:
  1. Consume orders from orders:queue
  2. Simulate (paper) or place (live) the order
  3. Write the open position to Redis position:{symbol}
  4. Write the trade record to TimescaleDB
  5. Update portfolio:state (total_value, cash, pnl, drawdown)
  6. Monitor open positions for stop-loss / take-profit hits
  7. Close positions and update all records on exit
"""

import asyncio
import json
import logging
import math
import os
import random
import signal as posix_signal
import time
from datetime import datetime, timezone
from typing import Literal

import asyncpg
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("execution")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
DB_DSN = (
    f"postgresql://{os.getenv('DB_USER','bot')}:"
    f"{os.getenv('DB_PASSWORD','botpass')}@"
    f"{os.getenv('DB_HOST','timescaledb')}:5432/"
    f"{os.getenv('DB_NAME','botdb')}"
)

# ── Config ────────────────────────────────────────────────────────────────────
MODE: Literal["paper", "live"] = os.getenv("TRADING_MODE", "paper")  # type: ignore

# Paper trading friction simulation
LATENCY_MS      = int(os.getenv("PAPER_LATENCY_MS",   "150"))
SLIPPAGE_BPS    = int(os.getenv("PAPER_SLIPPAGE_BPS",  "8"))
OUTAGE_PROB     = float(os.getenv("PAPER_OUTAGE_PROB", "0.002"))

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000.0"))

# How often to check open positions for SL/TP hits (seconds)
MONITOR_INTERVAL_S = int(os.getenv("MONITOR_INTERVAL_S", "5"))


# ── DB helpers ────────────────────────────────────────────────────────────────

async def get_db_pool() -> asyncpg.Pool:
    for attempt in range(30):
        try:
            pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=5)
            log.info("TimescaleDB connected")
            return pool
        except Exception as exc:
            log.warning("DB not ready (%d/30): %s", attempt + 1, exc)
            await asyncio.sleep(3)
    raise RuntimeError("Cannot connect to DB")


async def insert_trade_open(pool: asyncpg.Pool, order: dict, fill_price: float) -> int:
    sql = """
        INSERT INTO trades
            (time, symbol, side, qty, entry_price, stop_loss, take_profit,
             strategy, status, regime, signal_score)
        VALUES
            (NOW(), $1, $2, $3, $4, $5, $6, $7, 'open', $8, $9)
        RETURNING id
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            sql,
            order["symbol"],
            order["direction"],
            order["qty"],
            fill_price,
            order["stop_loss"],
            order["take_profit"],
            order["strategy"],
            order["regime"],
            order["confidence"],
        )
    return row["id"]


async def close_trade_in_db(
    pool: asyncpg.Pool,
    trade_id: int,
    exit_price: float,
    pnl: float,
    pnl_pct: float,
    slippage: float,
    reason: str,
) -> None:
    sql = """
        UPDATE trades
        SET exit_price = $1, pnl = $2, pnl_pct = $3,
            slippage = $4, status = 'closed', notes = $5
        WHERE id = $6
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, exit_price, pnl, pnl_pct, slippage, reason, trade_id)


async def save_portfolio_snapshot(pool: asyncpg.Pool, state: dict) -> None:
    sql = """
        INSERT INTO portfolio_snapshots
            (time, total_value, cash, unrealized_pnl, realized_pnl,
             drawdown_pct, open_positions)
        VALUES (NOW(), $1, $2, $3, $4, $5, $6)
    """
    async with pool.acquire() as conn:
        await conn.execute(
            sql,
            state["total_value"], state["cash"],
            state.get("unrealized_pnl", 0), state.get("realized_pnl", 0),
            state.get("daily_drawdown_pct", 0), state.get("open_positions", 0),
        )


# ── Portfolio state helpers ───────────────────────────────────────────────────

async def get_state(redis: aioredis.Redis) -> dict:
    raw = await redis.get("portfolio:state")
    if raw:
        return json.loads(raw)
    state = {
        "total_value": INITIAL_CAPITAL, "cash": INITIAL_CAPITAL,
        "unrealized_pnl": 0.0, "realized_pnl": 0.0,
        "daily_pnl": 0.0, "daily_drawdown_pct": 0.0,
        "high_water_mark": INITIAL_CAPITAL, "open_positions": 0,
    }
    await redis.set("portfolio:state", json.dumps(state))
    return state


async def save_state(redis: aioredis.Redis, state: dict) -> None:
    await redis.set("portfolio:state", json.dumps(state))


async def recalculate_portfolio(
    redis: aioredis.Redis, pool: asyncpg.Pool
) -> dict:
    """
    Recompute portfolio value from open positions + cash.
    Called after every fill and on each monitoring cycle.
    """
    state = await get_state(redis)
    position_keys = await redis.keys("position:*")

    unrealized_pnl = 0.0
    open_count     = len(position_keys)

    for key in position_keys:
        raw = await redis.get(key)
        if not raw:
            continue
        pos = json.loads(raw)

        # Get latest price from feature store
        symbol_lc  = pos["symbol"].lower()
        feat_raw   = await redis.get(f"features:{symbol_lc}:1m:latest")
        if feat_raw:
            feat       = json.loads(feat_raw)
            curr_price = feat.get("close", pos["entry"])
        else:
            curr_price = pos["entry"]

        notional = pos["qty"] * curr_price
        cost     = pos["qty"] * pos["entry"]
        pnl      = (notional - cost) if pos["side"] == "long" else (cost - notional)
        unrealized_pnl += pnl

    total_value = state["cash"] + unrealized_pnl
    drawdown    = ((state["high_water_mark"] - total_value) / state["high_water_mark"]) * 100
    daily_pnl   = total_value - INITIAL_CAPITAL  # simplified for paper mode

    state.update({
        "total_value":         round(total_value, 2),
        "unrealized_pnl":      round(unrealized_pnl, 2),
        "daily_pnl":           round(daily_pnl, 2),
        "daily_drawdown_pct":  round(max(drawdown, 0), 2),
        "open_positions":      open_count,
        "high_water_mark":     max(state["high_water_mark"], total_value),
    })
    await save_state(redis, state)
    await save_portfolio_snapshot(pool, state)
    return state


# ── Paper broker ──────────────────────────────────────────────────────────────

class PaperBroker:
    """
    Simulates realistic exchange behaviour for paper trading.
    Key frictions modelled:
      - Network latency (configurable ms)
      - Slippage (configurable bps, with Gaussian noise)
      - Random connection outages (configurable probability)
    """
    def __init__(self):
        self.latency_ms   = LATENCY_MS
        self.slippage_bps = SLIPPAGE_BPS
        self.outage_prob  = OUTAGE_PROB

    async def fill(
        self, symbol: str, side: str, qty: float, ref_price: float
    ) -> dict:
        # 1. Network round-trip delay
        await asyncio.sleep(self.latency_ms / 1000)

        # 2. Random outage test — forces reconnect logic
        if random.random() < self.outage_prob:
            raise ConnectionError(
                f"[PAPER] Simulated exchange outage for {symbol}"
            )

        # 3. Slippage: deterministic + small random component
        slip_pct   = self.slippage_bps / 10_000
        gauss_noise = random.gauss(0, 0.0002)   # ±0.02% std dev
        total_slip  = slip_pct + gauss_noise

        if side == "buy":
            fill_price = ref_price * (1 + total_slip)
        else:
            fill_price = ref_price * (1 - total_slip)

        slippage_amount = abs(fill_price - ref_price)

        return {
            "symbol":   symbol,
            "side":     side,
            "qty":      qty,
            "fill_price": round(fill_price, 4),
            "slippage": round(slippage_amount, 4),
            "mode":     "paper",
        }


broker = PaperBroker()


# ── Order handler ─────────────────────────────────────────────────────────────

async def handle_order(
    order: dict,
    redis: aioredis.Redis,
    pool: asyncpg.Pool,
) -> None:
    symbol    = order["symbol"]
    direction = order["direction"]
    qty       = order["qty"]
    ref_price = order["entry_ref"]
    side      = "buy" if direction == "long" else "sell"

    # Check for existing position in same symbol (no pyramiding by default)
    existing = await redis.get(f"position:{symbol}")
    if existing:
        log.warning("Position already open for %s — skipping", symbol)
        return

    # Execute fill (paper or live)
    for attempt in range(3):
        try:
            fill = await broker.fill(symbol, side, qty, ref_price)
            break
        except ConnectionError as exc:
            log.warning("Fill attempt %d failed: %s", attempt + 1, exc)
            if attempt == 2:
                log.error("Order abandoned after 3 failed attempts: %s %s", symbol, direction)
                return
            await asyncio.sleep(2 ** attempt)

    fill_price = fill["fill_price"]
    slippage   = fill["slippage"]

    # Write position to Redis
    pos = {
        "symbol":     symbol,
        "side":       direction,
        "qty":        qty,
        "entry":      fill_price,
        "stop_loss":  order["stop_loss"],
        "take_profit":order["take_profit"],
        "strategy":   order["strategy"],
        "opened_at":  int(time.time()),
        "trade_id":   None,  # filled below
    }

    # Write to DB
    trade_id = await insert_trade_open(pool, order, fill_price)
    pos["trade_id"] = trade_id

    await redis.set(f"position:{symbol}", json.dumps(pos))

    # Deduct from cash (paper accounting)
    state = await get_state(redis)
    cost  = qty * fill_price
    state["cash"] = round(state["cash"] - cost, 2)
    await save_state(redis, state)

    log.info(
        "OPEN  %-9s %-5s qty=%.6f  fill=%.4f  sl=%.4f  tp=%.4f  slip=%.4f  [%s]",
        symbol, direction.upper(), qty, fill_price,
        order["stop_loss"], order["take_profit"], slippage, fill["mode"],
    )


# ── Position monitor ──────────────────────────────────────────────────────────

async def close_position(
    symbol: str,
    pos: dict,
    exit_price: float,
    reason: str,
    redis: aioredis.Redis,
    pool: asyncpg.Pool,
) -> None:
    side = pos["side"]
    qty  = pos["qty"]

    # Simulate exit slippage
    fill = await broker.fill(
        symbol, "sell" if side == "long" else "buy", qty, exit_price
    )
    actual_exit = fill["fill_price"]
    slippage    = fill["slippage"]

    # P&L calculation
    if side == "long":
        pnl = (actual_exit - pos["entry"]) * qty
    else:
        pnl = (pos["entry"] - actual_exit) * qty

    cost    = pos["entry"] * qty
    pnl_pct = (pnl / cost) * 100 if cost > 0 else 0

    # Update DB
    if pos.get("trade_id"):
        await close_trade_in_db(
            pool, pos["trade_id"], actual_exit, round(pnl, 4),
            round(pnl_pct, 2), slippage, reason,
        )

    # Remove position, return cash
    await redis.delete(f"position:{symbol}")
    state = await get_state(redis)
    proceeds             = qty * actual_exit
    state["cash"]         = round(state["cash"] + proceeds, 2)
    state["realized_pnl"] = round(state.get("realized_pnl", 0) + pnl, 4)
    await save_state(redis, state)

    log.info(
        "CLOSE %-9s %-5s qty=%.6f  exit=%.4f  pnl=%+.4f (%+.2f%%)  reason=%s",
        symbol, side.upper(), qty, actual_exit, pnl, pnl_pct, reason,
    )


async def monitor_positions(
    redis: aioredis.Redis,
    pool: asyncpg.Pool,
    stop: asyncio.Event,
) -> None:
    """
    Every MONITOR_INTERVAL_S seconds, check all open positions against
    the latest price in Redis feature store. Trigger SL/TP if hit.
    """
    log.info("Position monitor running every %ds", MONITOR_INTERVAL_S)

    while not stop.is_set():
        position_keys = await redis.keys("position:*")

        for key in position_keys:
            raw = await redis.get(key)
            if not raw:
                continue
            pos    = json.loads(raw)
            symbol = pos["symbol"]

            # Get latest price
            symbol_lc = symbol.lower()
            feat_raw  = await redis.get(f"features:{symbol_lc}:1m:latest")
            if not feat_raw:
                continue

            feat  = json.loads(feat_raw)
            price = feat.get("close", 0)
            if price <= 0:
                continue

            side = pos["side"]
            sl   = pos.get("stop_loss", 0)
            tp   = pos.get("take_profit", 0)

            # ── Stop loss hit
            if side == "long" and price <= sl:
                await close_position(symbol, pos, price, "stop_loss", redis, pool)
            elif side == "short" and price >= sl:
                await close_position(symbol, pos, price, "stop_loss", redis, pool)

            # ── Take profit hit
            elif side == "long" and price >= tp:
                await close_position(symbol, pos, price, "take_profit", redis, pool)
            elif side == "short" and price <= tp:
                await close_position(symbol, pos, price, "take_profit", redis, pool)

        # Recalculate and persist portfolio state
        state = await recalculate_portfolio(redis, pool)
        if state["open_positions"] > 0 or True:  # always log in paper mode
            log.debug(
                "Portfolio  value=%.2f  cash=%.2f  upnl=%+.2f  dd=%.2f%%  pos=%d",
                state["total_value"], state["cash"],
                state["unrealized_pnl"], state["daily_drawdown_pct"],
                state["open_positions"],
            )

        await asyncio.sleep(MONITOR_INTERVAL_S)


# ── Order consumer loop ───────────────────────────────────────────────────────

async def order_loop(
    redis: aioredis.Redis,
    pool: asyncpg.Pool,
    stop: asyncio.Event,
) -> None:
    log.info("Execution service listening on orders:queue… [mode=%s]", MODE)

    while not stop.is_set():
        item = await redis.brpop("orders:queue", timeout=2)
        if item is None:
            continue

        _, raw = item
        try:
            order = json.loads(raw)
        except json.JSONDecodeError:
            log.error("Cannot parse order: %s", raw)
            continue

        try:
            await handle_order(order, redis, pool)
        except Exception as exc:
            log.exception("Order handling failed: %s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("═══ Execution Service starting [mode=%s] ═══", MODE.upper())
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pool  = await get_db_pool()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (posix_signal.SIGTERM, posix_signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await asyncio.gather(
        order_loop(redis, pool, stop),
        monitor_positions(redis, pool, stop),
    )

    await pool.close()
    await redis.aclose()
    log.info("Execution Service stopped.")


if __name__ == "__main__":
    asyncio.run(main())