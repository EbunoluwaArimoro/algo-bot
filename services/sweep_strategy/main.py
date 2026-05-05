"""
services/sweep_strategy/main.py
────────────────────────────────
Live paper trading implementation of the Liquidity Sweep strategy.
"""
import ta
import asyncio
import json
import logging
import os
import signal as posix_signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

import asyncpg
import numpy as np
import redis.asyncio as aioredis
from dotenv import load_dotenv
import xgboost as xgb  # Raw engine

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sweep_strategy")

# ── Config ────────────────────────────────────────────────────────────────────
TRADING_MODE        = os.getenv("TRADING_MODE", "paper")
SYMBOLS             = os.getenv("SYMBOLS", "solusdt").split(",")
THRESHOLD           = float(os.getenv("SWEEP_THRESHOLD", "0.60"))
SWEEP_MULT          = float(os.getenv("SWEEP_MULT", "0.6"))
TIGHT_STOP_MULT     = float(os.getenv("TIGHT_STOP_MULT", "0.3"))
LOOKBACK            = int(os.getenv("SWEEP_LOOKBACK", "10"))
WINDOW_BARS         = int(os.getenv("SWEEP_WINDOW_BARS", "4"))
TARGET_MODE         = os.getenv("SWEEP_TARGET_MODE", "opposite")
RISK_PCT            = float(os.getenv("RISK_PER_TRADE_PCT", "1.0")) / 100
INITIAL_CAPITAL     = float(os.getenv("INITIAL_CAPITAL", "10000.0"))
BE_TRIGGER_R        = float(os.getenv("BE_TRIGGER_R", "0.5"))
PARTIAL_R           = float(os.getenv("PARTIAL_R", "1.0"))
TRAIL_TRIGGER_R     = float(os.getenv("TRAIL_TRIGGER_R", "1.0"))
TRAIL_ATR_MULT      = float(os.getenv("TRAIL_ATR_MULT", "1.0"))

REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
DB_DSN      = (
    f"postgresql://{os.getenv('DB_USER','bot')}:"
    f"{os.getenv('DB_PASSWORD','botpass')}@"
    f"{os.getenv('DB_HOST','timescaledb')}:5432/"
    f"{os.getenv('DB_NAME','botdb')}"
)
COST_PER_SIDE = 0.0018   


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model_and_meta():
    model_dir  = os.getenv("MODEL_DIR", "ml/models")
    model_slug = os.getenv("MODEL_SLUG", "SOL_USDT_4h_alpha_volatility")
    model_path = os.path.join(model_dir, f"{model_slug}.json")
    meta_path  = os.path.join(model_dir, f"{model_slug}_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # USE RAW C++ BOOSTER (No scikit-learn dependency)
    model = xgb.Booster()
    model.load_model(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    feat_cols = meta.get("feature_cols", [])
    log.info("Model loaded: AUC=%.4f  threshold=%.2f  features=%d",
             meta.get("auc_roc", 0), THRESHOLD, len(feat_cols))
    return model, feat_cols


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(features_dict: dict, feat_cols: list[str]) -> np.ndarray:
    vals = []
    for col in feat_cols:
        val = features_dict.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            vals.append(0.0)
        elif isinstance(val, bool):
            vals.append(float(val))
        else:
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                vals.append(0.0)
    return np.array(vals, dtype=np.float32).reshape(1, -1)


# ── Redis data readers ─────────────────────────────────────────────────────────

async def get_latest_features(redis: aioredis.Redis, symbol: str, timeframe: str = "1h") -> dict | None:
    raw = await redis.get(f"features:{symbol}:{timeframe}:latest")
    if not raw:
        return None
    return json.loads(raw)


async def get_candle_history(
    redis: aioredis.Redis,
    pool: asyncpg.Pool,
    symbol: str,
    lookback: int,
    timeframe: str = "4h",
) -> list[dict]:
    db_symbol = symbol.upper().replace("usdt", "USDT")
    sql = """
        SELECT time, high, low, close, volume
        FROM ohlcv
        WHERE symbol = $1 AND timeframe = $2
        ORDER BY time DESC
        LIMIT $3
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, db_symbol, timeframe, lookback + 5)

    if not rows:
        return []

    candles = [dict(r) for r in reversed(rows)]
    return candles


# ── Sweep level computation ────────────────────────────────────────────────────

def compute_sweep_levels(
    candles: list[dict],
    current_atr: float,
    lookback: int,
) -> dict | None:
    if len(candles) < lookback:
        return None

    window      = candles[-lookback:]
    support     = min(c["low"]  for c in window)
    resistance  = max(c["high"] for c in window)
    range_size  = resistance - support
    range_mid   = (resistance + support) / 2

    if current_atr <= 0:
        return None

    limit_buy   = support    - SWEEP_MULT * current_atr
    limit_sell  = resistance + SWEEP_MULT * current_atr
    stop_buy    = limit_buy  - TIGHT_STOP_MULT * current_atr
    stop_sell   = limit_sell + TIGHT_STOP_MULT * current_atr

    return {
        "support":    round(support,    6),
        "resistance": round(resistance, 6),
        "range_mid":  round(range_mid,  6),
        "range_size": round(range_size, 6),
        "limit_buy":  round(limit_buy,  6),
        "limit_sell": round(limit_sell, 6),
        "stop_buy":   round(stop_buy,   6),
        "stop_sell":  round(stop_sell,  6),
    }


def compute_target(
    direction: str,
    entry_price: float,
    stop_loss: float,
    levels: dict,
) -> float:
    risk = abs(entry_price - stop_loss)
    if TARGET_MODE == "opposite":
        return levels["resistance"] if direction == "long" else levels["support"]
    elif TARGET_MODE == "midpoint":
        if direction == "long":
            return levels["range_mid"] + levels["range_size"] * 0.3
        else:
            return levels["range_mid"] - levels["range_size"] * 0.3
    else:  
        return (entry_price + risk * 2.0 if direction == "long"
                else entry_price - risk * 2.0)


# ── Portfolio state ────────────────────────────────────────────────────────────

async def get_portfolio_state(redis: aioredis.Redis) -> dict:
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


async def save_portfolio_state(redis: aioredis.Redis, state: dict) -> None:
    await redis.set("portfolio:state", json.dumps(state))


# ── Paper order manager ────────────────────────────────────────────────────────

class PaperOrderManager:
    def __init__(self, redis: aioredis.Redis, pool: asyncpg.Pool):
        self.redis = redis
        self.pool  = pool

    async def has_pending_order(self, symbol: str) -> bool:
        return bool(await self.redis.get(f"sweep_order:{symbol}"))

    async def has_open_position(self, symbol: str) -> bool:
        return bool(await self.redis.get(f"position:{symbol}"))

    async def place_sweep_order(
        self,
        symbol:    str,
        levels:    dict,
        vol_prob:  float,
        regime:    str,
        expires_at:float,  
    ) -> None:
        order = {
            "symbol":     symbol,
            "limit_buy":  levels["limit_buy"],
            "limit_sell": levels["limit_sell"],
            "stop_buy":   levels["stop_buy"],
            "stop_sell":  levels["stop_sell"],
            "support":    levels["support"],
            "resistance": levels["resistance"],
            "range_mid":  levels["range_mid"],
            "range_size": levels["range_size"],
            "vol_prob":   vol_prob,
            "regime":     regime,
            "expires_at": expires_at,
            "placed_at":  time.time(),
            "status":     "pending",
        }
        await self.redis.set(
            f"sweep_order:{symbol}",
            json.dumps(order),
            ex=int(expires_at - time.time()) + 60,
        )
        log.info(
            "SWEEP ORDER placed  %s  buy_limit=%.4f  sell_limit=%.4f  "
            "stop_b=%.4f  stop_s=%.4f  prob=%.3f",
            symbol, levels["limit_buy"], levels["limit_sell"],
            levels["stop_buy"], levels["stop_sell"], vol_prob,
        )

    async def cancel_order(self, symbol: str) -> None:
        await self.redis.delete(f"sweep_order:{symbol}")
        log.info("Order cancelled: %s", symbol)

    async def check_and_fill(self, symbol: str, current_price: float) -> None:
        raw = await self.redis.get(f"sweep_order:{symbol}")
        if not raw:
            return

        order = json.loads(raw)

        if time.time() > order["expires_at"]:
            await self.cancel_order(symbol)
            log.info("Order expired: %s", symbol)
            return

        lb = order["limit_buy"]
        ls = order["limit_sell"]

        long_filled  = current_price <= lb
        short_filled = current_price >= ls

        if not long_filled and not short_filled:
            return   

        if long_filled:
            direction   = "long"
            entry_price = lb * (1 + COST_PER_SIDE)   
            stop_loss   = order["stop_buy"]
        else:
            direction   = "short"
            entry_price = ls * (1 - COST_PER_SIDE)
            stop_loss   = order["stop_sell"]

        await self.cancel_order(symbol)

        take_profit = compute_target(direction, entry_price, stop_loss, order)

        state       = await get_portfolio_state(self.redis)
        portfolio   = state["total_value"]
        stop_dist   = abs(entry_price - stop_loss)
        if stop_dist <= 0:
            return

        dollar_risk = portfolio * RISK_PCT
        qty         = round(dollar_risk / stop_dist, 6)

        if qty <= 0:
            return

        position = {
            "symbol":       symbol,
            "side":         direction,
            "qty":          qty,
            "entry":        entry_price,
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "trail_stop":   0.0,
            "breakeven_set":False,
            "partial_done": False,
            "support":      order["support"],
            "resistance":   order["resistance"],
            "range_mid":    order["range_mid"],
            "opened_at":    time.time(),
            "vol_prob":     order["vol_prob"],
            "regime":       order["regime"],
        }
        await self.redis.set(f"position:{symbol}", json.dumps(position))
        await self._record_open_trade(position)

        log.info(
            "FILLED %s %s  qty=%.4f  entry=%.4f  stop=%.4f  tp=%.4f",
            direction.upper(), symbol, qty, entry_price, stop_loss, take_profit,
        )

    async def check_stops_and_targets(self, symbol: str, price: float, atr: float) -> None:
        raw = await self.redis.get(f"position:{symbol}")
        if not raw:
            return

        pos  = json.loads(raw)
        side = pos["side"]
        entry= pos["entry"]
        qty  = pos["qty"]
        sl   = pos["stop_loss"]
        tp   = pos["take_profit"]
        ts   = pos.get("trail_stop", 0.0)

        risk_dist = abs(entry - sl)
        if risk_dist <= 0:
            return

        profit_r = ((price - entry) / risk_dist if side == "long"
                    else (entry - price) / risk_dist)

        if not pos.get("breakeven_set") and profit_r >= BE_TRIGGER_R:
            buf = risk_dist * 0.05
            new_be = (entry + buf if side == "long" else entry - buf)
            if (side == "long" and new_be > sl) or (side == "short" and new_be < sl):
                pos["stop_loss"]    = round(new_be, 6)
                pos["breakeven_set"]= True
                sl                  = pos["stop_loss"]
                await self.redis.set(f"position:{symbol}", json.dumps(pos))
                log.info("Breakeven set: %s  new_sl=%.4f", symbol, new_be)

        if not pos.get("partial_done") and profit_r >= PARTIAL_R:
            partial_qty = round(qty * 0.40, 6)
            if partial_qty > 0:
                partial_exit= price * (1 - COST_PER_SIDE if side == "long" else 1 + COST_PER_SIDE)
                partial_pnl = ((partial_exit - entry) * partial_qty if side == "long"
                               else (entry - partial_exit) * partial_qty)
                pos["qty"]         -= partial_qty
                pos["partial_done"] = True
                await self.redis.set(f"position:{symbol}", json.dumps(pos))
                await self._update_portfolio(partial_pnl)
                log.info("Partial exit: %s  pnl=%.4f", symbol, partial_pnl)

        if profit_r >= TRAIL_TRIGGER_R and atr > 0:
            trail_dist = atr * TRAIL_ATR_MULT
            if side == "long":
                new_ts = price - trail_dist
                if new_ts > ts:
                    pos["trail_stop"] = round(new_ts, 6)
                    await self.redis.set(f"position:{symbol}", json.dumps(pos))
                    ts = pos["trail_stop"]
            else:
                new_ts = price + trail_dist
                if ts == 0 or new_ts < ts:
                    pos["trail_stop"] = round(new_ts, 6)
                    await self.redis.set(f"position:{symbol}", json.dumps(pos))
                    ts = pos["trail_stop"]

        active_sl = ts if ts > 0 else pos["stop_loss"]

        exit_reason = None
        exit_price  = 0.0

        if side == "long":
            if price <= active_sl:
                exit_price  = active_sl
                exit_reason = ("trail_stop" if ts > 0
                               else "breakeven" if pos.get("breakeven_set")
                               else "stop_loss")
            elif price >= tp:
                exit_price  = tp
                exit_reason = "take_profit"
        else:
            if price >= active_sl:
                exit_price  = active_sl
                exit_reason = ("trail_stop" if ts > 0
                               else "breakeven" if pos.get("breakeven_set")
                               else "stop_loss")
            elif price <= tp:
                exit_price  = tp
                exit_reason = "take_profit"

        if exit_reason:
            await self._close_position(symbol, pos, exit_price, exit_reason)

    async def _close_position(
        self,
        symbol: str,
        pos: dict,
        exit_price: float,
        reason: str,
    ) -> None:
        side = pos["side"]
        qty  = pos["qty"]
        entry= pos["entry"]

        net_exit = (exit_price * (1 - COST_PER_SIDE) if side == "long"
                    else exit_price * (1 + COST_PER_SIDE))
        pnl      = ((net_exit - entry) * qty if side == "long"
                    else (entry - net_exit) * qty)
        cost     = qty * entry
        pnl_pct  = (pnl / cost) * 100 if cost > 0 else 0

        await self.redis.delete(f"position:{symbol}")
        await self._update_portfolio(pnl)
        await self._record_close_trade(pos, net_exit, pnl, pnl_pct, reason)

        log.info(
            "CLOSED %s %s  exit=%.4f  pnl=%+.4f (%+.2f%%)  reason=%s",
            side.upper(), symbol, net_exit, pnl, pnl_pct, reason,
        )

    async def _update_portfolio(self, pnl: float) -> None:
        state = await get_portfolio_state(self.redis)
        state["cash"]         = round(state["cash"] + pnl, 4)
        state["realized_pnl"] = round(state.get("realized_pnl", 0) + pnl, 4)
        await save_portfolio_state(self.redis, state)

    async def _record_open_trade(self, pos: dict) -> None:
        sql = """
            INSERT INTO trades
                (time, symbol, side, qty, entry_price, stop_loss, take_profit,
                 strategy, status, regime, signal_score)
            VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, 'open', $8, $9)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    sql,
                    pos["symbol"].upper(), pos["side"], pos["qty"],
                    pos["entry"], pos["stop_loss"], pos["take_profit"],
                    "sweep_v2", pos.get("regime", "unknown"),
                    pos.get("vol_prob", 0) * 100,
                )
        except Exception as exc:
            log.warning("Failed to record open trade: %s", exc)

    async def _record_close_trade(
        self, pos: dict, exit_price: float, pnl: float, pnl_pct: float, reason: str
    ) -> None:
        sql = """
            UPDATE trades
            SET exit_price=$1, pnl=$2, pnl_pct=$3, status='closed', notes=$4
            WHERE symbol=$5 AND status='open'
            ORDER BY time DESC LIMIT 1
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    sql, exit_price, pnl, pnl_pct, reason,
                    pos["symbol"].upper(),
                )
        except Exception as exc:
            log.warning("Failed to record close trade: %s", exc)


# ── Signal generator ───────────────────────────────────────────────────────────

class SweepSignalGenerator:
    def __init__(
        self,
        model,
        feat_cols: list[str],
        redis: aioredis.Redis,
        pool: asyncpg.Pool,
        order_manager: PaperOrderManager,
    ):
        self.model         = model
        self.feat_cols     = feat_cols
        self.redis         = redis
        self.pool          = pool
        self.om            = order_manager
        self.last_4h_ts    = {}   

    async def _get_4h_features(self, symbol: str) -> dict | None:
        raw = await self.redis.get(f"features:{symbol}:4h:latest")
        if not raw:
            raw = await self.redis.get(f"features:{symbol}:1h:latest")
        if not raw:
            return None
        return json.loads(raw)

    async def check_symbol(self, symbol: str) -> None:
        features = await self._get_4h_features(symbol)
        if not features:
            return

        ts = features.get("timestamp", 0)
        if ts <= self.last_4h_ts.get(symbol, 0):
            return

        self.last_4h_ts[symbol] = ts

        if await self.om.has_pending_order(symbol):
            return
        if await self.om.has_open_position(symbol):
            return

        # RAW BOOSTER PREDICTION MATH
        X       = extract_features(features, self.feat_cols)
        prob    = float(self.model.predict(xgb.DMatrix(X))[0])
        regime  = features.get("regime", "ranging")

        log.debug("%s  vol_prob=%.3f  regime=%s", symbol, prob, regime)

        if prob < THRESHOLD:
            return

        candles = await get_candle_history(self.redis, self.pool, symbol, LOOKBACK)
        if len(candles) < LOOKBACK:
            log.warning("%s: insufficient candle history (%d)", symbol, len(candles))
            return

        atr = features.get("atr") or 0
        if atr <= 0:
            return

        levels = compute_sweep_levels(candles, atr, LOOKBACK)
        if not levels:
            return

        expires_at = time.time() + WINDOW_BARS * 4 * 3600

        await self.om.place_sweep_order(
            symbol    = symbol,
            levels    = levels,
            vol_prob  = prob,
            regime    = regime,
            expires_at= expires_at,
        )

        log.info(
            "SIGNAL %s  prob=%.3f  support=%.4f  resistance=%.4f  "
            "limit_buy=%.4f  limit_sell=%.4f",
            symbol.upper(), prob,
            levels["support"], levels["resistance"],
            levels["limit_buy"], levels["limit_sell"],
        )

    async def run(self, stop: asyncio.Event) -> None:
        log.info("Signal generator running for symbols: %s", SYMBOLS)
        while not stop.is_set():
            for symbol in SYMBOLS:
                try:
                    await self.check_symbol(symbol)
                except Exception as exc:
                    log.exception("Error checking %s: %s", symbol, exc)
            await asyncio.sleep(60)   


# ── Price monitor ──────────────────────────────────────────────────────────────

class PriceMonitor:
    def __init__(
        self,
        redis: aioredis.Redis,
        order_manager: PaperOrderManager,
    ):
        self.redis = redis
        self.om    = order_manager

    async def run(self, stop: asyncio.Event) -> None:
        log.info("Price monitor running")
        while not stop.is_set():
            for symbol in SYMBOLS:
                try:
                    raw = await self.redis.get(f"features:{symbol}:1m:latest")
                    if not raw:
                        continue
                    feat  = json.loads(raw)
                    price = feat.get("close", 0)
                    atr   = feat.get("atr", 0) or 0

                    if price <= 0:
                        continue

                    await self.om.check_and_fill(symbol, price)
                    await self.om.check_stops_and_targets(symbol, price, atr)

                except Exception as exc:
                    log.exception("Price monitor error for %s: %s", symbol, exc)

            await asyncio.sleep(5)   


# ── Status reporter ────────────────────────────────────────────────────────────

async def status_loop(redis: aioredis.Redis, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(900)
        try:
            state = await get_portfolio_state(redis)
            log.info(
                "PORTFOLIO  value=%.2f  cash=%.2f  realized_pnl=%+.2f  dd=%.2f%%",
                state.get("total_value", 0),
                state.get("cash", 0),
                state.get("realized_pnl", 0),
                state.get("daily_drawdown_pct", 0),
            )
            for symbol in SYMBOLS:
                pos_raw = await redis.get(f"position:{symbol}")
                if pos_raw:
                    pos   = json.loads(pos_raw)
                    price_raw = await redis.get(f"features:{symbol}:1m:latest")
                    price = json.loads(price_raw).get("close", 0) if price_raw else 0
                    if pos["side"] == "long":
                        upnl = (price - pos["entry"]) * pos["qty"]
                    else:
                        upnl = (pos["entry"] - price) * pos["qty"]
                    log.info(
                        "  OPEN %s %s  qty=%.4f  entry=%.4f  upnl=%+.4f  sl=%.4f",
                        pos["side"].upper(), symbol, pos["qty"],
                        pos["entry"], upnl, pos["stop_loss"],
                    )
        except Exception as exc:
            log.warning("Status loop error: %s", exc)


# ── DB connection ──────────────────────────────────────────────────────────────

async def get_db_pool() -> asyncpg.Pool:
    for attempt in range(30):
        try:
            pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=5)
            log.info("TimescaleDB connected")
            return pool
        except Exception as exc:
            log.warning("DB not ready (%d/30): %s", attempt + 1, exc)
            await asyncio.sleep(3)
    raise RuntimeError("Cannot connect to TimescaleDB")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("═══ Sweep Strategy Service starting [mode=%s] ═══", TRADING_MODE.upper())
    log.info("Symbols: %s  |  Threshold: %.2f  |  sweep_mult: %.2f  |  stop_mult: %.2f",
             SYMBOLS, THRESHOLD, SWEEP_MULT, TIGHT_STOP_MULT)

    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pool  = await get_db_pool()

    model, feat_cols = load_model_and_meta()

    order_manager = PaperOrderManager(redis, pool)
    signal_gen    = SweepSignalGenerator(model, feat_cols, redis, pool, order_manager)
    price_monitor = PriceMonitor(redis, order_manager)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (posix_signal.SIGTERM, posix_signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    log.info("All services initialised — running")

    await asyncio.gather(
        signal_gen.run(stop),
        price_monitor.run(stop),
        status_loop(redis, stop),
    )

    await pool.close()
    await redis.aclose()
    log.info("Sweep Strategy Service stopped.")


if __name__ == "__main__":
    asyncio.run(main())