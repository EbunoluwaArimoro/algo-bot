"""
Strategy Service — algo-bot
────────────────────────────
Responsibilities:
  1. Poll Redis feature store every N seconds per (symbol, timeframe)
  2. Run each strategy against the latest features
  3. Score and combine signals using regime-based weighting
  4. Push qualified signals onto the signals:queue Redis list
     (Risk service consumes from the other end)

Strategies implemented here:
  - TrendFollow   : EMA alignment + RSI filter + MACD confirmation
  - MeanReversion : Bollinger Band extremes + RSI divergence
  - Breakout      : volume spike + price above recent high
"""

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import asdict, dataclass
from typing import Literal

import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("strategy")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Symbols and timeframes to evaluate (must match ingestion config)
SYMBOLS:    list[str] = os.getenv("SYMBOLS",    "btcusdt,ethusdt,solusdt").split(",")
TIMEFRAMES: list[str] = os.getenv("TIMEFRAMES", "1m,5m,1h").split(",")

# How often (seconds) to poll Redis for new features per symbol
POLL_INTERVAL_S = int(os.getenv("STRATEGY_POLL_S", "5"))

# Minimum ML-style confidence score (0–100) to emit a signal
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "55.0"))

# Max signals sitting in queue at once (prevents backlog during outage)
MAX_QUEUE_DEPTH = 50

Direction = Literal["long", "short", "hold"]


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    symbol:     str
    timeframe:  str
    direction:  Direction
    strategy:   str
    confidence: float          # 0–100
    regime:     str
    entry_ref:  float          # reference price at signal time
    atr:        float          # for stop-loss sizing in risk service
    timestamp:  int            # unix epoch


# ── Strategy implementations ──────────────────────────────────────────────────

class TrendFollowStrategy:
    """
    Multi-EMA trend with RSI + MACD confirmation.

    Long entry conditions (all must pass):
      • EMA20 > EMA50 > EMA200  (bull alignment)
      • 40 < RSI < 70           (not overbought)
      • MACD histogram positive  (momentum confirming)
      • Volume ratio > 1.0       (above-average activity)

    Short entry — mirror conditions.
    Best regime: trending_bull / trending_bear
    """
    name = "trend_follow"

    def evaluate(self, f: dict, regime: str) -> Signal | None:
        rsi          = f.get("rsi")
        ema_bull     = f.get("ema_aligned_bull", False)
        ema_bear     = f.get("ema_aligned_bear", False)
        macd_hist    = f.get("macd_hist")
        volume_ratio = f.get("volume_ratio", 0)
        close        = f.get("close", 0)
        atr          = f.get("atr", 0)

        if None in (rsi, macd_hist):
            return None

        # Regime weight: boost confidence in trending regimes
        regime_bonus = 15 if regime in ("trending_bull", "trending_bear") else 0

        # ── Long
        if (ema_bull and
            40 < rsi < 70 and
            macd_hist > 0 and
            volume_ratio > 1.0):
            confidence = 50 + regime_bonus + min(20, (rsi - 40) * 0.5)
            if confidence >= MIN_CONFIDENCE:
                return Signal(
                    symbol=f["_symbol"], timeframe=f["_timeframe"],
                    direction="long",   strategy=self.name,
                    confidence=round(confidence, 1),
                    regime=regime,      entry_ref=close,
                    atr=atr,            timestamp=int(time.time()),
                )

        # ── Short
        if (ema_bear and
            30 < rsi < 60 and
            macd_hist < 0 and
            volume_ratio > 1.0):
            confidence = 50 + regime_bonus + min(20, (60 - rsi) * 0.5)
            if confidence >= MIN_CONFIDENCE:
                return Signal(
                    symbol=f["_symbol"], timeframe=f["_timeframe"],
                    direction="short",  strategy=self.name,
                    confidence=round(confidence, 1),
                    regime=regime,      entry_ref=close,
                    atr=atr,            timestamp=int(time.time()),
                )

        return None


class MeanReversionStrategy:
    """
    Bollinger Band extremes with RSI divergence.

    Long: price below lower BB + RSI < 30 (oversold)
    Short: price above upper BB + RSI > 70 (overbought)
    Best regime: ranging
    Suppressed in: trending_bull (shorts), trending_bear (longs)
    """
    name = "mean_reversion"

    def evaluate(self, f: dict, regime: str) -> Signal | None:
        close    = f.get("close", 0)
        bb_upper = f.get("bb_upper")
        bb_lower = f.get("bb_lower")
        bb_mid   = f.get("bb_mid")
        rsi      = f.get("rsi")
        atr      = f.get("atr", 0)

        if None in (bb_upper, bb_lower, bb_mid, rsi):
            return None

        # Suppress counter-trend MR in strong trends
        if regime == "trending_bull" and close > bb_mid:
            return None
        if regime == "trending_bear" and close < bb_mid:
            return None

        regime_bonus = 15 if regime == "ranging" else 0

        # ── Long (oversold bounce)
        if close < bb_lower and rsi < 32:
            # Deeper into oversold = higher confidence
            depth = (bb_lower - close) / (atr + 1e-9)
            confidence = 48 + regime_bonus + min(20, depth * 5)
            if confidence >= MIN_CONFIDENCE:
                return Signal(
                    symbol=f["_symbol"], timeframe=f["_timeframe"],
                    direction="long",   strategy=self.name,
                    confidence=round(confidence, 1),
                    regime=regime,      entry_ref=close,
                    atr=atr,            timestamp=int(time.time()),
                )

        # ── Short (overbought rejection)
        if close > bb_upper and rsi > 68:
            depth = (close - bb_upper) / (atr + 1e-9)
            confidence = 48 + regime_bonus + min(20, depth * 5)
            if confidence >= MIN_CONFIDENCE:
                return Signal(
                    symbol=f["_symbol"], timeframe=f["_timeframe"],
                    direction="short",  strategy=self.name,
                    confidence=round(confidence, 1),
                    regime=regime,      entry_ref=close,
                    atr=atr,            timestamp=int(time.time()),
                )

        return None


class BreakoutStrategy:
    """
    Volume-confirmed breakout above recent high / below recent low.

    Long:  close > recent 20-period high AND volume_ratio > 2.0
    Short: close < recent 20-period low  AND volume_ratio > 2.0
    Best regime: trending_bull (longs), high_volatility (avoid)
    """
    name = "breakout"

    def evaluate(self, f: dict, regime: str) -> Signal | None:
        if regime == "high_volatility":
            return None  # too noisy for breakout during news spikes

        close        = f.get("close", 0)
        high_20      = f.get("high_20")   # added by ingestion if available
        low_20       = f.get("low_20")
        volume_ratio = f.get("volume_ratio", 0)
        atr          = f.get("atr", 0)
        rsi          = f.get("rsi") or 50

        # Fallback: use BB upper/lower as proxy for 20-period range
        if high_20 is None:
            high_20 = f.get("bb_upper")
        if low_20 is None:
            low_20  = f.get("bb_lower")

        if None in (high_20, low_20):
            return None

        regime_bonus = 10 if regime in ("trending_bull", "trending_bear") else 0

        # ── Long breakout
        if close > high_20 and volume_ratio > 2.0 and rsi < 80:
            confidence = 52 + regime_bonus + min(15, (volume_ratio - 2.0) * 5)
            if confidence >= MIN_CONFIDENCE:
                return Signal(
                    symbol=f["_symbol"], timeframe=f["_timeframe"],
                    direction="long",   strategy=self.name,
                    confidence=round(confidence, 1),
                    regime=regime,      entry_ref=close,
                    atr=atr,            timestamp=int(time.time()),
                )

        # ── Short breakdown
        if close < low_20 and volume_ratio > 2.0 and rsi > 20:
            confidence = 52 + regime_bonus + min(15, (volume_ratio - 2.0) * 5)
            if confidence >= MIN_CONFIDENCE:
                return Signal(
                    symbol=f["_symbol"], timeframe=f["_timeframe"],
                    direction="short",  strategy=self.name,
                    confidence=round(confidence, 1),
                    regime=regime,      entry_ref=close,
                    atr=atr,            timestamp=int(time.time()),
                )

        return None


# ── Strategy orchestrator ──────────────────────────────────────────────────────

STRATEGIES = [
    TrendFollowStrategy(),
    MeanReversionStrategy(),
    BreakoutStrategy(),
]

# Regime → which strategies are active + weight multiplier
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "trending_bull":    {"trend_follow": 1.0, "breakout": 0.8, "mean_reversion": 0.4},
    "trending_bear":    {"trend_follow": 1.0, "breakout": 0.7, "mean_reversion": 0.4},
    "ranging":          {"trend_follow": 0.4, "breakout": 0.3, "mean_reversion": 1.0},
    "high_volatility":  {"trend_follow": 0.5, "breakout": 0.0, "mean_reversion": 0.5},
}


def apply_regime_weight(signal: Signal, regime: str) -> Signal:
    weights = REGIME_WEIGHTS.get(regime, {})
    w = weights.get(signal.strategy, 1.0)
    signal.confidence = round(signal.confidence * w, 1)
    return signal


def deduplicate(signals: list[Signal]) -> list[Signal]:
    """
    If multiple strategies agree on direction for the same symbol,
    keep the highest-confidence signal and boost it slightly.
    """
    best: dict[str, Signal] = {}
    for s in signals:
        key = f"{s.symbol}_{s.direction}"
        if key not in best or s.confidence > best[key].confidence:
            best[key] = s

    # Agreement bonus: if 2+ strategies agree, +5 confidence
    direction_counts: dict[str, int] = {}
    for s in signals:
        k = f"{s.symbol}_{s.direction}"
        direction_counts[k] = direction_counts.get(k, 0) + 1

    result = []
    for key, sig in best.items():
        if direction_counts[key] > 1:
            sig.confidence = min(100, sig.confidence + 5)
        result.append(sig)

    return result


# ── Redis helpers ─────────────────────────────────────────────────────────────

async def read_features(redis: aioredis.Redis, symbol: str, timeframe: str) -> dict | None:
    key = f"features:{symbol}:{timeframe}:latest"
    raw = await redis.get(key)
    if not raw:
        return None
    f = json.loads(raw)
    f["_symbol"]    = symbol
    f["_timeframe"] = timeframe
    return f


async def read_regime(redis: aioredis.Redis, symbol: str) -> str:
    regime = await redis.get(f"regime:{symbol}")
    return regime or "ranging"   # default safe assumption


async def push_signal(redis: aioredis.Redis, signal: Signal) -> bool:
    """Push signal to queue. Returns False if queue is full (backpressure)."""
    depth = await redis.llen("signals:queue")
    if depth >= MAX_QUEUE_DEPTH:
        log.warning("Signal queue full (%d) — dropping signal %s %s",
                    depth, signal.symbol, signal.direction)
        return False
    await redis.lpush("signals:queue", json.dumps(asdict(signal)))
    return True


# ── Main evaluation loop ───────────────────────────────────────────────────────

async def evaluate_once(redis: aioredis.Redis) -> None:
    """Run all strategies for all symbol/timeframe combinations once."""
    all_signals: list[Signal] = []

    for symbol in SYMBOLS:
        regime = await read_regime(redis, symbol)

        for timeframe in TIMEFRAMES:
            features = await read_features(redis, symbol, timeframe)
            if not features:
                continue

            for strategy in STRATEGIES:
                try:
                    signal = strategy.evaluate(features, regime)
                except Exception as exc:
                    log.error("Strategy %s error on %s/%s: %s",
                              strategy.name, symbol, timeframe, exc)
                    continue

                if signal and signal.confidence >= MIN_CONFIDENCE:
                    signal = apply_regime_weight(signal, regime)
                    if signal.confidence >= MIN_CONFIDENCE:
                        all_signals.append(signal)

    # Deduplicate and emit
    final_signals = deduplicate(all_signals)
    for sig in final_signals:
        pushed = await push_signal(redis, sig)
        if pushed:
            log.info(
                "Signal → %-9s %-6s %-5s  strategy=%-15s conf=%.1f  regime=%s",
                sig.symbol, sig.timeframe, sig.direction.upper(),
                sig.strategy, sig.confidence, sig.regime,
            )


async def evaluation_loop(redis: aioredis.Redis, stop: asyncio.Event) -> None:
    log.info("Strategy loop started — polling every %ds", POLL_INTERVAL_S)
    while not stop.is_set():
        try:
            await evaluate_once(redis)
        except Exception as exc:
            log.exception("Evaluation error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL_S)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("═══ Strategy Service starting ═══")
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)

    stop  = asyncio.Event()
    loop  = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await evaluation_loop(redis, stop)

    await redis.aclose()
    log.info("Strategy Service stopped.")


if __name__ == "__main__":
    import signal
    asyncio.run(main())