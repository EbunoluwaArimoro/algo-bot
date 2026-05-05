"""
# Latest candle per symbol/timeframe
features:{symbol}:{timeframe}:latest       → JSON hash, TTL 120s
  { "close": 67420.5, "rsi": 58.2, "ema20": 66800.1, "ema50": 65200.3,
    "atr": 1200.4, "bb_upper": 69000.0, "bb_lower": 64000.0,
    "macd": 320.1, "macd_signal": 280.5, "volume_ratio": 1.8,
    "timestamp": 1714000000 }

# Current regime
regime:{symbol}                            → string, TTL 3600s
  "trending_bull" | "trending_bear" | "ranging" | "high_volatility"

# Open positions (execution service owns writes)
position:{symbol}                          → JSON hash, no TTL
  { "side": "long", "qty": 0.05, "entry": 67000.0,
    "stop_loss": 65800.0, "strategy": "trend_follow",
    "opened_at": 1714000000 }

# Portfolio state
portfolio:state                            → JSON hash, no TTL
  { "total_value": 15420.50, "cash": 8200.00,
    "unrealized_pnl": 420.50, "daily_pnl": -120.30,
    "drawdown_pct": 0.8, "daily_drawdown_pct": 0.8 }

# Circuit breaker status
circuit_breaker:status                     → "open" | "closed", TTL irrelevant
  (if "open" → execution service rejects ALL orders)

# Signal queue (strategy → risk → execution pipeline)
signals:queue                              → Redis List (LPUSH / BRPOP)
  [{ "symbol": "BTC/USDT", "direction": "long", "strategy": "trend_follow",
     "confidence": 78.4, "timestamp": 1714000000 }]