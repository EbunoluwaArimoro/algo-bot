"""
Redis Schema Documentation — algo-bot
"""

# ── 1. INGESTION DATA ──────────────────────────────────────────────────

# Latest candle per symbol/timeframe (Ingestion writes, Strategy/Sweep reads)
# Key: features:{symbol}:{timeframe}:latest        → JSON hash, TTL 120s
"""
{
    "close": 67420.5, "open": 67000.0, "high": 68000.0, "low": 66500.0,
    "volume": 1542.3, "volume_ratio": 1.8,
    "ema20": 66800.1, "ema50": 65200.3, "ema200": 61000.0,
    "rsi": 58.2, "macd": 320.1, "macd_signal": 280.5, "macd_hist": 39.6,
    "atr": 1200.4, "bb_upper": 69000.0, "bb_lower": 64000.0, "bb_mid": 66500.0,
    "adx": 25.4, "ema_aligned_bull": true, "ema_aligned_bear": false,
    "timestamp": 1714000000
}
"""

# Current market regime (Ingestion writes, Strategy reads)
# Key: regime:{symbol}                             → string, TTL 3600s
# Values: "trending_bull" | "trending_bear" | "ranging" | "high_volatility"


# ── 2. COMMUNICATION QUEUES ────────────────────────────────────────────

# Signal queue (Strategy writes → Risk reads)
# Key: signals:queue                               → Redis List (LPUSH / BRPOP)
"""
[{
    "symbol": "solusdt", "timeframe": "4h", "direction": "long",
    "strategy": "trend_follow", "confidence": 78.4, "regime": "trending_bull",
    "entry_ref": 145.20, "atr": 5.4, "timestamp": 1714000000
}]
"""

# Orders queue (Risk writes → Execution reads)
# Key: orders:queue                                → Redis List (LPUSH / BRPOP)
"""
[{
    "symbol": "solusdt", "direction": "long", "qty": 10.5,
    "entry_ref": 145.20, "stop_loss": 140.00, "take_profit": 155.60,
    "strategy": "trend_follow", "confidence": 78.4, "regime": "trending_bull",
    "risk_pct": 1.0, "timestamp": 1714000000
}]
"""

# ── 3. STATE & RISK MANAGEMENT ─────────────────────────────────────────

# Signal Cooldowns (Risk Service uses this to prevent duplicate entries)
# Key: signal_cooldown:{symbol}:{direction}        → string ("1"), TTL 300s

# Circuit breaker status (Risk writes, Risk/Execution reads)
# Key: circuit_breaker:status                      → "open" | "closed"
# Key: circuit_breaker:reason                      → string (e.g., "daily_drawdown")

# Portfolio state (Risk reads, Execution/Sweep updates)
# Key: portfolio:state                             → JSON hash, no TTL
"""
{
    "total_value": 10420.50, "cash": 8200.00,
    "unrealized_pnl": 420.50, "realized_pnl": 2200.00,
    "daily_pnl": -120.30, "daily_drawdown_pct": 0.8,
    "high_water_mark": 10500.00, "open_positions": 1
}
"""

# ── 4. POSITIONS & SWEEP STRATEGY ──────────────────────────────────────

# Pending Sweep Orders (Sweep Strategy manages these limit orders)
# Key: sweep_order:{symbol}                        → JSON hash, TTL dynamic (expires_at)
"""
{
    "symbol": "solusdt", "limit_buy": 139.5, "limit_sell": 151.5,
    "stop_buy": 138.0, "stop_sell": 153.0, "support": 142.5,
    "resistance": 148.5, "range_mid": 145.5, "range_size": 6.0,
    "vol_prob": 0.85, "regime": "ranging",
    "expires_at": 1714014400, "placed_at": 1714000000, "status": "pending"
}
"""

# Active Open Positions (Sweep Strategy/Execution manages these)
# Key: position:{symbol}                           → JSON hash, no TTL
"""
{
    "symbol": "solusdt", "side": "long", "qty": 10.5, "entry": 140.0,
    "stop_loss": 138.0, "take_profit": 148.5, "trail_stop": 0.0,
    "breakeven_set": false, "partial_done": false,
    "support": 142.5, "resistance": 148.5, "range_mid": 145.5,
    "vol_prob": 0.85, "regime": "ranging", "opened_at": 1714000000
}
"""