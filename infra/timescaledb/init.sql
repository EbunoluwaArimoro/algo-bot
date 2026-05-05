-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── OHLCV tick data (the core time-series table) ──────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,   -- e.g. 'BTC/USDT'
    exchange    TEXT            NOT NULL,   -- e.g. 'binance'
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    timeframe   TEXT            NOT NULL    -- '1m','5m','1h','4h','1d'
);

-- This single call turns ohlcv into a hypertable (auto-partitioned by time)
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

-- Composite index for the most common query: all candles for symbol in range
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
    ON ohlcv (symbol, exchange, time DESC);

-- ── Trades log (every order placed, paper or live) ─────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    time            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    symbol          TEXT            NOT NULL,
    side            TEXT            NOT NULL,   -- 'buy' | 'sell'
    qty             DOUBLE PRECISION NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,           -- NULL until closed
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    strategy        TEXT            NOT NULL,   -- 'trend_follow','mean_rev', etc.
    status          TEXT            NOT NULL DEFAULT 'open',  -- open|closed|cancelled
    pnl             DOUBLE PRECISION,           -- filled on close
    pnl_pct         DOUBLE PRECISION,
    slippage        DOUBLE PRECISION,           -- expected_price - actual_price
    regime          TEXT,                       -- market regime at entry
    signal_score    DOUBLE PRECISION,           -- ML confidence 0-100
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_time    ON trades (time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol  ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status  ON trades (status);

-- ── Portfolio snapshots (equity curve data) ───────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    total_value     DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl    DOUBLE PRECISION NOT NULL DEFAULT 0,
    drawdown_pct    DOUBLE PRECISION NOT NULL DEFAULT 0,
    open_positions  INT             NOT NULL DEFAULT 0
);

SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

-- ── Signals log (every signal generated, even rejected ones) ──────────────
CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    time            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    symbol          TEXT            NOT NULL,
    strategy        TEXT            NOT NULL,
    direction       TEXT            NOT NULL,   -- 'long'|'short'
    confidence      DOUBLE PRECISION,
    regime          TEXT,
    executed        BOOLEAN         NOT NULL DEFAULT FALSE,
    reject_reason   TEXT            -- why risk engine said no
);

-- ── Continuous aggregate: 1h OHLCV rolled up from 1m data ─────────────────
-- (Run this AFTER loading data. Gives free hourly candles from 1m ticks.)
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol, exchange,
    FIRST(open,  time)  AS open,
    MAX(high)           AS high,
    MIN(low)            AS low,
    LAST(close,  time)  AS close,
    SUM(volume)         AS volume
FROM ohlcv
WHERE timeframe = '1m'
GROUP BY bucket, symbol, exchange;