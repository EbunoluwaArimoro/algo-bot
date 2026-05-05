# algo-bot Makefile
# Usage: make <target>

.PHONY: help up down logs db redis shell-db history backtest reset clean

help:
	@echo ""
	@echo "  algo-bot dev commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make up          Start all services"
	@echo "  make down        Stop all services"
	@echo "  make logs        Follow all service logs"
	@echo "  make logs-s SVC= Follow logs for one service (e.g. make logs-s SVC=ingestion)"
	@echo "  make db          Open psql shell in TimescaleDB"
	@echo "  make redis       Open redis-cli"
	@echo "  make history     Download historical data for all symbols"
	@echo "  make backtest    Run backtesting engine"
	@echo "  make stats       Print DB stats (row counts, date ranges)"
	@echo "  make reset       Wipe all Docker volumes and restart fresh"
	@echo "  make clean       Remove all containers, networks, volumes"
	@echo ""

# ── Infrastructure ────────────────────────────────────────────────────────────

up:
	@cp -n .env.example .env 2>/dev/null || true
	docker compose up --build -d
	@echo "Services started. Logs: make logs"

down:
	docker compose down

logs:
	docker compose logs -f

logs-s:
	docker compose logs -f $(SVC)

# ── Database ──────────────────────────────────────────────────────────────────

db:
	docker compose exec timescaledb psql -U $${DB_USER:-bot} -d botdb

redis:
	docker compose exec redis redis-cli

stats:
	docker compose exec timescaledb psql -U $${DB_USER:-bot} -d botdb -c \
	  "SELECT symbol, timeframe, COUNT(*) AS rows, MIN(time)::DATE AS earliest, MAX(time)::DATE AS latest \
	   FROM ohlcv GROUP BY symbol, timeframe ORDER BY symbol, timeframe;"

# ── Data & Backtest ───────────────────────────────────────────────────────────

history:
	pip install -q ccxt asyncpg pandas python-dotenv
	python scripts/load_history.py --all --timeframe 1h --years 3
	python scripts/load_history.py --all --timeframe 1m --years 1

backtest:
	python backtest/run_backtest.py $(ARGS)

# ── Monitoring ────────────────────────────────────────────────────────────────

portfolio:
	docker compose exec redis redis-cli GET portfolio:state | python3 -m json.tool

positions:
	@docker compose exec redis redis-cli KEYS "position:*" | \
	  xargs -I{} docker compose exec redis redis-cli GET {} 2>/dev/null | \
	  python3 -m json.tool 2>/dev/null || echo "No open positions"

signals:
	docker compose exec redis redis-cli LLEN signals:queue

regime:
	@for sym in btcusdt ethusdt solusdt; do \
	  echo -n "$$sym: "; \
	  docker compose exec redis redis-cli GET "regime:$$sym"; \
	done

# ── Reset ─────────────────────────────────────────────────────────────────────

reset:
	docker compose down -v
	docker compose up --build -d
	@echo "Fresh restart complete — run 'make history' to reload data"

clean:
	docker compose down -v --remove-orphans
	@echo "All containers and volumes removed"