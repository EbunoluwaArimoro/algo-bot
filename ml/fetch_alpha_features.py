"""
ml/fetch_alpha_features.py
───────────────────────────
Fetches non-price alpha features from free APIs and merges them with
the existing OHLCV CSV to produce an enriched feature CSV.

Data sources (all free, no authentication required for historical data):

1. Binance Futures API — funding rates (every 8h)
   Endpoint: GET /fapi/v1/fundingRate
   Why it works: extreme funding = crowded trade = mean reversion likely

2. Binance Futures API — open interest history (1h, 4h snapshots)
   Endpoint: GET /futures/data/openInterestHist
   Why it works: OI change direction > price change = conviction signal

3. Binance Spot API — aggregate trades (buy/sell volume split)
   Endpoint: GET /api/v3/aggTrades or taker buy volume from klines
   Why it works: volume delta separates informed from noise trades

4. CoinGlass API (free tier) — long/short ratio
   Endpoint: GET https://open-api.coinglass.com/public/v2/indicator/long_short_account_ratio
   Why it works: sentiment extremes predict reversals

Note on data availability:
  Binance Futures endpoints return up to 1000 records per call.
  Funding rates: 3 per day → ~1095 records per year → 3285 for 3 years.
  Open interest: available from ~2020 onward.
  All data is aligned to 4h candle timestamps before merging.

Usage:
    python ml/fetch_alpha_features.py --symbol BTCUSDT --days 1100
    python ml/fetch_alpha_features.py --symbol ETHUSDT --days 1100
    python ml/fetch_alpha_features.py --symbol BTCUSDT --days 1100 --no-coinglass
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.alpha")

BINANCE_FUTURES = "https://fapi.binance.com"
BINANCE_SPOT    = "https://api.binance.com"
COINGLASS_BASE  = "https://open-api.coinglass.com/public/v2"

SLEEP_S    = 0.25   # between API calls — stay well under rate limits
OUTPUT_DIR = "ml/data"


# ── Utilities ─────────────────────────────────────────────────────────────────

def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def safe_get(url: str, params: dict, label: str) -> list | dict | None:
    """GET with retry on rate limit."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                log.warning("%s rate limited — sleeping 10s", label)
                time.sleep(10)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            log.warning("%s attempt %d failed: %s", label, attempt + 1, exc)
            time.sleep(3)
    log.error("%s failed after 3 attempts", label)
    return None


# ── 1. Funding Rates ──────────────────────────────────────────────────────────

def fetch_funding_rates(symbol: str, days: int) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance Futures.
    Returns columns: time, funding_rate

    Funding rate is paid every 8 hours. We'll resample to 4h by forward-filling
    (each 4h bar carries the most recent funding rate in effect).

    Interpretation:
      funding_rate > 0.01%  → longs paying shorts → market is long-biased
      funding_rate > 0.05%  → extreme long crowding → short signal
      funding_rate < -0.01% → shorts paying longs → market is short-biased
      funding_rate < -0.03% → extreme short crowding → long signal
    """
    log.info("Fetching funding rates for %s (%d days)…", symbol, days)

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = dt_to_ms(since)
    records = []

    while True:
        data = safe_get(
            f"{BINANCE_FUTURES}/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": since_ms, "limit": 1000},
            "funding_rate",
        )
        if not data:
            break

        for item in data:
            records.append({
                "time":         ms_to_dt(item["fundingTime"]),
                "funding_rate": float(item["fundingRate"]),
            })

        if len(data) < 1000:
            break

        since_ms = data[-1]["fundingTime"] + 1
        time.sleep(SLEEP_S)

    if not records:
        log.warning("No funding rate data returned for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    log.info("  Fetched %d funding rate records (%s → %s)",
             len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))

    # Compute rolling funding features
    df["funding_rate_8h"]      = df["funding_rate"]
    df["funding_rate_24h_sum"] = df["funding_rate"].rolling(3).sum()   # 3 × 8h = 24h
    df["funding_rate_3d_sum"]  = df["funding_rate"].rolling(9).sum()   # 9 × 8h = 3d

    # Z-score: how extreme is current funding vs recent history?
    roll_std = df["funding_rate"].rolling(30).std()
    roll_mean= df["funding_rate"].rolling(30).mean()
    df["funding_zscore"] = (df["funding_rate"] - roll_mean) / (roll_std + 1e-10)

    # Flags: extreme crowding levels
    df["funding_extreme_long"]  = (df["funding_rate"] > 0.0005).astype(int)  # > 0.05%
    df["funding_extreme_short"] = (df["funding_rate"] < -0.0003).astype(int)

    return df[["time", "funding_rate_8h", "funding_rate_24h_sum",
               "funding_rate_3d_sum", "funding_zscore",
               "funding_extreme_long", "funding_extreme_short"]]


# ── 2. Open Interest ──────────────────────────────────────────────────────────

def fetch_open_interest(symbol: str, days: int, period: str = "4h") -> pd.DataFrame:
    """
    Fetch open interest history from Binance Futures.
    Returns columns: time, oi, oi_change_pct, oi_price_divergence

    Available periods: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d

    Interpretation:
      oi_change_pct > 0 AND price rising  → new longs entering → conviction move
      oi_change_pct > 0 AND price falling → new shorts entering → conviction drop
      oi_change_pct < 0 AND price rising  → short covering → weak rally
      oi_change_pct < 0 AND price falling → long liquidation → potential bottom
    """
    log.info("Fetching open interest for %s (%d days, %s)…", symbol, days, period)

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = dt_to_ms(since)
    records = []

    while True:
        data = safe_get(
            f"{BINANCE_FUTURES}/futures/data/openInterestHist",
            {"symbol": symbol, "period": period,
             "startTime": since_ms, "limit": 500},
            "open_interest",
        )
        if not data:
            break

        for item in data:
            records.append({
                "time": ms_to_dt(item["timestamp"]),
                "oi":   float(item["sumOpenInterest"]),
            })

        if len(data) < 500:
            break

        since_ms = data[-1]["timestamp"] + 1
        time.sleep(SLEEP_S)

    if not records:
        log.warning("No open interest data returned for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    log.info("  Fetched %d OI records (%s → %s)",
             len(df), str(df["time"].iloc[0].date()), str(df["time"].iloc[-1].date()))

    # Features derived from OI
    df["oi_change_pct"] = df["oi"].pct_change() * 100
    df["oi_change_3bar"]= df["oi"].pct_change(3) * 100   # 12h on 4h bars

    # Rolling z-score of OI change
    roll_std  = df["oi_change_pct"].rolling(30).std()
    roll_mean = df["oi_change_pct"].rolling(30).mean()
    df["oi_zscore"] = (df["oi_change_pct"] - roll_mean) / (roll_std + 1e-10)

    # OI acceleration: is OI change speeding up or slowing down?
    df["oi_acceleration"] = df["oi_change_pct"].diff()

    return df[["time", "oi", "oi_change_pct", "oi_change_3bar",
               "oi_zscore", "oi_acceleration"]]


# ── 3. Taker Buy Volume (Volume Delta proxy) ──────────────────────────────────

def fetch_taker_buy_volume(symbol: str, days: int, interval: str = "4h") -> pd.DataFrame:
    """
    Fetch kline data including taker buy base asset volume.
    Volume delta = taker_buy_volume - taker_sell_volume
    taker_sell_volume = total_volume - taker_buy_volume

    Taker buyers are aggressive — they cross the spread to buy immediately.
    High taker buy ratio on an up candle = strong conviction.
    High taker buy ratio on a down candle = buyers trying to hold support.

    This is the closest we can get to order flow without a Level 2 feed.

    Binance kline endpoint columns:
    0:open_time 1:open 2:high 3:low 4:close 5:volume 6:close_time
    7:quote_asset_volume 8:number_of_trades 9:taker_buy_base_volume
    10:taker_buy_quote_volume 11:ignore
    """
    log.info("Fetching taker buy volume for %s (%d days, %s)…", symbol, days, interval)

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = dt_to_ms(since)
    records  = []

    # Try futures endpoint first (has more data for perp symbols)
    endpoints = [
        f"{BINANCE_FUTURES}/fapi/v1/klines",
        f"{BINANCE_SPOT}/api/v3/klines",
    ]

    for endpoint in endpoints:
        temp_since = since_ms
        temp_records = []
        success = True

        while True:
            data = safe_get(
                endpoint,
                {"symbol": symbol, "interval": interval,
                 "startTime": temp_since, "limit": 1000},
                f"klines ({endpoint})",
            )
            if not data or isinstance(data, dict):
                success = False
                break

            for k in data:
                total_vol    = float(k[5])
                buy_vol      = float(k[9])
                sell_vol     = total_vol - buy_vol
                delta        = buy_vol - sell_vol
                buy_ratio    = buy_vol / total_vol if total_vol > 0 else 0.5

                temp_records.append({
                    "time":           ms_to_dt(k[0]),
                    "taker_buy_vol":  buy_vol,
                    "taker_sell_vol": sell_vol,
                    "volume_delta":   delta,
                    "buy_ratio":      buy_ratio,
                    "n_trades":       int(k[8]),
                })

            if len(data) < 1000:
                break

            temp_since = data[-1][6] + 1
            time.sleep(SLEEP_S)

        if success and temp_records:
            records = temp_records
            log.info("  Used endpoint: %s", endpoint)
            break

    if not records:
        log.warning("No taker volume data returned for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    log.info("  Fetched %d volume delta records", len(df))

    # Rolling features
    df["buy_ratio_ma10"]   = df["buy_ratio"].rolling(10).mean()
    df["buy_ratio_zscore"] = (
        (df["buy_ratio"] - df["buy_ratio"].rolling(30).mean()) /
        (df["buy_ratio"].rolling(30).std() + 1e-10)
    )

    # Cumulative delta over 3 bars (12h on 4h chart)
    df["cum_delta_3bar"] = df["volume_delta"].rolling(3).sum()

    # Delta divergence: price direction vs volume delta direction
    # 1 = agreement (price up + positive delta), -1 = divergence
    df["delta_ma"] = df["volume_delta"].rolling(5).mean()

    return df[["time", "taker_buy_vol", "taker_sell_vol", "volume_delta",
               "buy_ratio", "buy_ratio_ma10", "buy_ratio_zscore",
               "cum_delta_3bar", "delta_ma", "n_trades"]]


# ── 4. Long/Short Ratio (CoinGlass) ───────────────────────────────────────────

def fetch_long_short_ratio(symbol: str, days: int, period: str = "4h") -> pd.DataFrame:
    """
    Fetch long/short account ratio from CoinGlass free API.
    No API key required for the public endpoints.

    Interpretation:
      ls_ratio > 1.5 → far more longs than shorts → contrarian short signal
      ls_ratio < 0.7 → far more shorts than longs → contrarian long signal
      ls_ratio near 1.0 → balanced positioning → neutral

    Note: CoinGlass free tier has rate limits (~5 req/min) and limited history.
    If it fails, we skip it gracefully — the other features carry enough signal.
    """
    log.info("Fetching long/short ratio from CoinGlass for %s…", symbol)

    # CoinGlass symbol format: 'BTC' not 'BTCUSDT'
    cg_symbol = symbol.replace("USDT", "").replace("PERP", "")

    # CoinGlass free public endpoint (no key needed for basic data)
    url = f"{COINGLASS_BASE}/indicator/long_short_account_ratio"
    data = safe_get(url, {
        "ex": "Binance",
        "pair": f"{cg_symbol}USDT_PERP",
        "interval": period,
        "limit": 500,
    }, "coinglass_ls_ratio")

    if not data or not isinstance(data, dict) or "data" not in data:
        log.warning("CoinGlass data unavailable — using Binance L/S ratio instead")
        return fetch_binance_ls_ratio(symbol, days, period)

    records = []
    for item in data.get("data", []):
        try:
            records.append({
                "time":     ms_to_dt(item["createTime"]),
                "ls_ratio": float(item["longShortRatio"]),
                "long_pct": float(item.get("longAccount", 0.5)),
                "short_pct":float(item.get("shortAccount", 0.5)),
            })
        except (KeyError, ValueError):
            continue

    if not records:
        log.warning("No CoinGlass data parsed — falling back to Binance")
        return fetch_binance_ls_ratio(symbol, days, period)

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    df = _add_ls_features(df)
    log.info("  Fetched %d L/S ratio records from CoinGlass", len(df))
    return df


def fetch_binance_ls_ratio(symbol: str, days: int, period: str = "4h") -> pd.DataFrame:
    """
    Fallback: fetch global long/short account ratio from Binance Futures.
    This is the ratio of accounts that are net long vs net short.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = dt_to_ms(since)
    records  = []

    while True:
        data = safe_get(
            f"{BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": period,
             "startTime": since_ms, "limit": 500},
            "binance_ls_ratio",
        )
        if not data:
            break

        for item in data:
            try:
                records.append({
                    "time":     ms_to_dt(item["timestamp"]),
                    "ls_ratio": float(item["longShortRatio"]),
                    "long_pct": float(item["longAccount"]),
                    "short_pct":float(item["shortAccount"]),
                })
            except (KeyError, ValueError):
                continue

        if len(data) < 500:
            break

        since_ms = data[-1]["timestamp"] + 1
        time.sleep(SLEEP_S)

    if not records:
        log.warning("No L/S ratio data from Binance either — skipping")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    df = _add_ls_features(df)
    log.info("  Fetched %d L/S ratio records from Binance", len(df))
    return df


def _add_ls_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw long/short ratio."""
    df["ls_ratio_ma10"]   = df["ls_ratio"].rolling(10).mean()
    df["ls_ratio_zscore"] = (
        (df["ls_ratio"] - df["ls_ratio"].rolling(30).mean()) /
        (df["ls_ratio"].rolling(30).std() + 1e-10)
    )
    df["ls_ratio_change"] = df["ls_ratio"].pct_change()

    # Extreme sentiment flags
    df["crowd_extreme_long"]  = (df["ls_ratio"] > 1.8).astype(int)
    df["crowd_extreme_short"] = (df["ls_ratio"] < 0.6).astype(int)

    return df[["time", "ls_ratio", "ls_ratio_ma10", "ls_ratio_zscore",
               "ls_ratio_change", "crowd_extreme_long", "crowd_extreme_short"]]


# ── Merge pipeline ────────────────────────────────────────────────────────────

def align_to_ohlcv(alpha_df: pd.DataFrame, ohlcv_df: pd.DataFrame,
                   name: str) -> pd.DataFrame:
    """
    Merge alpha features onto OHLCV timestamps using asof merge.
    This forward-fills alpha data to the nearest preceding OHLCV candle —
    no look-ahead bias.
    """
    if alpha_df.empty:
        log.warning("Skipping empty alpha dataframe: %s", name)
        return ohlcv_df

    # Ensure UTC
    if alpha_df["time"].dt.tz is None:
        alpha_df["time"] = alpha_df["time"].dt.tz_localize("UTC")
    if ohlcv_df["time"].dt.tz is None:
        ohlcv_df["time"] = ohlcv_df["time"].dt.tz_localize("UTC")

    alpha_df = alpha_df.sort_values("time")
    ohlcv_df = ohlcv_df.sort_values("time")

    merged = pd.merge_asof(
        ohlcv_df, alpha_df,
        on="time",
        direction="backward",   # use the most recent alpha value before this candle
        tolerance=pd.Timedelta("8h"),  # don't use stale data older than 8h
    )

    new_cols = [c for c in alpha_df.columns if c != "time"]
    filled   = merged[new_cols].notna().all(axis=1).sum()
    log.info("  %s: merged %d/%d bars (%.0f%% coverage)",
             name, filled, len(merged), filled/len(merged)*100)

    return merged


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-feature interactions that the model can't discover on its own.
    These encode domain knowledge about how features combine.
    """
    # 1. Funding + OI divergence: funding extreme but OI declining = exhaustion
    if "funding_rate_8h" in df.columns and "oi_change_pct" in df.columns:
        df["funding_oi_diverge"] = (
            (df["funding_rate_8h"].abs() > 0.0003) &
            (df["oi_change_pct"] < 0)
        ).astype(int)

    # 2. Volume delta + price direction agreement
    if "volume_delta" in df.columns and "close" in df.columns:
        price_up   = df["close"] > df["close"].shift(1)
        delta_pos  = df["volume_delta"] > 0
        df["delta_price_agree"]  = (price_up == delta_pos).astype(int)
        df["delta_price_diverge"]= (price_up != delta_pos).astype(int)

    # 3. Crowd sentiment vs funding agreement
    if "ls_ratio" in df.columns and "funding_rate_8h" in df.columns:
        df["sentiment_agreement"] = (
            (df["ls_ratio"] > 1.2) & (df["funding_rate_8h"] > 0.0002)
        ).astype(int)

    # 4. OI + volume delta momentum confirmation
    if "oi_change_pct" in df.columns and "cum_delta_3bar" in df.columns:
        df["oi_delta_confirm"] = (
            np.sign(df["oi_change_pct"]) == np.sign(df["cum_delta_3bar"])
        ).astype(int)

    # 5. Funding rate mean reversion signal
    # When funding z-score > 2, market is extremely one-sided
    if "funding_zscore" in df.columns:
        df["funding_reversal_long"]  = (df["funding_zscore"] > 2.0).astype(int)
        df["funding_reversal_short"] = (df["funding_zscore"] < -2.0).astype(int)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fetch alpha features for ML training")
    p.add_argument("--symbol",      default="BTCUSDT",
                   help="Binance futures symbol e.g. BTCUSDT, ETHUSDT")
    p.add_argument("--days",        type=int, default=1100,
                   help="Days of history to fetch (default 1100 ≈ 3 years)")
    p.add_argument("--ohlcv-csv",   default=None,
                   help="Path to existing OHLCV CSV to merge with (auto-detected if omitted)")
    p.add_argument("--no-coinglass",action="store_true",
                   help="Skip CoinGlass API (use Binance L/S ratio instead)")
    p.add_argument("--output",      default=None,
                   help="Output CSV path (auto-named if omitted)")
    args = p.parse_args()

    symbol = args.symbol.upper()

    # Auto-detect OHLCV CSV
    if args.ohlcv_csv:
        ohlcv_path = args.ohlcv_csv
    else:
        slug = symbol.replace("USDT", "_USDT")
        candidates = [
            f"backtest/data/{slug}_4h.csv",
            f"backtest/data/{symbol}_4h.csv",
        ]
        ohlcv_path = next((c for c in candidates if os.path.exists(c)), None)
        if not ohlcv_path:
            log.error("Could not find OHLCV CSV. Specify with --ohlcv-csv")
            import sys; sys.exit(1)

    log.info("Loading OHLCV from %s", ohlcv_path)
    ohlcv_df = pd.read_csv(ohlcv_path, parse_dates=["time"])
    if ohlcv_df["time"].dt.tz is None:
        ohlcv_df["time"] = ohlcv_df["time"].dt.tz_localize("UTC")
    ohlcv_df = ohlcv_df.sort_values("time").reset_index(drop=True)
    log.info("Loaded %d OHLCV bars", len(ohlcv_df))

    # Fetch all alpha features
    print(f"\n{'═'*55}")
    print(f"  Fetching alpha features for {symbol}")
    print(f"{'═'*55}")

    funding_df = fetch_funding_rates(symbol, args.days)
    oi_df      = fetch_open_interest(symbol, args.days)
    vol_df     = fetch_taker_buy_volume(symbol, args.days)

    if args.no_coinglass:
        ls_df = fetch_binance_ls_ratio(symbol, args.days)
    else:
        ls_df = fetch_long_short_ratio(symbol, args.days)

    # Merge onto OHLCV timestamps
    print(f"\n  Merging to OHLCV timestamps (forward-fill, no look-ahead)…")
    merged = ohlcv_df.copy()
    merged = align_to_ohlcv(funding_df, merged, "funding_rates")
    merged = align_to_ohlcv(oi_df,      merged, "open_interest")
    merged = align_to_ohlcv(vol_df,     merged, "volume_delta")
    merged = align_to_ohlcv(ls_df,      merged, "ls_ratio")

    # Compute interaction features
    merged = compute_interaction_features(merged)

    # Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        slug = symbol.replace("USDT", "_USDT")
        out_path = os.path.join(OUTPUT_DIR, f"{slug}_4h_alpha.csv")

    merged.to_csv(out_path, index=False)

    # Summary
    alpha_cols = [c for c in merged.columns
                  if c not in ["time","open","high","low","close","volume"]]
    coverage   = merged[alpha_cols].notna().mean() * 100

    print(f"\n{'═'*55}")
    print(f"  Alpha feature summary")
    print(f"{'═'*55}")
    print(f"  Output file:   {out_path}")
    print(f"  Total rows:    {len(merged):,}")
    print(f"  Alpha columns: {len(alpha_cols)}")
    print()
    print(f"  Coverage (% of bars with data):")

    groups = {
        "Funding rates":    [c for c in alpha_cols if "funding" in c],
        "Open interest":    [c for c in alpha_cols if "oi" in c],
        "Volume delta":     [c for c in alpha_cols if any(x in c for x in ["delta","buy_ratio","n_trades","taker"])],
        "L/S ratio":        [c for c in alpha_cols if "ls_ratio" in c or "crowd" in c],
        "Interactions":     [c for c in alpha_cols if any(x in c for x in ["diverge","agree","confirm","reversal","sentiment"])],
    }

    for group, cols in groups.items():
        if cols:
            avg_cov = coverage[cols].mean()
            print(f"    {group:20}  {avg_cov:5.1f}%  ({len(cols)} features)")

    print(f"\n{'═'*55}")
    print(f"  Next step:")
    print(f"    python ml/generate_training_data.py --csv {out_path} --use-alpha")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()