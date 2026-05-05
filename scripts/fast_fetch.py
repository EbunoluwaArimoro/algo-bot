import requests
import pandas as pd
import time
import os

print("Bypassing firewalls and fetching ETH data directly via REST...")
url = "https://api.binance.com/api/v3/klines"
end_time = int(time.time() * 1000)
all_data = []

# 7 loops * 1000 = 7000 candles (approx 3 years of 4h data)
for i in range(7):
    params = {"symbol": "ETHUSDT", "interval": "4h", "limit": 1000, "endTime": end_time}
    try:
        res = requests.get(url, params=params, timeout=15)
        data = res.json()
        if not data or type(data) is dict: 
            break
        all_data = data + all_data
        end_time = data[0][0] - 1
        print(f"✓ Fetched batch {i+1}/7...")
        time.sleep(0.5)
    except Exception as e:
        print(f"Connection failed: {e}")
        break

if all_data:
    df = pd.DataFrame(all_data, columns=["time", "open", "high", "low", "close", "volume", "ct", "qv", "nt", "tbbv", "tbqv", "i"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[["time", "open", "high", "low", "close", "volume"]].drop_duplicates(subset=["time"]).sort_values("time")
    
    os.makedirs("backtest/data", exist_ok=True)
    df.to_csv("backtest/data/ETH_USDT_4h.csv", index=False)
    print(f"\n✅ SUCCESS! Saved {len(df)} candles directly to backtest/data/ETH_USDT_4h.csv")
else:
    print("\n❌ Failed to fetch data.")