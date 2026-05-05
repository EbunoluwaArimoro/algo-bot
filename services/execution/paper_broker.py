import asyncio, random

class PaperBroker:
    def __init__(self, latency_ms=150, slip_bps=8, outage_prob=0.002):
        self.latency_ms   = latency_ms   # simulate co-location gap
        self.slip_bps     = slip_bps     # 8 basis points slippage
        self.outage_prob  = outage_prob  # 0.2% chance of outage per order

    async def place_order(self, symbol, side, qty, price):
        # 1. Simulate network latency to exchange
        await asyncio.sleep(self.latency_ms / 1000)

        # 2. Simulate random outage — test recovery logic
        if random.random() < self.outage_prob:
            raise ConnectionError("Simulated exchange outage")

        # 3. Apply realistic slippage
        slip = price * (self.slip_bps / 10000)
        fill_price = price + slip if side == "buy" else price - slip

        # 4. Add randomness to fill — markets aren't clean
        fill_price *= (1 + random.gauss(0, 0.0003))

        return {"symbol": symbol, "side": side,
                "qty": qty, "fill_price": fill_price,
                "slippage": abs(fill_price - price)}