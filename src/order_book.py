#Simple order book simulation models bids and asks, matches orders when bid price ≥ ask price, 
# and plots the resulting order book after matching.

import numpy as np
import matplotlib.pyplot as plt

class OrderBook:
    def __init__(self):
        self.bids = []  # list of (price, quantity)
        self.asks = []  # list of (price, quantity)

    def add_order(self, side, price, quantity):
        book = self.bids if side == 'bid' else self.asks
        book.append((price, quantity))
        book.sort(key=lambda x: x[0], reverse=(side == 'bid'))

    def match_orders(self):
        trades = []
        while self.bids and self.asks and self.bids[0][0] >= self.asks[0][0]:
            bid_price, bid_qty = self.bids[0]
            ask_price, ask_qty = self.asks[0]
            trade_qty = min(bid_qty, ask_qty)
            trades.append((bid_price, ask_price, trade_qty))
            if bid_qty > trade_qty:
                self.bids[0] = (bid_price, bid_qty - trade_qty)
            else:
                self.bids.pop(0)
            if ask_qty > trade_qty:
                self.asks[0] = (ask_price, ask_qty - trade_qty)
            else:
                self.asks.pop(0)
        return trades

# Simulate order book activity
np.random.seed(42)
order_book = OrderBook()

# Add random orders
for _ in range(50):
    side = np.random.choice(['bid', 'ask'])
    price = 100 + np.random.randn()
    quantity = np.random.randint(1, 10)
    order_book.add_order(side, price, quantity)

# Match orders
trades = order_book.match_orders()

# Plot order book and trades
bid_prices = [p for p, q in order_book.bids]
bid_qtys = [q for p, q in order_book.bids]
ask_prices = [p for p, q in order_book.asks]
ask_qtys = [q for p, q in order_book.asks]

plt.figure(figsize=(10, 6))
plt.bar(bid_prices, bid_qtys, width=0.1, color='green', alpha=0.6, label='Bids')
plt.bar(ask_prices, ask_qtys, width=0.1, color='red', alpha=0.6, label='Asks')
plt.title('Simulated Order Book After Matching')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.show()

print(f"Number of trades matched: {len(trades)}")