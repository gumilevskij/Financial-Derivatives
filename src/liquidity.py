# Calculates bid-ask spread.
# Visualizing market depth (order book volumes).
# Computes and plots liquidity gap over time.
# Adds a simple liquidity risk indicator combining these metrics.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate order book data for 100 time points
np.random.seed(42)
time_index = pd.date_range(start='2026-07-26 09:30', periods=100, freq='T')

# Simulated bid and ask prices with noise
bid_prices = 100 + 0.05 * np.random.randn(100)
ask_prices = bid_prices + 0.05 + 0.02 * np.random.randn(100)  # ask always above bid

# Calculate bid-ask spread
spread = ask_prices - bid_prices

# Simulated volumes at top 3 bid and ask levels
bid_volumes = np.vstack([
    50 + 10 * np.random.rand(100),
    40 + 10 * np.random.rand(100),
    30 + 10 * np.random.rand(100)
])

ask_volumes = np.vstack([
    60 + 10 * np.random.rand(100),
    70 + 10 * np.random.rand(100),
    80 + 10 * np.random.rand(100)
])

# Liquidity gap simulation over 6 time buckets (e.g., days)
time_buckets = ['1D', '7D', '30D', '90D', '180D', '1Y']
liquid_assets = np.array([100, 90, 80, 70, 60, 50])
liabilities = np.array([80, 85, 75, 65, 55, 45])
liquidity_gap = liquid_assets - liabilities

# Create DataFrame for time series data
df = pd.DataFrame({
    'time': time_index,
    'bid_price': bid_prices,
    'ask_price': ask_prices,
    'spread': spread,
    'bid_volume_1': bid_volumes[0],
    'bid_volume_2': bid_volumes[1],
    'bid_volume_3': bid_volumes[2],
    'ask_volume_1': ask_volumes[0],
    'ask_volume_2': ask_volumes[1],
    'ask_volume_3': ask_volumes[2],
})

# Simple liquidity risk indicator: higher spread and lower volume indicate higher risk
# Normalize spread and volume for indicator calculation
spread_norm = (df['spread'] - df['spread'].min()) / (df['spread'].max() - df['spread'].min())
volume_sum = df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1)
volume_norm = (volume_sum - volume_sum.min()) / (volume_sum.max() - volume_sum.min())

# Liquidity risk score: weighted sum (spread up increases risk, volume up decreases risk)
liquidity_risk_score = 0.7 * spread_norm + 0.3 * (1 - volume_norm)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=False)

# 1. Bid-Ask Spread
axs[0,0].plot(df['time'], df['spread'], color='orange', label='Bid-Ask Spread')
axs[0,0].set_title('Bid-Ask Spread Over Time')
axs[0,0].set_ylabel('Spread')
axs[0,0].legend()
axs[0,0].grid(True)

# 2. Market Depth (Volumes)
axs[0,1].plot(df['time'], df['bid_volume_1'], label='Bid Volume Level 1', color='green')
axs[0,1].plot(df['time'], df['bid_volume_2'], label='Bid Volume Level 2', color='lime')
axs[0,1].plot(df['time'], df['bid_volume_3'], label='Bid Volume Level 3', color='darkgreen')
axs[0,1].plot(df['time'], df['ask_volume_1'], label='Ask Volume Level 1', color='red')
axs[0,1].plot(df['time'], df['ask_volume_2'], label='Ask Volume Level 2', color='orange')
axs[0,1].plot(df['time'], df['ask_volume_3'], label='Ask Volume Level 3', color='darkred')
axs[0,1].set_title('Market Depth Over Time')
axs[0,1].set_ylabel('Volume')
axs[0,1].legend()
axs[0,1].grid(True)

# 3. Liquidity Gap
axs[1,0].bar(time_buckets, liquidity_gap, color='blue')
axs[1,0].set_title('Liquidity Gap Over Time Buckets')
axs[1,0].set_ylabel('Liquidity Gap (Assets - Liabilities)')
axs[1,0].grid(axis='y')

# 4. Liquidity Risk Score
axs[1,1].plot(df['time'], liquidity_risk_score, color='purple', label='Liquidity Risk Score')
axs[1,1].set_title('Liquidity Risk Score Over Time')
axs[1,1].set_xlabel('Time')
axs[1,1].set_ylabel('Risk Score (Normalized)')
axs[1,1].legend()
axs[1,1].grid(True)

plt.tight_layout()
plt.show()
