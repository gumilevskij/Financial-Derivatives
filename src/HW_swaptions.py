# Hull-White swaption pricing code with plots to visualize:

# The simulated short rate path up to swaption expiry.
# The distribution of forward swap rates at expiry.
# The swaption price as a function of strike rate.

# Simulates 1000 short rate paths up to swaption expiry.
# Calculates forward swap rates and annuities at expiry for each path.
# Prices the swaption using the Black formula with Hull-White implied volatility.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Hull-White model parameters
a = 0.1          # mean reversion speed
sigma = 0.01     # volatility
r0 = 0.03        # initial short rate
T = 5            # swaption expiry in years
S = 10           # swap maturity in years (swaption underlying swap length)
dt = 0.01        # time step for simulation
N = int(T / dt)  # steps for expiry
M = 1000         # number of simulations

# Yield curve (flat for simplicity)
flat_rate = 0.03

# Discount factor function assuming flat yield curve
def discount_factor(t):
    return np.exp(-flat_rate * t)

# Zero-coupon bond price under Hull-White model (analytical)
def P(t, T, r_t):
    B = (1 - np.exp(-a * (T - t))) / a
    A = np.exp(
        (B - (T - t)) * (flat_rate - (sigma**2) / (2 * a**2)) -
        (sigma**2) * B**2 / (4 * a)
    )
    return A * np.exp(-B * r_t)

# Simulate short rate path to expiry T using Euler-Maruyama
def simulate_short_rate_path(r0, a, sigma, T, dt, M):
    N = int(T / dt)
    r = np.zeros((M, N + 1))
    r[:, 0] = r0
    for i in range(1, N + 1):
        dW = np.sqrt(dt) * np.random.randn(M)
        dr = a * (flat_rate - r[:, i-1]) * dt + sigma * dW
        r[:, i] = r[:, i-1] + dr
    return r

# Swaption parameters
swaption_strike = 0.035  # fixed rate of the swap
payment_times = np.arange(T + 1, T + S + 1)  # annual payments from expiry to maturity

# Calculate swap annuity at time t given short rate r_t
def swap_annuity(t, r_t):
    return sum(P(t, tau, r_t) for tau in payment_times)

# Calculate forward swap rate at time t given short rate r_t
def forward_swap_rate(t, r_t):
    numerator = P(t, payment_times[0], r_t) - P(t, payment_times[-1] + 1, r_t)
    denominator = swap_annuity(t, r_t)
    return numerator / denominator

# Black formula for swaption price
def black_swaption_price(F, K, sigma, T, annuity):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = annuity * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    return price

# Approximate swaption volatility (simplified)
swaption_vol = sigma * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a))

# Simulate M short rate paths at expiry T
np.random.seed(42)
r_paths = simulate_short_rate_path(r0, a, sigma, T, dt, M)
r_T = r_paths[:, -1]

# Calculate forward swap rates and annuities at expiry for all paths
F_values = np.array([forward_swap_rate(T, r) for r in r_T])
annuity_values = np.array([swap_annuity(T, r) for r in r_T])

# Calculate swaption prices for all paths using Black formula
swaption_prices = black_swaption_price(F_values, swaption_strike, swaption_vol, S, annuity_values)

# Plot 1: Sample short rate paths (first 20)
plt.figure(figsize=(14, 5))
for i in range(20):
    plt.plot(np.linspace(0, T, int(T/dt) + 1), r_paths[i], lw=0.8, alpha=0.7)
plt.title('Sample Hull-White Short Rate Paths up to Swaption Expiry')
plt.xlabel('Time (Years)')
plt.ylabel('Short Rate')
plt.grid(True)
plt.show()

# Plot 2: Distribution of forward swap rates at expiry
plt.figure(figsize=(8, 5))
plt.hist(F_values, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Forward Swap Rates at Swaption Expiry')
plt.xlabel('Forward Swap Rate')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot 3: Swaption price vs strike rate curve
strike_range = np.linspace(0.01, 0.06, 50)
avg_annuity = annuity_values.mean()
avg_F = F_values.mean()
prices_vs_strike = [black_swaption_price(avg_F, K, swaption_vol, S, avg_annuity) for K in strike_range]

plt.figure(figsize=(8, 5))
plt.plot(strike_range, prices_vs_strike, label='Swaption Price')
plt.axvline(swaption_strike, color='red', linestyle='--', label='Current Strike')
plt.title('Swaption Price vs Strike Rate')
plt.xlabel('Strike Rate')
plt.ylabel('Swaption Price')
plt.legend()
plt.grid(True)
plt.show()

print(f"Average swaption price from simulation: {swaption_prices.mean():.4f}")