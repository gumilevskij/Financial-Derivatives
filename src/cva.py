# Credit Valuation Adjustment (CVA) reduces the value of your asset to reflect the risk that 
# the counterparty might default.

# Simulates GBM paths for the underlying asset.
# Autocallable valuation checks early redemption at observation dates and calculates a range accrual coupon based on occupation time within a price range.
# CVA calculation estimates expected loss due to issuer default using constant hazard rate and recovery assumptions.
# Plot shows an example path with range accrual boundaries to visualize occupation time.

import numpy as np

# --- 1. Simulate Geometric Brownian Motion Paths ---
def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42):
    np.random.seed(seed)
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths, dt

# --- 2. Autocallable Valuation with Range Accrual Coupon ---
def autocallable_with_range_accrual(paths, dt, r, coupon_rate, autocall_barrier, autocall_times, range_low, range_high):
    n_paths, steps = paths.shape
    steps -= 1  # because paths include initial spot
    T = steps * dt
    discount_factors = np.exp(-r * np.array(autocall_times))
    
    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        path = paths[i]
        called = False
        # Check autocall triggers
        for j, t in enumerate(autocall_times):
            step = int(t / T * steps)
            if path[step] >= autocall_barrier:
                # Autocall triggered: pay coupon + principal discounted to call date
                payoffs[i] = (1 + coupon_rate) * discount_factors[j]
                called = True
                break
        if not called:
            # No autocall: payoff includes principal and range accrual coupon
            # Estimate occupation time fraction in range
            in_range = np.logical_and(path[1:] >= range_low, path[1:] <= range_high)
            occupation_fraction = np.sum(in_range) / steps
            # Coupon proportional to occupation fraction
            coupon_payoff = coupon_rate * occupation_fraction
            # Principal repayment assumed at maturity (no loss)
            payoffs[i] = (1 + coupon_payoff) * np.exp(-r * T)
    return np.mean(payoffs)

# --- 3. CVA Calculation ---
def cva(paths, dt, r, hazard_rate, recovery_rate, portfolio_values):
    """
    paths: simulated underlying paths (n_paths x steps+1)
    dt: time step size
    r: risk-free rate
    hazard_rate: constant hazard rate for issuer default
    recovery_rate: recovery rate on default
    portfolio_values: portfolio values at each time step (n_paths x steps+1)
    """
    n_paths, steps = paths.shape
    steps -= 1
    discount_factors = np.exp(-r * np.arange(steps + 1) * dt)
    survival_probs = np.exp(-hazard_rate * np.arange(steps + 1) * dt)
    default_probs = hazard_rate * survival_probs * dt
    
    # Exposure is positive portfolio value (assume zero if negative)
    exposures = np.maximum(portfolio_values[:, 1:], 0)
    
    # CVA per path: sum over time of discounted expected loss
    cva_paths = np.sum(discount_factors[1:] * exposures * default_probs[1:] * (1 - recovery_rate), axis=1)
    return np.mean(cva_paths)

# --- Parameters ---
S0 = 100
r = 0.01
sigma = 0.2
T = 3  # years
steps = 252 * T
n_paths = 10000

coupon_rate = 0.05
autocall_barrier = 105
autocall_times = [1, 2, 3]  # years

range_low = 95
range_high = 105

hazard_rate = 0.02  # 2% annual hazard rate
recovery_rate = 0.4

# --- Simulation ---
paths, dt = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths)

# Portfolio values at each time step (for CVA), simplified as discounted payoff if held to maturity
# Here, approximate portfolio value as discounted intrinsic value of underlying (proxy)
portfolio_values = np.maximum(paths - S0, 0) * np.exp(-r * np.arange(steps + 1) * dt)

# Valuation without credit risk
price = autocallable_with_range_accrual(paths, dt, r, coupon_rate, autocall_barrier, autocall_times, range_low, range_high)
print(f"Autocallable with Range Accrual Price (no credit risk): {price:.4f}")

# CVA adjustment
cva_value = cva(paths, dt, r, hazard_rate, recovery_rate, portfolio_values)
print(f"CVA Adjustment: {cva_value:.4f}")

# Price adjusted for issuer credit risk
price_credit_adjusted = price - cva_value
print(f"Price Adjusted for Issuer Credit Risk: {price_credit_adjusted:.4f}")

# --- Plot example path and occupation time ---
import matplotlib.pyplot as plt

example_path = paths[0]
time_grid = np.linspace(0, T, steps + 1)
plt.plot(time_grid, example_path, label='Underlying Price Path')
plt.axhline(range_low, color='red', linestyle='--', label='Range Low')
plt.axhline(range_high, color='green', linestyle='--', label='Range High')
plt.title('Example Underlying Path with Range Accrual Boundaries')
plt.xlabel('Time (years)')
plt.ylabel('Price')
plt.legend()
plt.show()
