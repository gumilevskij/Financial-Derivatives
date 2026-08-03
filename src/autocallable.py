# Autocallable Monte Carlo Valuation

import numpy as np

def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42):
    np.random.seed(seed)
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def autocallable_valuation(S0, r, sigma, T, steps, n_paths, strike, coupon, autocall_barrier, autocall_dates):
    dt = T / steps
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths)
    discount_factors = np.exp(-r * np.array(autocall_dates))
    
    ncall = 0
    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        path = paths[i]
        called = False
        for j, t in enumerate(autocall_dates):
            step = int(t / dt)
            if path[step] >= autocall_barrier:
                # Autocall triggered: pay coupon + principal discounted to call date
                payoffs[i] = (1 + coupon) * discount_factors[j]
                called = True
                ncall += 1
                break
        if not called:
            # No autocall: payoff depends on final underlying price
            final_price = path[-1]
            payoff = max(final_price / strike, 0)  # Example: principal protected or equity participation
            payoffs[i] = payoff * np.exp(-r * T)
    print(f"Called: {100.0*ncall/n_paths:.1f}% of time")
    
    return np.mean(payoffs)

# Parameters example
S0 = 100
r = 0.02
sigma = 0.1
n_paths = 10000
strike = 100
coupon = 0.05  # 5% coupon on autocall
autocall_barrier = 100
autocall_dates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # years
T = len(autocall_dates)  # years
steps = 252 * T

price = autocallable_valuation(S0, r, sigma, T, steps, n_paths, strike, coupon, autocall_barrier, autocall_dates)
print(f"Autocallable Estimated Price: {price:.4f}")
