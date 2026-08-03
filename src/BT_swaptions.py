# Hull-White Swaption Pricing with Greeks (Delta, Gamma, Theta)

# Delta: Measures sensitivity of swaption price to small changes in initial short rate r0.
# Gamma: Measures curvature of price with respect to r0.
# Theta: Measures sensitivity to time decay (expiry moving closer).
# Greeks are computed by finite difference approximations using small perturbations.
# The short rate binomial tree and swaption price are plotted for visualization.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def discount_factor(t):
    return np.exp(-flat_rate * t)

def bond_price(t, T, r):
    B = (1 - np.exp(-a * (T - t))) / a
    A = np.exp(
        (B - (T - t)) * (flat_rate - (sigma**2) / (2 * a**2)) -
        (sigma**2) * B**2 / (4 * a)
    )
    return A * np.exp(-B * r)

def swap_annuity(t, r):
    times = np.arange(t + dt, t + S + dt, dt)
    return sum(bond_price(t, tau, r) for tau in times)

def forward_swap_rate(t, r):
    times = np.arange(t + dt, t + S + dt, dt)
    numerator = bond_price(t, times[0], r) - bond_price(t, times[-1] + dt, r)
    denominator = swap_annuity(t, r)
    return numerator / denominator

def build_short_rate_tree(r0, u, d, N):
    tree = [np.array([r0])]
    for i in range(1, N + 1):
        prev = tree[-1]
        level = np.zeros(i + 1)
        for j in range(i + 1):
            if j == 0:
                level[j] = prev[0] * d
            elif j == i:
                level[j] = prev[-1] * u
            else:
                level[j] = prev[j - 1] * u
        tree.append(level)
    return tree

def price_swaption_binomial(tree, K, N, dt):
    # Calculate payoff at expiry
    t_expiry = N * dt
    rates_expiry = tree[-1]
    swaption_values = np.zeros(len(rates_expiry))
    for i, r in enumerate(rates_expiry):
        F = forward_swap_rate(t_expiry, r)
        ann = swap_annuity(t_expiry, r)
        payoff = max(F - K, 0) * ann
        swaption_values[i] = payoff

    # Backward induction
    for step in range(N - 1, -1, -1):
        values_next = swaption_values
        swaption_values = np.zeros(step + 1)
        rates = tree[step]
        for i in range(step + 1):
            disc = np.exp(-rates[i] * dt)
            swaption_values[i] = disc * (p * values_next[i + 1] + (1 - p) * values_next[i])
    return swaption_values[0]

def greeks():
    """Compute Geeks."""
    # Build base tree and price
    base_tree = build_short_rate_tree(r0, u, d, N)
    base_price = price_swaption_binomial(base_tree, K, N, dt)
    
    # Greeks calculation via finite differences
    
    # Delta: price sensitivity to small change in initial short rate r0
    h = 0.0001
    tree_up = build_short_rate_tree(r0 + h, u, d, N)
    price_up = price_swaption_binomial(tree_up, K, N, dt)
    
    tree_down = build_short_rate_tree(r0 - h, u, d, N)
    price_down = price_swaption_binomial(tree_down, K, N, dt)
    
    delta = (price_up - price_down) / (2 * h)
    
    # Gamma: second order sensitivity to initial short rate
    gamma = (price_up - 2 * base_price + price_down) / (h ** 2)
    
    # Theta: sensitivity to time decay (price change if expiry moves closer by dt)
    # Price with expiry reduced by dt (N-1 steps)
    if N > 1:
        tree_theta = build_short_rate_tree(r0, u, d, N - 1)
        price_theta = price_swaption_binomial(tree_theta, K, N - 1, dt)
        theta = (price_theta - base_price) / dt
    else:
        theta = np.nan  # Not defined for N=1
        
    return base_price,delta,gamma,theta

def black_swaption_greeks(F, K, sigma, T, A):
    """Black model."""
    # Calculate d1 and d2
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    #d2 = d1 - sigma * np.sqrt(T)

    # Greeks calculation
    delta = A * stats.norm.cdf(d1)                      # Sensitivity to forward swap rate
    vega = A * F * stats.norm.pdf(d1) * np.sqrt(T)     # Sensitivity to volatility
    theta = - (A * F * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))  # Time decay
    gamma = A * stats.norm.pdf(d1) / (F * sigma * np.sqrt(T))          # Convexity wrt forward rate

    return delta,vega,theta,gamma

# Hull-White parameters
a = 0.1          # mean reversion speed
sigma = 0.01     # volatility
r0 = 0.03        # initial short rate
T_expiry = 5     # swaption expiry in years
S = 10           # swap maturity in years
dt = 1.0         # yearly steps for simplicity
N = int(T_expiry / dt)  # number of steps to expiry

# Tree parameters
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u

flat_rate = 0.03
# Risk-neutral probability
p = 0.5 
# Swaption strike
K = 0.02  

# Annuity
A = swap_annuity(T_expiry, flat_rate)
# Forward swap rate
F = forward_swap_rate(T_expiry, flat_rate)

base_price,delta,gamma,theta = greeks()
base_tree = build_short_rate_tree(r0, u, d, N)
delta_b,vega_b,theta_b,gamma_b = black_swaption_greeks(F, K, sigma, T_expiry, A)

# Output results
print(f"Swaption Price: {base_price:.6f}")
print(f"Delta: {delta:.3f}, Black: {delta_b:.3f}")
print(f"Gamma: {gamma:.3f}, Black: {gamma_b:.3f}")
print(f"Theta: {theta:.3f}, Black: {theta_b:.3f}")

# Plot short rate tree (first 6 levels)
plt.figure(figsize=(10, 6))
for t in range(min(6, N + 1)):
    plt.scatter([t] * len(base_tree[t]), base_tree[t], color='blue')
plt.title('Hull-White Short Rate Binomial Tree (first 6 steps)')
plt.xlabel('Time Steps')
plt.ylabel('Short Rate')
plt.grid(True)
plt.show()