import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Black-Scholes European Call option pricing formula
def black_scholes_call_price(S, K, T, r, sigma):
    """
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate (annual)
    sigma: Volatility (annual)
    """
    if T <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Implied volatility calculation using Brent's method
def implied_volatility_call(C_market, S, K, T, r, sigma_bounds=(1e-6, 5.0)):
    """
    C_market: Market price of the call option
    S, K, T, r: as above
    sigma_bounds: search interval for volatility
    """
    # Objective function: difference between model price and market price
    def objective(sigma):
        return black_scholes_call_price(S, K, T, r, sigma) - C_market

    # Use Brent's method to find root
    implied_vol = brentq(objective, sigma_bounds[0], sigma_bounds[1])
    return implied_vol

# Example parameters
S = 100       # Spot price
K = 100       # Strike price
T = 1.0       # Time to maturity (1 year)
r = 0.05      # Risk-free rate (5%)
C_market = 10 # Market price of the call option

# Calculate implied volatility
iv = implied_volatility_call(C_market, S, K, T, r)
print(f"Implied Volatility: {iv:.4f} or {iv*100:.2f}%")