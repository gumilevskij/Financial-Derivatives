#Vectorized Python implementation for calibrating both SABR and Heston models efficiently:

# SABR uses a vectorized Hagan formula for implied volatilities.
# Heston uses the COS method for fast option pricing.
# Both calibrations use differential evolution optimization.
# The final plot compares market implied vols with calibrated model vols.

# The COS method (Fourier-Cosine expansion method) is a fast and accurate numerical technique for option pricing, 
# especially under models with known characteristic functions such as the Heston stochastic volatility model. 
# It was introduced by Fang and Oosterlee (2008) and is widely used because it converges rapidly and is 
# computationally efficient.

#This code efficiently calibrates SABR and Heston models to a synthetic implied volatility smile 
# using vectorized pricing and global optimization, then plots the fit.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, brentq
from scipy.stats import norm

# Vectorized SABR implied vol (Hagan et al.)
def sabr_implied_vol_vec(F, K, T, alpha, beta, rho, nu):
    K = np.array(K)
    logFK = np.log(F / K)
    FK_beta = (F * K)**((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    z_over_xz = np.ones_like(z)
    mask = np.abs(z) > 1e-7
    z_over_xz[mask] = z[mask] / x_z[mask]
    z_over_xz = np.nan_to_num(z_over_xz, nan=1.0)

    term1 = ((1 - beta)**2 / 24) * (alpha**2) / (FK_beta**2)
    term2 = (rho * beta * nu * alpha) / (4 * FK_beta)
    term3 = (nu**2 * (2 - 3 * rho**2)) / 24

    denom = FK_beta * (1 + ((1 - beta)**2 / 24) * logFK**2 + ((1 - beta)**4 / 1920) * logFK**4)
    vol = (alpha / denom) * z_over_xz * (1 + (term1 + term2 + term3) * T)
    return vol

# COS method for Heston pricing
def heston_cos_price(S0, K, T, r, params, N=128, L=8):
    kappa, theta, sigma, rho, v0 = params
    x = np.log(S0 / K)
    a = -L * np.sqrt(T)
    b = L * np.sqrt(T)
    k = np.arange(N).reshape(-1, 1)
    u = k * np.pi / (b - a)

    d = np.sqrt((rho * sigma * 1j * u - kappa)**2 + sigma**2 * (1j * u + u**2))
    g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)

    C = (kappa * theta / sigma**2) * ((kappa - rho * sigma * 1j * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho * sigma * 1j * u - d) / sigma**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    phi = np.exp(C + D * v0 + 1j * u * (x + r * T))

    def chi(c, d, a, b, k):
        res = 1 / (1 + (k * np.pi / (b - a))**2) * (
            np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) -
            np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c) +
            (k * np.pi / (b - a)) * np.sin(k * np.pi * (d - a) / (b - a)) * np.exp(d) -
            (k * np.pi / (b - a)) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
        )
        return res

    def psi(c, d, a, b, k):
        res = np.zeros_like(k, dtype=float)
        mask = k == 0
        res[mask] = d - c
        res[~mask] = (b - a) / (k[~mask] * np.pi) * (
            np.sin(k[~mask] * np.pi * (d - a) / (b - a)) -
            np.sin(k[~mask] * np.pi * (c - a) / (b - a))
        )
        return res

    U_k = 2 / (b - a) * (chi(0, b, a, b, k) - psi(0, b, a, b, k))
    U_k[0] *= 0.5

    price = np.exp(-r * T) * K * np.real(np.sum(phi * U_k))
    return max(0, price)

# Implied vol from price via bisection
def get_implied_vol(price, S0, K, T, r):
    def bs_obj(sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - price
    try:
        return brentq(bs_obj, 1e-9, 5)
    except:
        return 0.0

# Calibration objective for SABR
def calibration_objective_sabr(params, F, T, strikes, market_vols):
    model_vols = sabr_implied_vol_vec(F, strikes, T, *params)
    if np.any(model_vols <= 0) or np.any(np.isnan(model_vols)):
        return 1e10
    return np.sum((model_vols - market_vols)**2)

# Calibration objective for Heston
def calibration_objective_heston(params, S0, r, T, strikes, market_vols):
    kappa, theta, sigma, rho, v0 = params
    penalty = 0
    if 2 * kappa * theta < sigma**2:
        penalty = 10
    err = 0
    for K, mkt_vol in zip(strikes, market_vols):
        price = heston_cos_price(S0, K, T, r, params)
        model_vol = get_implied_vol(price, S0, K, T, r)
        err += (model_vol - mkt_vol)**2
    return np.sqrt(err) + penalty

# Example market data
S0, r, T = 100, 0.03, 1.0
strikes = np.array([80, 90, 95, 100, 105, 110, 120])
market_vols = np.array([0.28, 0.24, 0.22, 0.20, 0.21, 0.23, 0.26])

# Calibrate SABR
res_sabr = differential_evolution(calibration_objective_sabr,
                                 [(0.01, 1), (0.0, 1.0), (-0.9, 0.9), (0.01, 1.0)],
                                 args=(S0*np.exp(r*T), T, strikes, market_vols), maxiter=100)

# Calibrate Heston
res_heston = differential_evolution(calibration_objective_heston,
                                   [(0.1, 5), (0.01, 0.5), (0.01, 1), (-0.9, 0.5), (0.01, 0.5)],
                                   args=(S0, r, T, strikes, market_vols), maxiter=30)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(strikes, market_vols, 'ko', label='Market')
plt.plot(strikes, sabr_implied_vol_vec(S0*np.exp(r*T), strikes, T, *res_sabr.x), '--', label='SABR')
heston_vols = [get_implied_vol(heston_cos_price(S0, k, T, r, res_heston.x), S0, k, T, r) for k in strikes]
plt.plot(strikes, heston_vols, 'r:', label='Heston')
plt.legend()
plt.title("Model Calibration to Implied Volatility Smile")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.show()

print("Calibrated SABR parameters:", res_sabr.x)
print("Calibrated Heston parameters:", res_heston.x)
