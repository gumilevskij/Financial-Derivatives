# Heath-Jarrow-Morton (HJM) model describes the evolution of the instantaneous 
# forward rate curve: df(t,T)=α(t,T)dt+σ(t,T)dW.
# where σ(t,T) is the volatility function of the forward rate,
# and α(t,T) is the drift term that must satisfy the no arbitrage condition:
# α(t,T) = σ(t,T) * ∫_t^T σ(t,s) ds

# This code:
# - Simulates the two-factor HJM forward curve and short rates,
# - Prices compounded SOFR caplets by Monte Carlo,
# - Calibrates the model parameters to synthetic market caplet prices,
# - Compares and plots market vs calibrated model prices.

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -------------------------------
# HJM two-factor simulator (returns short_rates and discount integrals)
# -------------------------------
def simulate_hjm_two_factor_short_rates(v1, a1, v2, a2, rho,
    r0=0.02,T_max=5.0,dt=1/252,dT=1/12,n_paths=2000,seed=None):
    """
    Simulate two-factor HJM forward curve and return short rates r(t) for each path/time.
    Returns:
      time_grid, matur_grid, short_rates (n_paths x n_t), integrals (n_paths x n_t cumulative integral up to t)
    """
    rng = np.random.default_rng(seed)

    time_grid = np.arange(0.0, T_max + dt/2, dt)
    matur_grid = np.arange(dT, T_max + dT/2, dT)
    n_t = len(time_grid)
    n_T = len(matur_grid)

    # initial forward curve flat at r0
    f_paths = np.full((n_paths, n_T), r0, dtype=float)

    def sigma_k_at_t(vk, ak, t):
        tau = matur_grid - t
        tau = np.clip(tau, 0.0, None)
        return vk * np.exp(-ak * tau)

    def integral_sigma_k(vk, ak, t):
        tau = matur_grid - t
        tau = np.clip(tau, 0.0, None)
        if ak == 0:
            return vk * tau
        return vk * (1.0 - np.exp(-ak * tau)) / ak

    corr = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(corr)

    short_rates = np.zeros((n_paths, n_t))
    cumulative_integral = np.zeros((n_paths, n_t))  # ∫_0^t r_s ds approximated discretely

    sqrt_dt = np.sqrt(dt)
    integral_so_far = np.zeros(n_paths)

    for i in range(n_t):
        t = time_grid[i]
        # short rate approx = f(t, first maturity >= t)
        idxs = np.where(matur_grid >= t)[0]
        if len(idxs) == 0:
            sr = f_paths[:, -1]
        else:
            sr = f_paths[:, idxs[0]]
        short_rates[:, i] = sr

        # update cumulative integral (left Riemann)
        if i > 0:
            integral_so_far += short_rates[:, i-1] * dt
        cumulative_integral[:, i] = integral_so_far

        if i == n_t - 1:
            break

        sig1 = sigma_k_at_t(v1, a1, t)
        sig2 = sigma_k_at_t(v2, a2, t)
        int1 = integral_sigma_k(v1, a1, t)
        int2 = integral_sigma_k(v2, a2, t)
        alpha = sig1 * int1 + sig2 * int2  # shape (n_T,)

        z = rng.normal(size=(n_paths, 2))
        dW = z @ L.T * sqrt_dt  # shape (n_paths, 2)

        # Euler update of forward curves for all paths
        # broadcasting: alpha[None,:] -> (n_paths, n_T)
        f_paths += alpha[None, :] * dt + sig1[None, :] * dW[:, 0:1] + sig2[None, :] * dW[:, 1:2]

    return time_grid, matur_grid, short_rates, cumulative_integral

# -------------------------------
# Price a compounded-SOFR caplet via Monte Carlo given HJM parameters
# -------------------------------
def price_compounded_sofr_caplet_mc(
    params,  # [v1,a1,v2,a2,rho]
    caplet_spec,  # dict with keys: accrual_start, accrual_end, K, notional (optional)
    market_convention_dt=1/252,
    r0=0.02,dt=1/252, dT=1/12,n_paths=4000,seed=None
):
    v1, a1, v2, a2, rho = params
    accrual_start = caplet_spec['accrual_start']
    accrual_end = caplet_spec['accrual_end']
    K = caplet_spec['K']
    N = caplet_spec.get('notional', 1.0)

    T_max = accrual_end
    time_grid, matur_grid, short_rates, cumulative_integral = simulate_hjm_two_factor_short_rates(
        v1, a1, v2, a2, rho,
        r0=r0, T_max=T_max, dt=dt, dT=dT, n_paths=n_paths, seed=seed
    )

    # indices in time grid for accrual window
    accrual_idx = np.where((time_grid >= accrual_start) & (time_grid <= accrual_end))[0]
    if accrual_idx.size == 0:
        raise ValueError("Accrual window too small for chosen dt or grids.")

    # compute discrete compounded rate per path: G = ∏ (1 + r_t * dt) over accrual days
    # use short_rates[:, idx] and dt
    # compute G in log-space for numerical stability
    log_factors = np.log(1.0 + short_rates[:, accrual_idx] * dt)  # shape (n_paths, n_accrual_steps)
    log_G = np.sum(log_factors, axis=1)
    G = np.exp(log_G)
    R_comp = G - 1.0

    # discount factor to payment time (assume paid at accrual_end): DF = exp(-∫_0^{accrual_end} r_s ds)
    idx_acc_end = accrual_idx[-1]
    integrals_to_end = cumulative_integral[:, idx_acc_end] + short_rates[:, idx_acc_end] * 0.0  # left Riemann already included
    DF = np.exp(-integrals_to_end)

    payoff = N * np.maximum(R_comp - K, 0.0)
    discounted_payoff = payoff * DF

    price = np.mean(discounted_payoff)
    price_std_error = np.std(discounted_payoff) / np.sqrt(n_paths)

    return price, price_std_error

# -------------------------------
# Calibration routine: minimize squared error to market caplet prices
# -------------------------------
def calibrate_to_caplets(
    market_caplets,   # list of dicts: each dict contains accrual_start, accrual_end, K, market_price
    initial_guess=[0.01, 0.3, 0.007, 0.05, 0.2],
    bounds=[(1e-5, 0.1), (1e-3, 2.0), (1e-5, 0.1), (1e-3, 2.0), (-0.95, 0.95)],
    r0=0.02,dt=1/252, dT=1/12,n_paths=2000, verbose=False,seed=0
):
    """
    Calibrate [v1,a1,v2,a2,rho] to match market caplet prices (least-squares).
    market_caplets: list of dicts with keys 'accrual_start','accrual_end','K','market_price'
    """

    def objective(x):
        # x = [v1,a1,v2,a2,rho]
        sq_err = 0.0
        for cl in market_caplets:
            model_price, se = price_compounded_sofr_caplet_mc(
                x, cl, r0=r0, dt=dt, dT=dT, n_paths=n_paths, seed=seed
            )
            mkt_price = cl['market_price']
            # Weight by inverse variance (use se^2) if available to stabilize
            weight = 1.0 / max(se, 1e-8)
            err = (model_price - mkt_price) * weight
            sq_err += err**2
        if verbose:
            print(f"params {x}, obj {sq_err:.6e}")
        return sq_err

    res = minimize(objective, x0=np.array(initial_guess), bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 50, 'disp': verbose})
    return res

# -------------------------------
# Synthetic market prices and calibration
# -------------------------------
if __name__ == "__main__":
    # True parameters used to generate synthetic market prices
    true_params = [0.012, 0.4, 0.007, 0.05, 0.25]  # v1,a1,v2,a2,rho

    # Define a few caplets (accrual windows of 3 months at different future dates)
    caplets = [
        {'accrual_start': 0.5, 'accrual_end': 0.75, 'K': 0.01, 'notional': 1.0},
        {'accrual_start': 1.0, 'accrual_end': 1.25, 'K': 0.012, 'notional': 1.0},
        {'accrual_start': 2.0, 'accrual_end': 2.25, 'K': 0.015,  'notional': 1.0}
    ]

    # Generate synthetic market prices by pricing under true_params with a large MC sample
    print("Generating synthetic market prices (this may take some seconds)...")
    for cl in caplets:
        p, se = price_compounded_sofr_caplet_mc(true_params, cl, r0=0.015, dt=1/252, dT=1/12, n_paths=10000, seed=123)
        cl['market_price'] = p
        print(f"Caplet {cl['accrual_start']}-{cl['accrual_end']} K={cl['K']:.3%} market_price={p:.8f} se~{se:.5f}")

    # Calibration starting guess (intentionally off)
    init = [0.014, 0.85, 0.0105, 0.22, -0.04]

    print("\nStarting calibration...")
    res = calibrate_to_caplets(
        market_caplets=caplets,initial_guess=init,
        bounds=[(1e-5, 0.05), (1e-3, 2.0), (1e-5, 0.05), (1e-3, 2.0), (-0.95, 0.95)],
        r0=0.015,dt=1/252, dT=1/12,
        n_paths=2500,   # smaller n_paths for speed during calibration
        verbose=True,seed=42)

    print("\nCalibration result:")
    print(res)

    # Compare market vs model prices at calibrated params
    calibrated_params = res.x
    model_prices = []
    market_prices = []
    for cl in caplets:
        model_p, model_se = price_compounded_sofr_caplet_mc(calibrated_params, cl, r0=0.015, dt=1/252, dT=1/12, n_paths=10000, seed=99)
        model_prices.append(model_p)
        market_prices.append(cl['market_price'])
        print(f"Caplet {cl['accrual_start']}-{cl['accrual_end']}: market={cl['market_price']:.8f}, model={model_p:.8f}, se~{model_se:.6f}")

    # Plot market vs model prices
    indices = np.arange(len(caplets))
    width = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(indices - width/2, market_prices, width=width, label='Market (synthetic)')
    plt.bar(indices + width/2, model_prices,  width=width, label='Model (calibrated)')
    labels = [f"{cl['accrual_start']:.2f}-{cl['accrual_end']:.2f}" for cl in caplets]
    plt.xticks(indices, labels)
    plt.ylabel('Caplet price')
    plt.title('Market vs Model Caplet Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
