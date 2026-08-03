# A Dynamic Conditional Correlation GARCH (DCC-GARCH) model is a multivariate econometric framework introduced 
# by Robert Engle in 2002 to estimate time-varying correlations and conditional covariances in financial time series.
# It avoids the curse of dimensionality by splitting estimation into univariate GARCH volatilities and a dynamic 
# correlation structure.

# Fits univariate GARCH(1,1) to each series to get standardized residuals.
# Use those residuals to estimate DCC parameters a,b and dynamic correlations R t
# Plots the estimated time-varying correlation against the true constant correlation used in simulation.

import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import random
from scipy.optimize import minimize

def fit_univariate_garch(series):
    model = arch_model(series, vol='Garch', p=1, q=1, dist='normal')
    res = model.fit(disp='off')
    cond_var = res.conditional_volatility
    residuals = res.resid
    std_resid = residuals / cond_var
    return std_resid

def dcc_negloglik(theta, u):
    a, b = theta
    T, k = u.shape
    Qbar = np.cov(u, rowvar=False)
    Q = Qbar.copy()
    negloglik = 0.0

    for t in range(T):
        if t > 0:
            ztm1 = u[t-1][:, None]
            Q = (1 - a - b) * Qbar + a * (ztm1 @ ztm1.T) + b * Q
        diag_sqrt = np.sqrt(np.diag(Q))
        inv_sqrt = 1.0 / diag_sqrt
        invD = np.diag(inv_sqrt)
        R = invD @ Q @ invD

        detR = np.linalg.det(R)
        if detR <= 0:
            return 1e10
        invR = np.linalg.inv(R)
        ut = u[t][:, None]
        negloglik += 0.5 * (np.log(detR) + (ut.T @ invR @ ut).item())

    return negloglik

def fit_dcc(u):
    init = np.array([0.02, 0.96])
    bounds = [(1e-6, 0.999), (1e-6, 0.999)]
    cons = ({'type': 'ineq', 'fun': lambda x: 0.999 - (x[0] + x[1])})
    res = minimize(dcc_negloglik, init, args=(u,), bounds=bounds, constraints=cons, method='SLSQP')
    if not res.success:
        raise RuntimeError("DCC optimization failed: " + res.message)
    a_hat, b_hat = res.x

    T, k = u.shape
    Qbar = np.cov(u, rowvar=False)
    Q = Qbar.copy()
    R_ts = np.zeros((T, k, k))
    for t in range(T):
        if t > 0:
            ztm1 = u[t-1][:, None]
            Q = (1 - a_hat - b_hat) * Qbar + a_hat * (ztm1 @ ztm1.T) + b_hat * Q
        diag_sqrt = np.sqrt(np.diag(Q))
        inv_sqrt = 1.0 / diag_sqrt
        invD = np.diag(inv_sqrt)
        R = invD @ Q @ invD
        R_ts[t] = R

    return {'a': a_hat, 'b': b_hat, 'R_ts': R_ts}

# Generate synthetic data with constant correlation
np.random.seed(42)
T = 200
lst = []; corr_true = []
for t in range(T):
    corr = 0.5+0.1*np.sin(t/10) + 0.1*random.random()
    cov_true = np.array([[1, corr], [corr, 1]])
    x = np.random.multivariate_normal(mean=[0,0], cov=cov_true, size=1)
    corr_true.append(corr)
    lst.append(np.squeeze(x))
data = np.array(lst)
corr_true = np.array(corr_true)

# Fit univariate GARCH(1,1) to each series
std_resid_0 = fit_univariate_garch(data[:,0])
std_resid_1 = fit_univariate_garch(data[:,1])

# Stack standardized residuals
u = np.column_stack((std_resid_0, std_resid_1))

# Fit DCC model
result = fit_dcc(u)

# Extract time-varying correlations
corr_ts = result['R_ts'][:, 0, 1]
err = corr_ts/corr_true - 1
print(f"Error: {100*np.mean(err):.2f}%")

# Plot
plt.figure(figsize=(12,6))
plt.plot(corr_ts, label='Estimated Time-varying Correlation')
plt.scatter(range(T),corr_true, color='r', linestyle='--', label='True Correlation')
plt.title('DCC(1,1) Estimated Time-varying Correlation with Univariate GARCH')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.show()

print(f"Estimated parameters: a = {result['a']:.4f}, b = {result['b']:.4f}")
