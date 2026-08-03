# Combines PCA and machine learning (Random Forest) to model and forecast volatility surface dynamics:

# We simulate a time series of volatility surfaces.
# Apply PCA to reduce dimensionality to 3 principal components.
# Use past 5 days of PCA scores as features to train a Random Forest to predict the next day’s PCA scores.
# Forecast 10 days ahead iteratively.
# Reconstruct and plot forecasted volatility surfaces.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate volatility surface time series data
np.random.seed(42)
maturities = np.array([0.1, 0.25, 0.5, 1, 2, 3])
strikes = np.linspace(80, 120, 9)

n_days = 200
base_vol = 0.2

vol_surfaces = []
for day in range(n_days):
    daily_surface = base_vol + 0.1 * np.exp(-maturities[:, None]) * (strikes[None, :] / 100 - 1)**2
    daily_surface += 0.01 * np.random.randn(*daily_surface.shape)  # noise
    daily_surface += 0.001 * day  # slow drift
    vol_surfaces.append(daily_surface.flatten())

vol_surfaces = np.array(vol_surfaces)

# PCA for dimensionality reduction
pca = PCA(n_components=3)
scores = pca.fit_transform(vol_surfaces)  # shape (n_days, 3)

# Prepare supervised dataset: past 5 days PCA scores to predict next day
window_size = 5
X, y = [], []
for i in range(window_size, n_days):
    X.append(scores[i-window_size:i].flatten())
    y.append(scores[i])
X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest MSE on test set: {mse:.6f}")

# Plot actual vs predicted PC1 scores on test set
plt.figure(figsize=(10,5))
plt.plot(y_test[:, 0], label='Actual PC1')
plt.plot(y_pred[:, 0], label='Predicted PC1', linestyle='--')
plt.title('Random Forest Prediction of PC1 Scores (Volatility Surface Level)')
plt.xlabel('Test Sample')
plt.ylabel('PC1 Score')
plt.legend()
plt.grid(True)
plt.show()

# Forecast next 10 days iteratively
last_window = scores[-window_size:].flatten().reshape(1, -1)
forecast_steps = 10
forecasted_scores = []
for _ in range(forecast_steps):
    pred = model.predict(last_window)
    forecasted_scores.append(pred.flatten())
    last_window = np.roll(last_window, -3)
    last_window[0, -3:] = pred

forecasted_scores = np.array(forecasted_scores)

# Reconstruct forecasted volatility surfaces
reconstructed_surfaces = []
for score_vec in forecasted_scores:
    reconstructed_surface = pca.inverse_transform(score_vec)
    reconstructed_surfaces.append(reconstructed_surface.reshape(len(maturities), len(strikes)))

# Plot forecasted volatility surfaces
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i, ax in enumerate(axs.flatten()):
    if i < forecast_steps:
        c = ax.contourf(strikes, maturities, reconstructed_surfaces[i], cmap='viridis')
        ax.set_title(f'Day +{i+1}')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
    else:
        ax.axis('off')
fig.colorbar(c, ax=axs.ravel().tolist(), shrink=0.6)
plt.suptitle('Forecasted Volatility Surfaces from Random Forest on PCA Scores')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()