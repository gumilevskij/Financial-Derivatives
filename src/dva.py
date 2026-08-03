# Credit Valuation Adjustment (CVA) reduces the value of your asset to reflect the risk that 
# the counterparty might default.
# Debit Valuation Adjustment (DVA) increases the value of your liability to reflect the possibility 
# that entity might default and you not have to pay the full amount.

# The plot shows CVA increasing linearly with counterparty default probability, 
# reflecting expected loss on the asset side.
# DVA also increases linearly but at a lower level due to smaller negative exposure.
# The net BCVA (bilateral CVA) is the difference, representing the overall credit risk adjustment.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
exposure = 1_000_000          # Positive exposure (asset)
negative_exposure = -500_000  # Negative exposure (liability)
rec_counterparty = 0.4        # Recovery rate for counterparty
rec_own = 0.4                 # Recovery rate for own entity
risk_free_rate = 0.02         # 2% annual risk-free rate
maturity = 1                  # 1 year

# Simulate a range of default probabilities (0% to 10%)
default_probs = np.linspace(0, 0.1, 100)

# Calculate discount factor
discount_factor = np.exp(-risk_free_rate * maturity)

# Calculate Adjustments
CVA_values = (1 - rec_counterparty) * exposure * default_probs * discount_factor
DVA_values = (1 - rec_own) * abs(negative_exposure) * default_probs * discount_factor
BCVA_values = CVA_values - DVA_values  # Bilateral CVA (net adjustment)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(default_probs * 100, CVA_values / 1000, label='CVA (Counterparty Risk)', color='blue')
plt.plot(default_probs * 100, DVA_values / 1000, label='DVA (Own Default Risk)', color='red')
plt.plot(default_probs * 100, BCVA_values / 1000, label='Net BCVA', color='purple', linestyle='--')

plt.xlabel('Default Probability (%)')
plt.ylabel('Adjustment Value (Thousands of $)')
plt.title('Bilateral Credit Valuation Adjustment Analysis')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()