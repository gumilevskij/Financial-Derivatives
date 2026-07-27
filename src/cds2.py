import numpy as np

# Parameters
notional = 10_000_000  # 10 million
recovery_rate = 0.4
hazard_rate = 0.02  # constant hazard rate (2%)
risk_free_rate = 0.01  # constant risk-free rate (1%)
spread = 0.01  # 100 bps initial guess
maturity = 5  # years
payment_freq = 4  # quarterly payments

# Time grid for payments
dt = 1 / payment_freq
payment_times = np.arange(dt, maturity + dt, dt)

# Survival probability function Q(0,t) = exp(-lambda * t)
def survival_prob(t, hazard_rate):
    return np.exp(-hazard_rate * t)

# Discount factor function D(0,t) = exp(-r * t)
def discount_factor(t, risk_free_rate):
    return np.exp(-risk_free_rate * t)

# Calculate premium leg PV
def premium_leg_pv(spread, notional, payment_times, hazard_rate, risk_free_rate):
    pv = 0.0
    for t in payment_times:
        q = survival_prob(t, hazard_rate)
        d = discount_factor(t, risk_free_rate)
        pv += spread * dt * notional * d * q
    return pv

# Calculate protection leg PV
def protection_leg_pv(notional, recovery_rate, payment_times, hazard_rate, risk_free_rate):
    pv = 0.0
    prev_q = 1.0
    for t in payment_times:
        q = survival_prob(t, hazard_rate)
        d = discount_factor(t, risk_free_rate)
        default_prob = prev_q - q
        pv += notional * (1 - recovery_rate) * d * default_prob
        prev_q = q
    return pv

# Calculate fair spread
def fair_spread(notional, recovery_rate, payment_times, hazard_rate, risk_free_rate):
    prot_leg = protection_leg_pv(notional, recovery_rate, payment_times, hazard_rate, risk_free_rate)
    prem_leg = premium_leg_pv(1.0, notional, payment_times, hazard_rate, risk_free_rate)  # unit spread
    return prot_leg / prem_leg

# Compute values
premium_pv = premium_leg_pv(spread, notional, payment_times, hazard_rate, risk_free_rate)
protection_pv = protection_leg_pv(notional, recovery_rate, payment_times, hazard_rate, risk_free_rate)
fair_spread_value = fair_spread(notional, recovery_rate, payment_times, hazard_rate, risk_free_rate)

print(f"Premium Leg PV (spread={spread*10000:.0f} bps): ${premium_pv:,.2f}")
print(f"Protection Leg PV: ${protection_pv:,.2f}")
print(f"Fair Spread: {fair_spread_value*10000:.2f} bps")