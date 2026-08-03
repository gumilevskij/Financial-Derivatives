# Pricing call option with Quantlib
import numpy as np
import QuantLib as ql


def price(spot_price,strike_price,risk_free_rate,dividend_rate,volatility,maturity):
    
    # Set evaluation date
    calendar = ql.TARGET()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    maturity_date = calendar.advance(today, ql.Period(maturity, ql.Months))
    
    # Construct the option payoff and exercise
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)
    
    # Create the European option
    european_option = ql.VanillaOption(payoff, exercise)
    
    # Market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, volatility, ql.Actual365Fixed()))
    
    # Black-Scholes-Merton process
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)
    
    # Pricing engine
    engine = ql.AnalyticEuropeanEngine(bsm_process)
    european_option.setPricingEngine(engine)
    
    # Calculate option price and Greeks
    npv = european_option.NPV()
    delta = european_option.delta()
    gamma = european_option.gamma()
    vega = european_option.vega()
    theta = european_option.theta()
    rho = european_option.rho()
    
    return npv,delta,gamma,vega,theta,rho


# Option parameters
spot_price = 100.0
strike_price = 110.0
risk_free_rate = 0.05
dividend_rate = 0.02
volatility = 0.20
maturities = [6,12,24,36,120,360]
T = len(maturities)

npv=np.zeros(T); delta=np.zeros(T); gamma=np.zeros(T); vega=np.zeros(T); theta=np.zeros(T); rho=np.zeros(T)

for i,maturity in enumerate(maturities):
    npv[i],delta[i],gamma[i],vega[i],theta[i],rho[i] = price(spot_price,strike_price,risk_free_rate,dividend_rate,volatility,maturity)
    
# Output results
print("\nEuropean Call Option Greeks vs Maturity:")
print(f"NPV: {npv}")
print(f"Delta: {delta}")
print(f"Gamma: {gamma}")
print(f"Vega: {vega}")
print(f"Theta: {theta}")
print(f"Rho: {rho}")
