import QuantLib as ql

# Set evaluation date
calendar = ql.TARGET()
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# Option parameters
spot_price = 100.0
strike_price = 100.0
risk_free_rate = 0.05
dividend_rate = 0.02
volatility = 0.20
maturity_date = calendar.advance(today, ql.Period(6, ql.Months))

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

# Output results
print(f"European Call Option Price: {npv:.4f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")
print(f"Vega: {vega:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Rho: {rho:.4f}")
