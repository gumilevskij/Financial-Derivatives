import QuantLib as ql

def numerical_greek(option, bump_func, bump_size=0.01):
    """ Generic numerical Greek calculator using bump-and-revalue.
    
    bump_func: function to bump the relevant parameter (e.g., spot price, volatility)
    bump_size: size of the bump
    """
    # Base NPV
    base_npv = option.NPV()

    # Bump up
    bump_func(bump_size)
    npv_up = option.NPV()

    # Bump down
    bump_func(-2 * bump_size)
    npv_down = option.NPV()

    # Reset bump
    bump_func(bump_size)

    # Central difference approximation
    return (npv_up - npv_down) / (2 * bump_size)

# Setup same as before
calendar = ql.TARGET()
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

spot_price = 100.0
strike_price = 100.0
risk_free_rate = 0.05
dividend_rate = 0.02
volatility = 0.20
maturity_date = calendar.advance(today, ql.Period(6, ql.Months))

payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
option = ql.VanillaOption(payoff, exercise)

spot = ql.SimpleQuote(spot_price)
spot_handle = ql.QuoteHandle(spot)
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, volatility, ql.Actual365Fixed()))

bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)
engine = ql.AnalyticEuropeanEngine(bsm_process)
option.setPricingEngine(engine)

# Numerical Delta
def bump_spot(amount):
    spot.setValue(spot.value() + amount)

numerical_delta = numerical_greek(option, bump_spot, bump_size=0.01)

# Numerical Vega
def bump_vol(amount):
    vol_ts.blackForwardVol(today, maturity_date, volatility + amount, True)

numerical_vega = numerical_greek(option, bump_vol, bump_size=0.01)

print(f"Numerical Delta: {numerical_delta:.3e}")
print(f"Numerical Vega: {numerical_vega:.3e}")
