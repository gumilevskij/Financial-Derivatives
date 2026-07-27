import QuantLib as ql

# Set evaluation date
calendar = ql.TARGET()
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# Market data
risk_free_rate = 0.01
volatility = 0.20  # Black volatility for swaption
settlement_days = 2

# FIX 1: Use Relinkable handles so the pricing engine automatically updates when market data bumps
discount_curve_handle = ql.RelinkableYieldTermStructureHandle()
flat_curve = ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed())
discount_curve_handle.linkTo(flat_curve)

# Swaption parameters
option_maturity = ql.Period(5, ql.Years)  # Swaption expiry
swap_tenor = ql.Period(10, ql.Years)     # Underlying swap length
fixed_leg_frequency = ql.Annual
fixed_leg_convention = ql.Unadjusted
fixed_leg_daycount = ql.Thirty360(ql.Thirty360.European)

# Pass the relinkable handle to the index
floating_leg_index = ql.Euribor6M(discount_curve_handle)

# FIX 2: Calculate the exact exercise date, then align the swap's start date to it
exercise_date = calendar.advance(today, option_maturity)
start_date = calendar.advance(exercise_date, settlement_days, ql.Days)

# Construct swap schedules relative to the correct forward timeline
fixed_schedule = ql.Schedule(start_date, calendar.advance(start_date, swap_tenor),
                             ql.Period(fixed_leg_frequency), calendar,
                             fixed_leg_convention, fixed_leg_convention,
                             ql.DateGeneration.Forward, False)
floating_schedule = ql.Schedule(start_date, calendar.advance(start_date, swap_tenor),
                                ql.Period(ql.Semiannual), calendar,
                                ql.ModifiedFollowing, ql.ModifiedFollowing,
                                ql.DateGeneration.Forward, False)

# Create vanilla swap (payer)
nominal = 1000000
fixed_rate = 0.015
vanilla_swap = ql.VanillaSwap(ql.VanillaSwap.Payer, nominal,
                             fixed_schedule, fixed_rate, fixed_leg_daycount,
                             floating_schedule, floating_leg_index, 0.0,
                             floating_leg_index.dayCounter())

# Set swap pricing engine
swap_engine = ql.DiscountingSwapEngine(discount_curve_handle)
vanilla_swap.setPricingEngine(swap_engine)

# Create European exercise for swaption
exercise = ql.EuropeanExercise(exercise_date)

# Create swaption
swaption = ql.Swaption(vanilla_swap, exercise)

# FIX 3: Use Relinkable handle for volatility structure
# vol_ts_handle = ql.RelinkableBlackVolTermStructureHandle()
# volatility_handle = ql.BlackConstantVol(today, calendar, volatility, ql.Actual365Fixed())
# vol_ts_handle.linkTo(volatility_handle)


vol_quote = ql.SimpleQuote(volatility)
vol_handle = ql.QuoteHandle(vol_quote)

# Create a constant Swaption Volatility Structure
constant_swaption_vol = ql.ConstantSwaptionVolatility(
    today, calendar, ql.ModifiedFollowing, vol_handle, ql.Actual365Fixed()
)

# Use a SwaptionVolatilityStructureHandle instead of a BlackVolTermStructureHandle
vol_ts_handle = ql.RelinkableSwaptionVolatilityStructureHandle()
vol_ts_handle.linkTo(constant_swaption_vol)


# Black swaption engine
swaption_engine = ql.BlackSwaptionEngine(discount_curve_handle, vol_ts_handle)
swaption.setPricingEngine(swaption_engine)

# Calculate Base NPV
base_npv = swaption.NPV()

# Numerical Greeks via safe bump-and-revalue
bump_size = 0.0001

# Cleaned up Bump Helper Functions
def get_bumped_rate_npv(bump):
    new_curve = ql.FlatForward(today, risk_free_rate + bump, ql.Actual365Fixed())
    discount_curve_handle.linkTo(new_curve)  # Updates both engines and index simultaneously
    val = swaption.NPV()
    return val

def get_bumped_vol_npv(bump):
    new_vol = ql.BlackConstantVol(today, calendar, volatility + bump, ql.Actual365Fixed())
    vol_ts_handle.linkTo(new_vol)
    val = swaption.NPV()
    return val

# Calculate delta (sensitivity to interest rates)
npv_up_rate = get_bumped_rate_npv(bump_size)
npv_down_rate = get_bumped_rate_npv(-bump_size)
get_bumped_rate_npv(0.0)  # Reset back to original baseline

delta = (npv_up_rate - npv_down_rate) / (2 * bump_size)

# Calculate vega (sensitivity to volatility)
npv_up_vol = get_bumped_vol_npv(bump_size)
npv_down_vol = get_bumped_vol_npv(-bump_size)
get_bumped_vol_npv(0.0)  # Reset back to original baseline

vega = (npv_up_vol - npv_down_vol) / (2 * bump_size)

print(f"Swaption NPV: {base_npv:.2f}")
print(f"Numerical Delta (rate sensitivity): {delta:.2f}")
print(f"Numerical Vega (volatility sensitivity): {vega:.2f}")
