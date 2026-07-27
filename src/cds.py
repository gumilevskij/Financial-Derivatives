import QuantLib as ql

# Set evaluation date
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today
calendar = ql.TARGET()

# CDS parameters
notional = 10000000  # 10 million
spread = 100  # in basis points (100 bps = 1%)
recovery_rate = 0.4
maturity = ql.Date(26, 7, 2029)  # approx 3 years from today

# Schedule for premium leg payments (quarterly)
schedule = ql.Schedule(
    today,
    maturity,
    ql.Period(ql.Quarterly),
    calendar,
    ql.Following,
    ql.Unadjusted,
    ql.DateGeneration.Forward,
    False
)

# Build flat hazard rate curve (constant default intensity)
hazard_rate = 0.01  # 1% hazard rate
hazard_rate_quote = ql.SimpleQuote(hazard_rate)
hazard_rate_handle = ql.QuoteHandle(hazard_rate_quote)
default_ts = ql.FlatHazardRate(today, hazard_rate_handle, ql.Actual365Fixed())
default_curve_handle = ql.DefaultProbabilityTermStructureHandle(default_ts)

# Flat risk-free discount curve
risk_free_rate = 0.01
discount_curve = ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed())
discount_curve_handle = ql.YieldTermStructureHandle(discount_curve)

# Create CDS instrument (protection buyer)
cds = ql.CreditDefaultSwap(
    ql.Protection.Buyer,
    notional,
    spread / 10000.0,  # convert bps to decimal
    schedule,
    ql.Following,
    ql.Actual365Fixed(),
    True,
    True
)

# Set pricing engine (ISDA compliant)
engine = ql.MidPointCdsEngine(default_curve_handle, recovery_rate, discount_curve_handle)
cds.setPricingEngine(engine)

# Calculate NPV and fair spread
npv = cds.NPV()
fair_spread = cds.fairSpread() * 10000  # convert to bps

print(f"CDS NPV: {npv:.2f}")
print(f"Fair Spread: {fair_spread:.2f} bps")
