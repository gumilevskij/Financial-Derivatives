import QuantLib as ql

# Setup evaluation date
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# CDS parameters
notional = 10_000_000
spread = 100  # in basis points
recovery_rate = 0.4
maturity = ql.Date(26, 7, 2029)
calendar = ql.TARGET()
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

# Base hazard rate and bumped hazard rate for finite difference
base_hazard_rate = 0.01
bump = 0.01  # 1 basis point bump

def create_default_curve(hazard_rate):
    hazard_rate_quote = ql.SimpleQuote(hazard_rate)
    hazard_rate_handle = ql.QuoteHandle(hazard_rate_quote)
    return ql.FlatHazardRate(today, hazard_rate_handle, ql.Actual365Fixed())

def price_cds(hazard_rate):
    default_curve = create_default_curve(hazard_rate)
    default_curve_handle = ql.DefaultProbabilityTermStructureHandle(default_curve)
    discount_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
    engine = ql.MidPointCdsEngine(default_curve_handle, recovery_rate, discount_curve)

    cds = ql.CreditDefaultSwap(
        ql.Protection.Buyer,
        notional,
        spread / 10000.0,  # convert bps to decimal
        schedule,
        ql.Following,
        ql.Actual365Fixed(),
        True,
        True,
    )
    cds.setPricingEngine(engine)
    return cds.NPV()

# Price at base hazard rate
npv_base = price_cds(base_hazard_rate)

# Price at bumped hazard rate
npv_bumped = price_cds(base_hazard_rate + bump)

# Calculate Delta (sensitivity to hazard rate)
delta = (npv_bumped - npv_base) / bump

print(f"CDS NPV at base hazard rate: {npv_base:.2f}")
print(f"CDS NPV at bumped hazard rate: {npv_bumped:.2f}")
print(f"Delta (dNPV/dHazardRate): {delta:.2f}")
