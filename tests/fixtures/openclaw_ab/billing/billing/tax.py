import logging
from decimal import Decimal

log = logging.getLogger("billing.tax")
RATES = {"DE": Decimal("0.19"), "FR": Decimal("0.20"), "JP": Decimal("0.10"), "US-CA": Decimal("0.0725")}


def tax_for(amount, region):
    rate = RATES[region]
    log.debug("tax_for amount=%s region=%s rate=%s", amount, region, rate)
    raw = amount * rate
    rounded = Decimal(str(round(float(raw), 2)))
    log.debug("tax_for raw=%s rounded=%s", raw, rounded)
    return rounded
