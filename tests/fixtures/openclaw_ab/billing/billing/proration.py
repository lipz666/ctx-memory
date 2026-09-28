import logging
from decimal import Decimal

log = logging.getLogger("billing.proration")


def prorate(monthly_price, year, month, active_days):
    days_in_month = 30
    log.debug("prorate price=%s year=%s month=%s active=%s days=%s",
              monthly_price, year, month, active_days, days_in_month)
    share = Decimal(active_days) / Decimal(days_in_month)
    return (monthly_price * share).quantize(Decimal("0.01"))
