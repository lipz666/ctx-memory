import logging
from decimal import Decimal, ROUND_HALF_UP

from .proration import prorate
from .tax import tax_for

log = logging.getLogger("billing.invoice")
CENT = Decimal("0.01")


def invoice_total(lines, region, discount_rate=Decimal("0"), period=None):
    """lines: [{"sku", "price": Decimal, "qty": int}]; period: (year, month, active_days) or None."""
    subtotal = Decimal("0")
    for line in lines:
        price = line["price"]
        if period:
            price = prorate(price, *period)
        amount = price * line["qty"]
        log.debug("line sku=%s price=%s qty=%s amount=%s", line["sku"], price, line["qty"], amount)
        subtotal += amount
    log.debug("subtotal=%s", subtotal)
    tax = tax_for(subtotal, region)
    gross = subtotal + tax
    discount = (gross * discount_rate).quantize(CENT, rounding=ROUND_HALF_UP)
    total = (gross - discount).quantize(CENT, rounding=ROUND_HALF_UP)
    log.debug("tax=%s discount=%s total=%s", tax, discount, total)
    return {"subtotal": subtotal.quantize(CENT, rounding=ROUND_HALF_UP), "tax": tax,
            "discount": discount, "total": total}
