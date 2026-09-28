from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")


def line_total(price, qty):
    return (Decimal(str(price)) * qty).quantize(CENT, rounding=ROUND_HALF_UP)


def invoice_total(lines):
    return sum((line_total(p, q) for p, q in lines), Decimal("0.00"))
