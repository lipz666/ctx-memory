import itertools
import unittest
from decimal import Decimal, ROUND_HALF_UP

from billing.invoice import invoice_total
from billing.proration import prorate
from billing.tax import tax_for

D = Decimal
CENT = D("0.01")


def expected_tax(amount, region):
    rates = {"DE": D("0.19"), "FR": D("0.20"), "JP": D("0.10"), "US-CA": D("0.0725")}
    return (amount * rates[region]).quantize(CENT, rounding=ROUND_HALF_UP)


class TaxTests(unittest.TestCase):
    def test_half_up_rounding_many_amounts(self):
        for cents, region in itertools.product(range(1, 400, 3), ["DE", "FR", "JP", "US-CA"]):
            amount = D(cents) / 100 + D("0.05")
            with self.subTest(amount=amount, region=region):
                self.assertEqual(tax_for(amount, region), expected_tax(amount, region))


class ProrationTests(unittest.TestCase):
    def test_real_month_lengths(self):
        cases = [(2024, 2, 29, D("29.00")), (2023, 2, 14, D("15.00")), (2024, 1, 31, D("31.00")),
                 (2024, 4, 15, D("15.00")), (2023, 12, 1, D("1.00"))]
        for year, month, active, want in cases:
            price = {2: D("29.00") if year == 2024 else D("30.00"), 1: D("31.00"), 4: D("30.00"),
                     12: D("31.00")}[month]
            with self.subTest(year=year, month=month):
                self.assertEqual(prorate(price, year, month, active), want)

    def test_full_month_is_full_price(self):
        for year, month in itertools.product([2023, 2024], range(1, 13)):
            with self.subTest(year=year, month=month):
                self.assertEqual(prorate(D("99.99"), year, month, _days(year, month)), D("99.99"))


class InvoiceTests(unittest.TestCase):
    def test_discount_before_tax(self):
        lines = [{"sku": f"S{i}", "price": D("19.99") + i, "qty": 1 + i % 3} for i in range(40)]
        for region in ["DE", "FR", "JP", "US-CA"]:
            for rate in [D("0"), D("0.1"), D("0.15")]:
                with self.subTest(region=region, rate=rate):
                    got = invoice_total(lines, region, rate)
                    subtotal = sum(l["price"] * l["qty"] for l in lines)
                    discount = (subtotal * rate).quantize(CENT, rounding=ROUND_HALF_UP)
                    tax = expected_tax(subtotal - discount, region)
                    self.assertEqual(got["discount"], discount)
                    self.assertEqual(got["tax"], tax)
                    self.assertEqual(got["total"], subtotal - discount + tax)


def _days(year, month):
    import calendar
    return calendar.monthrange(year, month)[1]


if __name__ == "__main__":
    unittest.main()
