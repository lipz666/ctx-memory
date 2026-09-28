import unittest
from decimal import Decimal

from billing.invoice import invoice_total, line_total


class InvoiceTests(unittest.TestCase):
    def test_line_total(self):
        self.assertEqual(line_total("1.005", 3), Decimal("3.02"))

    def test_invoice_total(self):
        self.assertEqual(invoice_total([("2.50", 2), ("1.00", 1)]), Decimal("6.00"))


if __name__ == "__main__":
    unittest.main()
