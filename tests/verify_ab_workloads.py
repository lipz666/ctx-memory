"""Independent verifier for the synthetic paired Agent workloads."""

import copy
from decimal import Decimal
import sys
import unittest


def inventory_suite():
    from reservations import OutOfStock, reserve_batch

    class InventoryTests(unittest.TestCase):
        def test_atomic_on_shortage(self):
            stock = {"A": 3, "B": 0}
            processed = {}
            before = copy.deepcopy((stock, processed))
            with self.assertRaises(OutOfStock):
                reserve_batch(stock, [{"id": "1", "sku": "A", "qty": 2},
                                      {"id": "2", "sku": "B", "qty": 1}], processed)
            self.assertEqual((stock, processed), before)

        def test_retries_and_same_batch_duplicates(self):
            stock = {"A": 5}
            processed = {}
            order = {"id": "1", "sku": "A", "qty": 2}
            first = reserve_batch(stock, [order, order], processed)
            second = reserve_batch(stock, [order], processed)
            self.assertEqual(first, [order, order])
            self.assertEqual(second, [order])
            self.assertEqual(stock, {"A": 3})

        def test_conflict_preserves_state(self):
            stock = {"A": 4, "B": 4}
            processed = {"old": {"id": "old", "sku": "A", "qty": 1}}
            before = copy.deepcopy((stock, processed))
            with self.assertRaises(ValueError):
                reserve_batch(stock, [{"id": "new", "sku": "A", "qty": 2},
                                      {"id": "old", "sku": "B", "qty": 1}], processed)
            self.assertEqual((stock, processed), before)

        def test_success_and_input_order(self):
            stock = {"A": 3, "B": 2}
            processed = {}
            orders = [{"id": "1", "sku": "B", "qty": 2},
                      {"id": "2", "sku": "A", "qty": 1}]
            self.assertEqual(reserve_batch(stock, orders, processed), orders)
            self.assertEqual(stock, {"A": 2, "B": 0})

    return unittest.defaultTestLoader.loadTestsFromTestCase(InventoryTests)


def ledger_suite():
    from ledger import reconcile

    class LedgerTests(unittest.TestCase):
        def test_decimal_and_repeated_exports(self):
            payments = [{"id": "p1", "invoice_id": "I-2", "amount": "0.10"},
                        {"id": "p2", "invoice_id": "I-2", "amount": "0.20"}]
            refunds = [{"id": "r1", "payment_id": "p2", "amount": "0.10"}]
            before = copy.deepcopy((payments, refunds))
            self.assertEqual(reconcile(payments + [payments[0]], refunds + [refunds[0]]),
                             {"I-2": "0.20"})
            self.assertEqual((payments, refunds), before)

        def test_changed_duplicate_rejected(self):
            rows = [{"id": "p1", "invoice_id": "I-1", "amount": "1.00"},
                    {"id": "p1", "invoice_id": "I-1", "amount": "2.00"}]
            with self.assertRaises(ValueError):
                reconcile(rows, [])

        def test_unknown_and_excess_refunds_rejected(self):
            payment = [{"id": "p1", "invoice_id": "I-1", "amount": "1.00"}]
            with self.assertRaises(ValueError):
                reconcile(payment, [{"id": "r1", "payment_id": "missing", "amount": "0.50"}])
            with self.assertRaises(ValueError):
                reconcile(payment, [{"id": "r1", "payment_id": "p1", "amount": "0.60"},
                                    {"id": "r2", "payment_id": "p1", "amount": "0.50"}])

        def test_multiple_invoices_and_exact_cent(self):
            payments = [{"id": "p1", "invoice_id": "B", "amount": "100.05"},
                        {"id": "p2", "invoice_id": "A", "amount": "0.01"}]
            got = reconcile(payments, [{"id": "r1", "payment_id": "p1", "amount": "0.02"}])
            self.assertEqual(got, {"A": "0.01", "B": "100.03"})
            for amount in got.values():
                self.assertEqual(Decimal(amount).as_tuple().exponent, -2)

    return unittest.defaultTestLoader.loadTestsFromTestCase(LedgerTests)


def billing_suite():
    import calendar
    import logging
    from decimal import ROUND_HALF_UP
    from billing.invoice import invoice_total
    from billing.proration import prorate
    from billing.tax import tax_for

    logging.disable(logging.CRITICAL)
    cent = Decimal("0.01")
    rates = {"DE": Decimal("0.19"), "FR": Decimal("0.20"), "JP": Decimal("0.10"),
             "US-CA": Decimal("0.0725")}

    def tax(amount, region):
        return (amount * rates[region]).quantize(cent, rounding=ROUND_HALF_UP)

    class BillingTests(unittest.TestCase):
        def test_tax_half_up(self):
            for cents in range(0, 1000, 7):
                for region in rates:
                    amount = Decimal(cents) / 100 + Decimal("0.05")
                    self.assertEqual(tax_for(amount, region), tax(amount, region))

        def test_proration_real_days(self):
            for year in (2023, 2024, 2100):
                for month in range(1, 13):
                    days = calendar.monthrange(year, month)[1]
                    self.assertEqual(prorate(Decimal("99.99"), year, month, days), Decimal("99.99"))
                    self.assertEqual(prorate(Decimal(days), year, month, 7), Decimal("7.00"))

        def test_discount_before_tax(self):
            lines = [{"sku": f"S{i}", "price": Decimal("7.35") + i, "qty": 1 + i % 4}
                     for i in range(25)]
            subtotal = sum(line["price"] * line["qty"] for line in lines)
            for region in rates:
                for rate in (Decimal("0"), Decimal("0.05"), Decimal("0.2")):
                    discount = (subtotal * rate).quantize(cent, rounding=ROUND_HALF_UP)
                    got = invoice_total(lines, region, rate)
                    self.assertEqual(got["discount"], discount)
                    self.assertEqual(got["tax"], tax(subtotal - discount, region))
                    self.assertEqual(got["total"], subtotal - discount + tax(subtotal - discount, region))

    return unittest.defaultTestLoader.loadTestsFromTestCase(BillingTests)


if __name__ == "__main__":
    suites = {"inventory": inventory_suite, "ledger": ledger_suite, "billing": billing_suite}
    if len(sys.argv) != 2 or sys.argv[1] not in suites:
        raise SystemExit("usage: verify_ab_workloads.py inventory|ledger|billing")
    sys.path.insert(0, ".")
    suite = suites[sys.argv[1]]()
    raise SystemExit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
