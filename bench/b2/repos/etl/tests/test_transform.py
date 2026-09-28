import unittest

from etl.transform import clean


class CleanTests(unittest.TestCase):
    def test_clean(self):
        rows = clean([{"sku": " ab-1", "qty": "2"}, {"sku": "", "qty": "5"}, {"sku": "cd-9", "qty": ""}])
        self.assertEqual(rows, [{"sku": "AB-1", "qty": 2}, {"sku": "CD-9", "qty": 0}])


if __name__ == "__main__":
    unittest.main()
