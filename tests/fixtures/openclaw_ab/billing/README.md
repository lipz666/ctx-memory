# billing

Monthly invoice calculation. CI is red: `python3 -m unittest discover -s tests` fails.

Rules the code must follow:

- Money is `Decimal`, rounded to cents with ROUND_HALF_UP (never binary float rounding).
- Proration uses the real number of days in the billing month (February, leap years).
- Discounts apply to the subtotal **before** tax; tax is computed on the discounted amount.
- Do not edit or delete tests. Do not silence logging; the DEBUG output is part of CI.
