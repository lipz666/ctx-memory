# Payment ledger reconciliation

`ledger.py` reconciles a payment gateway's payment and refund export. The
gateway sometimes repeats rows in a later export. Amounts are decimal strings
in the account currency; float arithmetic is unsuitable for financial totals.

`reconcile(payments, refunds)` returns a dictionary mapping invoice ID to a
two-decimal string net balance, sorted by invoice ID. Each payment row has
`id`, `invoice_id`, and positive `amount`. Each refund row has `id`,
`payment_id`, and positive `amount`. Repeated rows with the same ID and same
contents count once. Repeated IDs with changed contents, refunds for unknown
payments, or total refunds greater than the associated payment must raise
`ValueError`. No input object may be modified. Preserve the API and use the
standard library only.
