def reconcile(payments, refunds):
    totals = {}
    for row in payments:
        invoice_id = row["invoice_id"]
        totals[invoice_id] = totals.get(invoice_id, 0.0) + float(row["amount"])
    payment_invoice = {row["id"]: row["invoice_id"] for row in payments}
    for row in refunds:
        invoice_id = payment_invoice[row["payment_id"]]
        totals[invoice_id] -= float(row["amount"])
    return {invoice_id: f"{round(amount, 2):.2f}"
            for invoice_id, amount in sorted(totals.items())}
