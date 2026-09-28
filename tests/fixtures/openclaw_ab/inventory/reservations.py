class OutOfStock(Exception):
    pass


def reserve_batch(stock, orders, processed):
    receipts = []
    for order in orders:
        order_id = order["id"]
        sku = order["sku"]
        qty = order["qty"]
        if order_id in processed:
            receipts.append(processed[order_id])
            continue
        if stock.get(sku, 0) < qty:
            raise OutOfStock(sku)
        stock[sku] = stock.get(sku, 0) - qty
        receipt = {"id": order_id, "sku": sku, "qty": qty}
        processed[order_id] = receipt
        receipts.append(receipt)
    return receipts
