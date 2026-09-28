# Inventory reservation service

`reservations.py` implements a small in-memory batch reservation used by a
checkout worker. Its current implementation can leave stock and receipts
partially changed when a later order fails. The worker also retries whole
batches after a timeout, so order IDs must be idempotent.

The public API is `reserve_batch(stock, orders, processed)`. `stock` maps SKU
to available integer units. Each order has `id`, `sku`, and positive integer
`qty`. `processed` maps order ID to a receipt with the same three fields.
Return one receipt per order in input order. A duplicate ID with identical
SKU and quantity returns its existing receipt without consuming stock. A
duplicate ID with different details raises `ValueError`. If any new order
cannot be filled, raise `OutOfStock`. On every error, leave both dictionaries
exactly as they were before the call. Preserve the API and do not add external
dependencies.
