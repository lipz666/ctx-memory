export function addItem(cart, sku, qty = 1) {
  const items = { ...cart.items };
  items[sku] = (items[sku] ?? 0) + qty;
  return { ...cart, items };
}

export function itemCount(cart) {
  return Object.values(cart.items).reduce((a, b) => a + b, 0);
}
