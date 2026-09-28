import { test } from "node:test";
import assert from "node:assert/strict";
import { addItem, itemCount } from "../src/cart.js";

test("adds items", () => {
  const cart = addItem(addItem({ items: {} }, "a"), "a", 2);
  assert.equal(itemCount(cart), 3);
});
