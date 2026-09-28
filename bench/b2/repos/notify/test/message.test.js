import { test } from "node:test";
import assert from "node:assert/strict";
import { orderShipped } from "../src/message.js";

test("shipped message", () => {
  assert.equal(orderShipped({ id: 7 }), "Order 7 has shipped.");
});
