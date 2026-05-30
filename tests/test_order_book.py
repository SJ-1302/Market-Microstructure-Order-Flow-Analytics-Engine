"""
Unit tests for the Limit Order Book matching engine in python/data_generation/order_book_engine.py.
"""

from __future__ import annotations

import unittest
from python.data_generation.order_book_engine import (
    Side,
    Order,
    TradeRecord,
    CancelRecord,
    OrderEvent,
    OrderBook,
)


class TestOrderBook(unittest.TestCase):
    def test_order_book_initialization(self) -> None:
        book = OrderBook("TEST_SYMBOL", tick_size=0.05)
        self.assertEqual(book.symbol, "TEST_SYMBOL")
        self.assertEqual(book.tick_size, 0.05)
        self.assertIsNone(book.get_best_bid())
        self.assertIsNone(book.get_best_ask())
        self.assertIsNone(book.get_mid_price())
        self.assertIsNone(book.get_spread())
        self.assertEqual(book.total_bid_volume, 0)
        self.assertEqual(book.total_ask_volume, 0)
        self.assertEqual(len(book.trades), 0)
        self.assertEqual(len(book.events), 0)

    def test_add_limit_orders(self) -> None:
        book = OrderBook("TEST", tick_size=0.05)

        # Place bids
        oid1 = book.add_limit_order(Side.BID, 100.0, 100, 1.0)
        oid2 = book.add_limit_order(Side.BID, 100.05, 200, 2.0)
        oid3 = book.add_limit_order(Side.BID, 99.90, 150, 3.0)

        # Place asks
        oid4 = book.add_limit_order(Side.ASK, 100.15, 100, 4.0)
        oid5 = book.add_limit_order(Side.ASK, 100.20, 300, 5.0)

        self.assertEqual(book.get_best_bid(), 100.05)
        self.assertEqual(book.get_best_ask(), 100.15)
        self.assertEqual(book.get_mid_price(), 100.10)
        self.assertAlmostEqual(book.get_spread(), 0.10)
        self.assertEqual(book.total_bid_volume, 450)
        self.assertEqual(book.total_ask_volume, 400)

        # Check total orders count
        self.assertEqual(book.get_total_orders(), 5)

    def test_order_matching_fifo(self) -> None:
        book = OrderBook("TEST", tick_size=0.05)

        # Place two bids at the same price (100.0)
        # FIFO priority: oid1 should be filled before oid2
        oid1 = book.add_limit_order(Side.BID, 100.0, 50, 1.0)
        oid2 = book.add_limit_order(Side.BID, 100.0, 100, 2.0)

        # Place a market sell order of quantity 70
        trades = book.add_market_order(Side.ASK, 70, 3.0)

        self.assertEqual(len(trades), 2)
        # First fill: oid1 fully filled
        self.assertEqual(trades[0].maker_order_id, oid1)
        self.assertEqual(trades[0].quantity, 50)
        self.assertEqual(trades[0].price, 100.0)

        # Second fill: oid2 partially filled (20 out of 100)
        self.assertEqual(trades[1].maker_order_id, oid2)
        self.assertEqual(trades[1].quantity, 20)

        # Verify book status
        self.assertEqual(book.total_bid_volume, 80)  # 0 + 80 remaining from oid2
        self.assertEqual(book.get_best_bid(), 100.0)
        self.assertEqual(book.get_total_orders(), 1)

    def test_cancel_order(self) -> None:
        book = OrderBook("TEST", tick_size=0.05)
        oid1 = book.add_limit_order(Side.BID, 100.0, 100, 1.0)
        oid2 = book.add_limit_order(Side.BID, 99.5, 200, 2.0)

        self.assertEqual(book.total_bid_volume, 300)
        self.assertEqual(book.get_best_bid(), 100.0)

        # Cancel the top bid
        success = book.cancel_order(oid1, 3.0)
        self.assertTrue(success)
        self.assertEqual(book.total_bid_volume, 200)
        self.assertEqual(book.get_best_bid(), 99.5)

        # Try to cancel it again (should fail/return False)
        success2 = book.cancel_order(oid1, 4.0)
        self.assertFalse(success2)

        # Try to cancel non-existent order
        self.assertFalse(book.cancel_order(99999, 5.0))

    def test_modify_order(self) -> None:
        book = OrderBook("TEST", tick_size=0.05)
        oid1 = book.add_limit_order(Side.BID, 100.0, 100, 1.0)

        # Modify to a smaller quantity (shrinks order, retains priority queue position)
        success = book.modify_order(oid1, 40, 2.0)
        self.assertTrue(success)
        self.assertEqual(book.total_bid_volume, 40)
        self.assertEqual(book.get_total_orders(), 1)

        # Modify to a larger quantity (cancels and inserts a new order at the end of queue)
        success2 = book.modify_order(oid1, 150, 3.0)
        self.assertTrue(success2)
        # Original order oid1 is now inactive, a new limit order is placed.
        # Let's verify that the total bid volume is now 150.
        self.assertEqual(book.total_bid_volume, 150)
        self.assertEqual(book.get_total_orders(), 1)

        # Modify to 0 should cancel
        success3 = book.modify_order(oid1, 0, 4.0)
        # Note: since oid1 was replaced in the queue by a new order (due to quantity expansion),
        # modifying the original oid1 to 0 might return False because it's already cancelled.
        # Let's check the active order. The new order ID would be oid1 + 1 (since it uses _next_order_id).
        # We can find the new order id from book._orders values
        new_oid = [o.order_id for o in book._orders.values() if o.is_active][0]
        success4 = book.modify_order(new_oid, 0, 5.0)
        self.assertTrue(success4)
        self.assertEqual(book.total_bid_volume, 0)
        self.assertEqual(book.get_total_orders(), 0)

    def test_get_snapshot(self) -> None:
        book = OrderBook("TEST", tick_size=0.05)

        # Bids
        book.add_limit_order(Side.BID, 100.0, 100, 1.0)
        book.add_limit_order(Side.BID, 100.0, 50, 2.0)
        book.add_limit_order(Side.BID, 99.95, 200, 3.0)

        # Asks
        book.add_limit_order(Side.ASK, 100.05, 120, 4.0)
        book.add_limit_order(Side.ASK, 100.10, 80, 5.0)

        snap = book.get_snapshot(levels=2)
        self.assertEqual(snap["symbol"], "TEST")
        self.assertEqual(snap["best_bid"], 100.0)
        self.assertEqual(snap["best_ask"], 100.05)
        self.assertEqual(snap["mid_price"], 100.025)
        self.assertAlmostEqual(snap["spread"], 0.05)

        # Bid side levels
        self.assertEqual(snap["bid_prices"], [100.0, 99.95])
        self.assertEqual(snap["bid_quantities"], [150, 200])  # 100 + 50 = 150 at 100.0
        self.assertEqual(snap["bid_order_counts"], [2, 1])

        # Ask side levels
        self.assertEqual(snap["ask_prices"], [100.05, 100.10])
        self.assertEqual(snap["ask_quantities"], [120, 80])
        self.assertEqual(snap["ask_order_counts"], [1, 1])

        self.assertEqual(snap["total_bid_volume"], 350)
        self.assertEqual(snap["total_ask_volume"], 200)


if __name__ == "__main__":
    unittest.main()
