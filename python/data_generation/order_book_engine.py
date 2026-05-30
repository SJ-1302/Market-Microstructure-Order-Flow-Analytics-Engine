"""
Limit Order Book Engine with Price-Time Priority
=================================================

Implements a high-fidelity limit order book (LOB) simulator with:
- Price-time priority matching (FIFO at each price level)
- Running book volume tracking in O(1) time
- Fast get_snapshot top-level queries
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Side(Enum):
    """Order side."""
    BID = "BID"
    ASK = "ASK"


@dataclass
class Order:
    """Represents a single limit order in the book."""
    order_id: int
    timestamp: float
    side: Side
    price: float
    quantity: int
    remaining_qty: int
    is_active: bool = True


@dataclass
class TradeRecord:
    """Represents a single trade execution (fill)."""
    trade_id: int
    timestamp: float
    price: float
    quantity: int
    aggressor_side: Side
    maker_order_id: int
    taker_order_id: int


@dataclass
class CancelRecord:
    """Record of a cancelled order."""
    timestamp: float
    order_id: int
    side: Side
    price: float
    cancelled_qty: int


@dataclass
class OrderEvent:
    """Generic order event for the event log."""
    timestamp: float
    event_type: str
    order_id: int
    side: str
    price: float
    quantity: int


class OrderBook:
    """
    Limit Order Book with price-time priority (FIFO) matching.
    """

    def __init__(self, symbol: str, tick_size: float = 0.05) -> None:
        self.symbol = symbol
        self.tick_size = tick_size

        self._bids: Dict[float, deque] = {}  # price → deque[Order]
        self._asks: Dict[float, deque] = {}  # price → deque[Order]

        self._bid_heap: List[float] = []  # contains -price values
        self._ask_heap: List[float] = []  # contains price values

        self._bid_prices: set = set()
        self._ask_prices: set = set()

        self._orders: Dict[int, Order] = {}

        self._next_order_id: int = 1
        self._next_trade_id: int = 1

        self.trades: List[TradeRecord] = []
        self.events: List[OrderEvent] = []
        self.cancels: List[CancelRecord] = []
        self.order_count: int = 0

        # Running total of book volumes for O(1) query
        self.total_bid_volume: int = 0
        self.total_ask_volume: int = 0

    def _round_price(self, price: float) -> float:
        """Round price to the nearest tick size."""
        return round(round(price / self.tick_size) * self.tick_size, 2)

    def _clean_bid_heap(self) -> None:
        """Remove stale entries from the bid heap top."""
        while self._bid_heap and (-self._bid_heap[0]) not in self._bid_prices:
            heapq.heappop(self._bid_heap)

    def _clean_ask_heap(self) -> None:
        """Remove stale entries from the ask heap top."""
        while self._ask_heap and self._ask_heap[0] not in self._ask_prices:
            heapq.heappop(self._ask_heap)

    def _remove_price_level(self, side: Side, price: float) -> None:
        """Remove an empty price level from tracking."""
        if side == Side.BID:
            self._bid_prices.discard(price)
            if price in self._bids:
                del self._bids[price]
        else:
            self._ask_prices.discard(price)
            if price in self._asks:
                del self._asks[price]

    def _add_to_book(self, order: Order) -> None:
        """Insert an order into the appropriate side of the book."""
        price = order.price
        if order.side == Side.BID:
            if price not in self._bids:
                self._bids[price] = deque()
                self._bid_prices.add(price)
                heapq.heappush(self._bid_heap, -price)
            self._bids[price].append(order)
            self.total_bid_volume += order.remaining_qty
        else:
            if price not in self._asks:
                self._asks[price] = deque()
                self._ask_prices.add(price)
                heapq.heappush(self._ask_heap, price)
            self._asks[price].append(order)
            self.total_ask_volume += order.remaining_qty

    def _match_incoming_order(
        self,
        order: Order,
        is_market_order: bool = False,
    ) -> List[TradeRecord]:
        """
        Attempt to match an incoming order against the opposite side.
        """
        fills: List[TradeRecord] = []

        if order.side == Side.BID:
            opposite_book = self._asks
            get_best = self.get_best_ask
            price_check = lambda ask_price: (
                is_market_order or order.price >= ask_price
            )
        else:
            opposite_book = self._bids
            get_best = self.get_best_bid
            price_check = lambda bid_price: (
                is_market_order or order.price <= bid_price
            )

        while order.remaining_qty > 0:
            best_price = get_best()
            if best_price is None:
                break
            if not price_check(best_price):
                break

            if order.side == Side.BID:
                queue = self._asks.get(best_price)
            else:
                queue = self._bids.get(best_price)

            if queue is None or len(queue) == 0:
                if order.side == Side.BID:
                    self._remove_price_level(Side.ASK, best_price)
                    self._clean_ask_heap()
                else:
                    self._remove_price_level(Side.BID, best_price)
                    self._clean_bid_heap()
                continue

            maker = queue[0]

            if not maker.is_active or maker.remaining_qty <= 0:
                queue.popleft()
                if len(queue) == 0:
                    if order.side == Side.BID:
                        self._remove_price_level(Side.ASK, best_price)
                        self._clean_ask_heap()
                    else:
                        self._remove_price_level(Side.BID, best_price)
                        self._clean_bid_heap()
                continue

            fill_qty = min(order.remaining_qty, maker.remaining_qty)
            fill_price = best_price

            order.remaining_qty -= fill_qty
            maker.remaining_qty -= fill_qty

            # Update running total volumes for matching maker
            if order.side == Side.BID:
                self.total_ask_volume -= fill_qty
            else:
                self.total_bid_volume -= fill_qty

            trade = TradeRecord(
                trade_id=self._next_trade_id,
                timestamp=order.timestamp,
                price=fill_price,
                quantity=fill_qty,
                aggressor_side=order.side,
                maker_order_id=maker.order_id,
                taker_order_id=order.order_id,
            )
            self._next_trade_id += 1
            self.trades.append(trade)
            fills.append(trade)

            self.events.append(OrderEvent(
                timestamp=order.timestamp,
                event_type="FILL",
                order_id=maker.order_id,
                side=maker.side.value,
                price=fill_price,
                quantity=fill_qty,
            ))

            if maker.remaining_qty <= 0:
                maker.is_active = False
                queue.popleft()
                if len(queue) == 0:
                    if order.side == Side.BID:
                        self._remove_price_level(Side.ASK, best_price)
                        self._clean_ask_heap()
                    else:
                        self._remove_price_level(Side.BID, best_price)
                        self._clean_bid_heap()

        if order.remaining_qty <= 0:
            order.is_active = False

        return fills

    def add_limit_order(
        self,
        side: Side,
        price: float,
        quantity: int,
        timestamp: float,
    ) -> int:
        """Submit a limit order to the book."""
        price = self._round_price(price)
        order_id = self._next_order_id
        self._next_order_id += 1
        self.order_count += 1

        order = Order(
            order_id=order_id,
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity,
            remaining_qty=quantity,
        )
        self._orders[order_id] = order

        self.events.append(OrderEvent(
            timestamp=timestamp,
            event_type="LIMIT_ORDER",
            order_id=order_id,
            side=side.value,
            price=price,
            quantity=quantity,
        ))

        self._match_incoming_order(order, is_market_order=False)

        if order.remaining_qty > 0 and order.is_active:
            self._add_to_book(order)

        return order_id

    def add_market_order(
        self,
        side: Side,
        quantity: int,
        timestamp: float,
    ) -> List[TradeRecord]:
        """Submit a market order."""
        order_id = self._next_order_id
        self._next_order_id += 1
        self.order_count += 1

        order = Order(
            order_id=order_id,
            timestamp=timestamp,
            side=side,
            price=0.0 if side == Side.BID else float("inf"),
            quantity=quantity,
            remaining_qty=quantity,
        )
        self._orders[order_id] = order

        self.events.append(OrderEvent(
            timestamp=timestamp,
            event_type="MARKET_ORDER",
            order_id=order_id,
            side=side.value,
            price=0.0,
            quantity=quantity,
        ))

        fills = self._match_incoming_order(order, is_market_order=True)
        order.is_active = False
        return fills

    def cancel_order(self, order_id: int, timestamp: float = 0.0) -> bool:
        """Cancel an active order by its ID."""
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return False

        cancelled_qty = order.remaining_qty
        order.is_active = False
        order.remaining_qty = 0

        # Update running total volumes
        if order.side == Side.BID:
            self.total_bid_volume -= cancelled_qty
        else:
            self.total_ask_volume -= cancelled_qty

        price = order.price
        if order.side == Side.BID:
            queue = self._bids.get(price)
        else:
            queue = self._asks.get(price)

        if queue is not None:
            try:
                queue.remove(order)
            except ValueError:
                pass

            if len(queue) == 0:
                self._remove_price_level(order.side, price)
                if order.side == Side.BID:
                    self._clean_bid_heap()
                else:
                    self._clean_ask_heap()

        self.events.append(OrderEvent(
            timestamp=timestamp,
            event_type="CANCEL",
            order_id=order_id,
            side=order.side.value,
            price=price,
            quantity=cancelled_qty,
        ))
        self.cancels.append(CancelRecord(
            timestamp=timestamp,
            order_id=order_id,
            side=order.side,
            price=price,
            cancelled_qty=cancelled_qty,
        ))

        return True

    def modify_order(
        self,
        order_id: int,
        new_qty: int,
        timestamp: float = 0.0,
    ) -> bool:
        """Modify the quantity of an active order."""
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return False
        if new_qty <= 0:
            return self.cancel_order(order_id, timestamp)

        already_filled = order.quantity - order.remaining_qty
        new_remaining = new_qty - already_filled
        if new_remaining <= 0:
            return self.cancel_order(order_id, timestamp)

        if new_qty > order.quantity:
            price = order.price
            side = order.side
            self.cancel_order(order_id, timestamp)
            self.add_limit_order(side, price, new_remaining, timestamp)
        else:
            diff = order.remaining_qty - new_remaining
            order.quantity = new_qty
            order.remaining_qty = new_remaining
            # Update running volumes
            if order.side == Side.BID:
                self.total_bid_volume -= diff
            else:
                self.total_ask_volume -= diff

        self.events.append(OrderEvent(
            timestamp=timestamp,
            event_type="MODIFY",
            order_id=order_id,
            side=order.side.value,
            price=order.price,
            quantity=new_qty,
        ))

        return True

    def get_best_bid(self) -> Optional[float]:
        """Get the current best bid price."""
        self._clean_bid_heap()
        if not self._bid_heap:
            return None
        return -self._bid_heap[0]

    def get_best_ask(self) -> Optional[float]:
        """Get the current best ask price."""
        self._clean_ask_heap()
        if not self._ask_heap:
            return None
        return self._ask_heap[0]

    def get_mid_price(self) -> Optional[float]:
        """Compute the mid-price."""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def get_spread(self) -> Optional[float]:
        """Compute the bid-ask spread."""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def get_snapshot(self, levels: int = 5) -> Dict[str, Any]:
        """Capture a snapshot of the top N price levels on each side."""
        # Bid side: sorted descending by price
        bid_prices_sorted = sorted(self._bid_prices, reverse=True)
        bid_prices_out: List[float] = []
        bid_qtys_out: List[int] = []
        bid_order_counts: List[int] = []

        for price in bid_prices_sorted[:levels]:
            queue = self._bids.get(price)
            if queue:
                level_qty = sum(o.remaining_qty for o in queue)
                bid_prices_out.append(price)
                bid_qtys_out.append(level_qty)
                bid_order_counts.append(len(queue))

        # Ask side: sorted ascending by price
        ask_prices_sorted = sorted(self._ask_prices)
        ask_prices_out: List[float] = []
        ask_qtys_out: List[int] = []
        ask_order_counts: List[int] = []

        for price in ask_prices_sorted[:levels]:
            queue = self._asks.get(price)
            if queue:
                level_qty = sum(o.remaining_qty for o in queue)
                ask_prices_out.append(price)
                ask_qtys_out.append(level_qty)
                ask_order_counts.append(len(queue))

        return {
            "symbol": self.symbol,
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "bid_prices": bid_prices_out,
            "bid_quantities": bid_qtys_out,
            "bid_order_counts": bid_order_counts,
            "ask_prices": ask_prices_out,
            "ask_quantities": ask_qtys_out,
            "ask_order_counts": ask_order_counts,
            "total_bid_volume": self.total_bid_volume,
            "total_ask_volume": self.total_ask_volume,
        }

    def get_total_orders(self) -> int:
        """Return the total number of active orders in the book."""
        count = 0
        for queue in self._bids.values():
            count += len(queue)
        for queue in self._asks.values():
            count += len(queue)
        return count

    def __repr__(self) -> str:
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        spread = self.get_spread()
        return (
            f"OrderBook({self.symbol}, "
            f"bid={bb}, ask={ba}, spread={spread}, "
            f"trades={len(self.trades)}, "
            f"active_orders={self.get_total_orders()})"
        )
