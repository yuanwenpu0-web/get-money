"""Broker-emulator replay for Simple ZigZag Double (Pine v5).

Indicators and order conditions are evaluated at each bar close. Stop orders
are persistent and first become eligible on the next tick; opposite entries
reverse an existing position. This engine does not apply the application's
generic risk filters, ATR exits, cooldowns, or forced end-of-data liquidation.
"""
from __future__ import annotations

import math

import pandas as pd


def simulate_zigzag(data, signals, params, initial, fee, slippage, interval):
    """Return mark-to-market equity and completed trades for one symbol.

    ``fee`` and ``slippage`` are proportional rates; Pine's default here is
    0.001 commission per side and zero slippage. Price path within each bar
    follows the default TradingView OHLC emulator, without Bar Magnifier.
    ``interval`` is accepted to match the other backtest engine's signature.
    """
    del interval
    if data.empty:
        return pd.Series(dtype=float, index=data.index), []

    rows = list(data.itertuples())
    decisions = list(signals.reindex(data.index).itertuples())
    balance = float(initial)
    position = None
    pending = {}  # order ID -> stop order; orders persist until filled/cancelled
    market_close = False
    values, trades = [], []
    lots = {1: None, -1: None}  # Pine's self-referential series starts at na
    previous_size = 0.0

    def mark(price):
        return balance + ((price - position['entry']) * position['qty'] * position['side'] if position else 0.0)

    def close_at(price, stamp, reason):
        nonlocal balance, position
        p = position
        fill = price * (1 - p['side'] * slippage)
        gross = (fill - p['entry']) * p['qty'] * p['side']
        exit_fee = abs(fill * p['qty']) * fee
        balance += gross - exit_fee
        net = gross - p['fee'] - exit_fee
        margin = abs(p['entry'] * p['qty'])
        trades.append(dict(entry_time=str(p['time']), exit_time=str(stamp),
                           position_id=str(p['time']), side='long' if p['side'] == 1 else 'short',
                           entry_price=p['entry'], exit_price=fill, size=p['qty'], pnl=net,
                           return_pct=net / margin * 100 if margin else 0.0,
                           fees=p['fee'] + exit_fee, status=reason, leverage=1, margin=margin))
        position = None

    def fill_order(order_id, price, stamp):
        nonlocal balance, position
        order = pending.pop(order_id, None)
        if order is None:
            return
        side = order['side']
        if order['kind'] == 'exit':
            if position is not None and position['side'] == -side:
                close_at(price, stamp, 'stop_loss')
            return
        if position is not None and position['side'] == side:
            return  # pyramiding=0
        if position is not None:
            close_at(price, stamp, 'signal_reverse')
        fill = price * (1 + side * slippage)
        # qty=na selects strategy default 100% of equity. An explicit qty is
        # frozen when the order is created/updated, just as in strategy.entry.
        qty = order['qty'] if order['qty'] is not None else max(balance, 0.0) / fill
        if not math.isfinite(qty) or qty <= 0 or fill <= 0:
            return
        entry_fee = abs(fill * qty) * fee
        balance -= entry_fee
        position = dict(side=side, qty=qty, entry=fill, fee=entry_fee, time=stamp)

    def crossed_at(order, price):
        return price >= order['price'] if order['side'] == 1 else price <= order['price']

    def active(order):
        return order['kind'] != 'exit' or (position is not None and position['side'] == -order['side'])

    def process_activated_exit(price, stamp):
        # A bracket attached to an entry becomes executable only after that
        # entry fills. If its stop is already crossed it activates immediately.
        for order_id in list(pending):
            order = pending.get(order_id)
            if order is not None and order['kind'] == 'exit' and active(order) and crossed_at(order, price):
                fill_order(order_id, price, stamp)

    def process_at_open(price, stamp):
        # A crossed or already marketable stop fills at the next available open.
        for order_id in list(pending):
            order = pending.get(order_id)
            if order is not None and active(order) and (order['marketable'] or crossed_at(order, price)):
                fill_order(order_id, price, stamp)
                process_activated_exit(price, stamp)

    def process_segment(start, end, stamp):
        if start == end:
            return
        up = end > start
        while True:
            eligible = [(o['price'], k) for k, o in pending.items()
                        if active(o) and (o['side'] == 1) == up and
                        (start < o['price'] <= end if up else end <= o['price'] < start)]
            if not eligible:
                return
            level, order_id = min(eligible, key=lambda x: x[0]) if up else max(eligible, key=lambda x: x[0])
            fill_order(order_id, level, stamp)
            process_activated_exit(level, stamp)
            start = level
            # Equal-price orders also execute on the same tick, in submission order.
            for other in list(pending):
                order = pending.get(other)
                if order is not None and active(order) and order['side'] == (1 if up else -1) and order['price'] == level:
                    fill_order(other, level, stamp)

    def submit(order_id, kind, side, level, qty, current_close):
        if not math.isfinite(level) or level <= 0:
            return
        if kind == 'entry' and position is not None and position['side'] == side:
            return
        pending[order_id] = dict(kind=kind, side=side, price=level, qty=qty,
                                 marketable=(current_close >= level if side == 1 else current_close <= level))

    for row, signal in zip(rows, decisions):
        stamp = row.Index
        open_price, high, low, close = float(row.open), float(row.high), float(row.low), float(row.close)
        if market_close:
            if position is not None:
                close_at(open_price, stamp, 'time_end')
            market_close = False
        process_at_open(open_price, stamp)
        # TradingView specifies the nearer extreme first; ties are not documented.
        path = [open_price, high, low, close] if abs(open_price - high) < abs(open_price - low) else [open_price, low, high, close]
        for a, b in zip(path, path[1:]):
            process_segment(a, b, stamp)

        size = position['side'] * position['qty'] if position else 0.0
        if params['fixed_lot']:
            lots = {1: float(params['lot_long']), -1: float(params['lot_short'])}
        elif size != previous_size:
            equity = mark(close)
            lots = {1: equity / close * float(params['lot_long']) / 100,
                    -1: equity / close * float(params['lot_short']) / 100}
        previous_size = size

        long_level = float(getattr(signal, 'zz_long_level', float('nan')))
        short_level = float(getattr(signal, 'zz_short_level', float('nan')))
        if getattr(signal, 'zz_long_entry', False):
            submit('Long', 'entry', 1, long_level, lots[1], close)
        if getattr(signal, 'zz_short_entry', False):
            submit('Short', 'entry', -1, short_level, lots[-1], close)
        if not params['use_long']:
            pending.pop('Long', None)
            if (position is not None and position['side'] == -1) or 'Short' in pending:
                submit('Exit Short', 'exit', 1, long_level, None, close)
        if not params['use_short']:
            pending.pop('Short', None)
            if (position is not None and position['side'] == 1) or 'Long' in pending:
                submit('Exit Long', 'exit', -1, short_level, None, close)
        ts = int(pd.Timestamp(stamp).timestamp() * 1000)
        if ts > int(params['timestamp_end']):
            market_close = True
            pending.clear()
        values.append(mark(close))

    equity = pd.Series(values, index=data.index, dtype=float)
    equity.attrs['open_position'] = (
        dict(side='long' if position['side'] == 1 else 'short',
             qty=position['qty'], entry_price=position['entry'],
             entry_time=str(position['time']), entry_fee=position['fee'],
             unrealized_pnl=(float(rows[-1].close) - position['entry']) * position['qty'] * position['side'])
        if position else None
    )
    return equity, trades
