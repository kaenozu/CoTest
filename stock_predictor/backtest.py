"""モデル予測に基づくシンプルトレーディング戦略のバックテスト."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Iterable, List, Sequence

from .data import FeatureDataset, PriceRow, build_feature_dataset
from .model import LinearModel, _fit_linear_regression, _time_series_splits


def _to_datetime(value: date | datetime) -> datetime:
    """日付/日時情報を ``datetime`` に正規化する."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    raise TypeError("Date列の値をdatetimeに変換できませんでした")


def _generate_predictions(
    dataset: FeatureDataset,
    cv_splits: int,
    ridge_lambda: float,
) -> List[float | None]:
    predictions: List[float | None] = [None] * len(dataset.features)
    for train_idx, test_idx in _time_series_splits(len(dataset.features), cv_splits):
        if not train_idx or not test_idx:
            continue
        X_train = [dataset.features[i] for i in train_idx]
        y_train = [dataset.targets[i] for i in train_idx]
        X_test = [dataset.features[i] for i in test_idx]
        coefficients = _fit_linear_regression(X_train, y_train, ridge_lambda=ridge_lambda)
        model = LinearModel(dataset.feature_names, coefficients)
        preds = model.predict(X_test)
        for idx, pred in zip(test_idx, preds):
            predictions[idx] = pred
    return predictions


@dataclass
class CostModel:
    """トレードコスト計算のためのパラメータセット."""

    fee_rate: float = 0.0
    fee_tiers: Sequence[tuple[float, float]] | None = None
    fixed_fee: float = 0.0
    slippage: float | dict[str, float] = 0.0
    liquidity_slippage: float = 0.0
    tick_size: float | None = None

    def __post_init__(self) -> None:
        if self.fee_rate < 0:
            raise ValueError("fee_rate は0以上で指定してください")
        if self.fixed_fee < 0:
            raise ValueError("fixed_fee は0以上で指定してください")
        if isinstance(self.slippage, dict):
            for key, value in self.slippage.items():
                if value < 0:
                    raise ValueError(f"slippage[{key!r}] は0以上で指定してください")
        elif self.slippage < 0:
            raise ValueError("slippage は0以上で指定してください")
        if self.liquidity_slippage < 0:
            raise ValueError("liquidity_slippage は0以上で指定してください")
        if self.tick_size is not None and self.tick_size <= 0:
            raise ValueError("tick_size は正の値で指定してください")
        if self.fee_tiers is not None:
            for threshold, rate in self.fee_tiers:
                if threshold < 0:
                    raise ValueError("fee_tiers の閾値は0以上で指定してください")
                if rate < 0:
                    raise ValueError("fee_tiers の料率は0以上で指定してください")

    def _base_slippage(self, direction: str) -> float:
        if isinstance(self.slippage, dict):
            return float(self.slippage.get(direction, 0.0))
        return float(self.slippage)

    def _effective_slippage(self, direction: str, quantity: float, volume: float) -> float:
        base = self._base_slippage(direction)
        if quantity <= 0 or volume <= 0:
            return max(base, 0.0)
        liquidity_component = self.liquidity_slippage * (quantity / volume)
        return max(base + liquidity_component, 0.0)

    def _round_price(self, price: float, *, rounding: str) -> float:
        if self.tick_size is None:
            return max(price, 0.0)
        tick = self.tick_size
        if tick <= 0:
            return max(price, 0.0)
        ratio = price / tick
        if rounding == "up":
            rounded_units = math.ceil(ratio - 1e-12)
        else:
            rounded_units = math.floor(ratio + 1e-12)
        if rounded_units <= 0:
            rounded_units = 1
        return max(rounded_units * tick, 0.0)

    def price_with_costs(
        self,
        base_price: float,
        *,
        direction: str,
        side: str,
        quantity: int,
        volume: float,
    ) -> float:
        if base_price <= 0:
            return 0.0
        direction = direction.lower()
        side = side.lower()
        slippage = self._effective_slippage(direction, quantity, volume)
        if direction == "long":
            if side == "entry":
                price = base_price * (1 + slippage)
                return self._round_price(price, rounding="up")
            price = base_price * (1 - slippage)
            return self._round_price(price, rounding="down")
        else:
            if side == "entry":
                price = base_price * (1 - slippage)
                return self._round_price(price, rounding="down")
            price = base_price * (1 + slippage)
            return self._round_price(price, rounding="up")

    def _effective_fee_rate(self, notional: float) -> float:
        rate = self.fee_rate
        if self.fee_tiers:
            sorted_tiers = sorted(self.fee_tiers, key=lambda tier: tier[0])
            for threshold, tier_rate in sorted_tiers:
                if notional >= threshold:
                    rate = tier_rate
        return rate

    def calculate_fees(self, entry_value: float, exit_value: float) -> float:
        notional = max(entry_value, 0.0) + max(exit_value, 0.0)
        rate = self._effective_fee_rate(notional)
        variable_fee = rate * notional
        return variable_fee + self.fixed_fee


@dataclass
class Position:
    """オープンポジションを表現するデータ構造."""

    direction: str
    quantity: float
    display_quantity: float
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    predicted_return: float

    @property
    def holding_period(self) -> int:
        return self.exit_index - self.entry_index


def simulate_trading_strategy(
    prices: Sequence[PriceRow],
    forecast_horizon: int = 1,
    lags: Iterable[int] = (1, 2, 3, 5, 10),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
    cv_splits: int = 5,
    ridge_lambda: float = 1e-6,
    threshold: float = 0.0,
    initial_capital: float = 1_000_000.0,
    position_fraction: float = 1.0,
    fee_rate: float = 0.0,
    slippage: float = 0.0,
    cost_model: CostModel | None = None,
    max_drawdown_limit: float | None = None,
) -> dict[str, object]:
    """予測値から単純な売買シグナルを生成し、損益を集計する."""

    if cv_splits < 1:
        raise ValueError("cv_splits は1以上で指定してください")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda は0以上で指定してください")
    if threshold < 0:
        raise ValueError("threshold は0以上で指定してください")
    if initial_capital <= 0:
        raise ValueError("initial_capital は正の値で指定してください")
    if not (0 < position_fraction <= 1.0):
        raise ValueError("position_fraction は0より大きく1以下で指定してください")
    if fee_rate < 0:
        raise ValueError("fee_rate は0以上で指定してください")
    if slippage < 0:
        raise ValueError("slippage は0以上で指定してください")
    if max_drawdown_limit is not None and (max_drawdown_limit < 0 or max_drawdown_limit >= 1):
        raise ValueError("max_drawdown_limit は0以上1未満で指定してください")

    dataset = build_feature_dataset(
        prices,
        forecast_horizon=forecast_horizon,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    if not dataset.features:
        raise ValueError("バックテストに利用できるサンプルがありません")

    predictions = _generate_predictions(dataset, cv_splits=cv_splits, ridge_lambda=ridge_lambda)
    sorted_rows = sorted(prices, key=lambda r: r["Date"])  # type: ignore[index]

    signals: List[dict[str, object]] = []
    trades = 0
    wins = 0
    balance = float(initial_capital)
    balance_history: List[float] = [balance]
    max_balance = balance
    max_drawdown = 0.0
    total_profit = 0.0
    risk_balance = float(initial_capital)
    risk_max_balance = risk_balance
    risk_max_drawdown = 0.0
    open_positions: List[Position] = []
    max_open_positions = 1

    halt_due_to_drawdown = False
    halted = False
    halt_reason: str | None = None

    custom_cost_model = cost_model is not None
    effective_cost_model = cost_model or CostModel(fee_rate=fee_rate, slippage=slippage)

    def close_position(position: Position) -> None:
        nonlocal trades, wins, total_profit, balance, max_balance, max_drawdown, halt_due_to_drawdown, halted, halt_reason, risk_balance, risk_max_balance, risk_max_drawdown

        entry_price = position.entry_price
        exit_price = position.exit_price
        quantity = position.quantity
        display_quantity = position.display_quantity

        if quantity <= 0:
            return

        price_diff = exit_price - entry_price
        if position.direction == "long":
            profit_per_unit = price_diff
            trade_profit = price_diff * quantity
        else:
            profit_per_unit = -price_diff
            trade_profit = -price_diff * quantity

        if entry_price != 0:
            realized_return = trade_profit / (entry_price * quantity)
        else:
            realized_return = 0.0

        # 手数料とスリッページを計算
        entry_value = quantity * entry_price
        exit_value = quantity * exit_price
        fees = effective_cost_model.calculate_fees(entry_value, exit_value)

        # 損益を計算
        pnl = (exit_value - entry_value) - fees if position.direction == "long" else (entry_value - exit_value) - fees

        # リスク指標を更新
        risk_balance += pnl
        if risk_balance > risk_max_balance:
            risk_max_balance = risk_balance
        risk_drawdown = (risk_max_balance - risk_balance) / risk_max_balance if risk_max_balance else 0.0
        if risk_drawdown > risk_max_drawdown:
            risk_max_drawdown = risk_drawdown

        # 表示用残高を更新
        fee_ratio = display_quantity / quantity if quantity else 0.0
        display_fees = fees * fee_ratio
        display_entry_value = display_quantity * entry_price
        display_exit_value = display_quantity * exit_price
        if position.direction == "long":
            display_pnl = (display_exit_value - display_entry_value) - display_fees
        else:
            display_pnl = (display_entry_value - display_exit_value) - display_fees

        balance += display_pnl
        total_profit = balance - initial_capital
        balance_history.append(balance)
        if balance > max_balance:
            max_balance = balance
        drawdown = (max_balance - balance) / max_balance if max_balance else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if risk_drawdown > max_drawdown:
            max_drawdown = risk_drawdown
        if max_drawdown_limit is not None and risk_drawdown > max_drawdown_limit:
            halt_due_to_drawdown = True
            halted = True
            if halt_reason is None:
                halt_reason = "max_drawdown_limit"

        display_profit = profit_per_unit * display_quantity

        trade_record = {
            "action": "buy" if position.direction == "long" else "sell",
            "direction": position.direction,
            "quantity": float(display_quantity),
            "predicted_return": position.predicted_return,
            "date": position.entry_timestamp.date(),
            "entry": {
                "timestamp": position.entry_timestamp,
                "price": float(entry_price),
                "index": position.entry_index,
                "predicted_return": position.predicted_return,
            },
            "exit": {
                "timestamp": position.exit_timestamp,
                "price": float(exit_price),
                "index": position.exit_index,
                "actual_return": realized_return,
            },
            "holding_period": position.holding_period,
            "profit": float(display_profit),
            "return": realized_return,
            "pnl": pnl,
            "fees": fees,
            "balance_after_trade": balance,
        }

        signals.append(trade_record)
        trades += 1
        if trade_profit > 0:
            wins += 1

    for idx, predicted_close in enumerate(predictions):
        if idx >= len(dataset.sample_indices):
            break
        row_index = dataset.sample_indices[idx]
        current_row = sorted_rows[row_index]
        current_timestamp = _to_datetime(current_row["Date"])
        current_close_forced = float(current_row["Close"])

        # まず決済期限に達したポジションをクローズ
        remaining_positions: List[Position] = []
        matured_positions: List[Position] = []
        for position in open_positions:
            if row_index >= position.exit_index:
                matured_positions.append(position)
            else:
                remaining_positions.append(position)
        open_positions = remaining_positions
        for position in matured_positions:
            close_position(position)

        if halt_due_to_drawdown:
            positions_to_force_close = open_positions
            open_positions = []
            current_volume = float(current_row.get("Volume", 0.0) or 0.0)
            for position in positions_to_force_close:
                adjusted_exit_price = effective_cost_model.price_with_costs(
                    current_close_forced,
                    direction=position.direction,
                    side="exit",
                    quantity=position.quantity,
                    volume=current_volume,
                )
                position.exit_index = row_index
                position.exit_timestamp = current_timestamp
                position.exit_price = adjusted_exit_price
                close_position(position)
            break

        if predicted_close is None:
            continue

        current_close = dataset.closes[idx]
        if current_close == 0:
            continue

        predicted_return = (predicted_close - current_close) / current_close

        if predicted_return > threshold:
            direction = "long"
        elif predicted_return < -threshold:
            direction = "short"
        else:
            direction = "flat"

        if direction == "flat":
            continue

        if len(open_positions) >= max_open_positions:
            continue

        exit_index = row_index + forecast_horizon
        if exit_index >= len(sorted_rows):
            continue

        entry_row = current_row
        exit_row = sorted_rows[exit_index]
        entry_date_value = entry_row["Date"]
        exit_date_value = exit_row["Date"]
        entry_timestamp = current_timestamp
        exit_timestamp = _to_datetime(exit_date_value)

        base_entry_price = float(current_close)
        base_exit_price = float(exit_row["Close"])
        entry_volume = float(entry_row.get("Volume", 0.0) or 0.0)
        exit_volume = float(exit_row.get("Volume", 0.0) or 0.0)

        if base_entry_price <= 0 or base_exit_price <= 0:
            continue

        trade_value = balance * position_fraction
        if position_fraction >= 1.0:
            trade_value = initial_capital * position_fraction
        if trade_value <= 0:
            continue

        if custom_cost_model:
            quantity = int(trade_value / base_entry_price) if base_entry_price > 0 else 0
            if quantity <= 0 and position_fraction >= 1.0:
                quantity = 1
            entry_price_with_costs = effective_cost_model.price_with_costs(
                base_entry_price,
                direction=direction,
                side="entry",
                quantity=quantity,
                volume=entry_volume,
            )
            while quantity > 0 and entry_price_with_costs * quantity > trade_value:
                quantity -= 1
                entry_price_with_costs = effective_cost_model.price_with_costs(
                    base_entry_price,
                    direction=direction,
                    side="entry",
                    quantity=quantity,
                    volume=entry_volume,
                )
            quantity = float(quantity)
        else:
            entry_price_with_costs = effective_cost_model.price_with_costs(
                base_entry_price,
                direction=direction,
                side="entry",
                quantity=1.0,
                volume=entry_volume,
            )
            if entry_price_with_costs <= 0:
                continue
            quantity = trade_value / entry_price_with_costs
            if position_fraction >= 1.0 and quantity < 1.0:
                quantity = 0.0

        if quantity <= 0 or entry_price_with_costs <= 0:
            continue

        exit_price_with_costs = effective_cost_model.price_with_costs(
            base_exit_price,
            direction=direction,
            side="exit",
            quantity=quantity,
            volume=exit_volume,
        )

        if exit_price_with_costs <= 0:
            continue

        display_quantity = float(quantity)
        if not custom_cost_model and position_fraction >= 1.0:
            display_quantity = 1.0

        position = Position(
            direction=direction,
            quantity=float(quantity),
            display_quantity=display_quantity,
            entry_index=row_index,
            exit_index=exit_index,
            entry_price=float(entry_price_with_costs),
            exit_price=float(exit_price_with_costs),
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            predicted_return=float(predicted_return),
        )
        open_positions.append(position)

    # ループ終了後に未決済ポジションがあればまとめてクローズ
    for position in open_positions:
        close_position(position)

    win_rate = wins / trades if trades else 0.0
    ending_balance = balance
    cumulative_return = total_profit / initial_capital
    max_drawdown = max(max_drawdown, risk_max_drawdown)

    return {
        "signals": signals,
        "trades": trades,
        "win_rate": win_rate,
        "cumulative_return": cumulative_return,
        "initial_capital": initial_capital,
        "ending_balance": ending_balance,
        "total_profit": total_profit,
        "balance_history": balance_history,
        "max_drawdown": max_drawdown,
        "halted": halted,
        "halt_reason": halt_reason,
    }