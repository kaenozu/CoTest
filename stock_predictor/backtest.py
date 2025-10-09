"""モデル予測に基づくシンプルトレーディング戦略のバックテスト."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Callable, Dict, Iterable, List, Sequence

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
class Position:
    """オープンポジションを表現するデータ構造."""

    direction: str
    total_quantity: float
    remaining_quantity: float
    entry_index: int
    target_exit_index: int
    entry_price: float
    entry_timestamp: datetime
    predicted_return: float
    ticker: str | None = None

    def reduce(self, quantity: float) -> None:
        self.remaining_quantity = max(self.remaining_quantity - quantity, 0.0)


ExitDecision = Dict[str, object]

EPSILON = 1e-9


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
    max_drawdown_limit: float | None = None,
    max_open_positions: int = 1,
    allocation_method: str = "fixed-fraction",
    exit_horizons: Dict[str, int] | None = None,
    position_size_adjuster: Callable[[float, Dict[str, object]], float] | None = None,
    exit_condition: Callable[[Position, PriceRow, int], object | None] | None = None,
) -> dict[str, object]:
    """予測値から売買シグナルを生成し、柔軟なポジション管理で損益を集計する."""

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
    if max_open_positions < 1:
        raise ValueError("max_open_positions は1以上で指定してください")

    normalized_allocation = allocation_method.replace("_", "-").lower()
    if normalized_allocation not in {"fixed-fraction", "equal-weight"}:
        raise ValueError("allocation_method には 'fixed-fraction' または 'equal-weight' を指定してください")

    normalized_exit_horizons: Dict[str, int] = {}
    if exit_horizons:
        for key, value in exit_horizons.items():
            horizon = int(value)
            if horizon < 1:
                raise ValueError("exit_horizons の値は1以上で指定してください")
            normalized_exit_horizons[key.lower()] = horizon

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
    open_positions: List[Position] = []

    halt_due_to_drawdown = False
    halted = False
    halt_reason: str | None = None

    def apply_slippage(price: float, direction: str, side: str) -> float:
        if side == "entry":
            return price * (1 + slippage) if direction == "long" else price * (1 - slippage)
        return price * (1 - slippage) if direction == "long" else price * (1 + slippage)

    def normalize_exit_decisions(raw: object | None) -> List[ExitDecision]:
        if raw is None:
            return []
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return [{"close_fraction": float(raw)}]
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, (list, tuple)):
            decisions: List[ExitDecision] = []
            for item in raw:
                if item is None:
                    continue
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    decisions.append({"close_fraction": float(item)})
                elif isinstance(item, dict):
                    decisions.append(item)
                else:
                    raise TypeError("exit_condition は dict または数値のリストを返す必要があります")
            return decisions
        raise TypeError("exit_condition は dict/数値/リストのいずれかを返す必要があります")

    def register_trade(
        position: Position,
        quantity: float,
        exit_index: int,
        exit_timestamp: datetime,
        exit_price: float,
        reason: str,
    ) -> None:
        nonlocal balance, total_profit, max_balance, max_drawdown, trades, wins, halt_due_to_drawdown, halted, halt_reason

        quantity = min(quantity, position.remaining_quantity)
        if quantity <= EPSILON:
            return

        entry_price = position.entry_price
        direction = position.direction
        entry_value = entry_price * quantity
        exit_value = exit_price * quantity

        if direction == "long":
            profit = exit_value - entry_value
            pnl = profit - fee_rate * (entry_value + exit_value)
        else:
            profit = entry_value - exit_value
            pnl = profit - fee_rate * (entry_value + exit_value)

        balance += pnl
        total_profit = balance - initial_capital
        balance_history.append(balance)
        if balance > max_balance:
            max_balance = balance
        drawdown = (max_balance - balance) / max_balance if max_balance else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if max_drawdown_limit is not None and drawdown > max_drawdown_limit:
            halt_due_to_drawdown = True
            halted = True
            if halt_reason is None:
                halt_reason = "max_drawdown_limit"

        if entry_value != 0:
            realized_return = profit / entry_value
        else:
            realized_return = 0.0

        position.reduce(quantity)

        trade_record = {
            "direction": direction,
            "action": "buy" if direction == "long" else "sell",
            "quantity": float(quantity),
            "entry_quantity": float(position.total_quantity),
            "remaining_quantity": float(position.remaining_quantity),
            "predicted_return": position.predicted_return,
            "date": position.entry_timestamp.date(),
            "ticker": position.ticker,
            "entry": {
                "timestamp": position.entry_timestamp,
                "price": float(entry_price),
                "index": position.entry_index,
                "predicted_return": position.predicted_return,
            },
            "exit": {
                "timestamp": exit_timestamp,
                "price": float(exit_price),
                "index": exit_index,
                "actual_return": realized_return,
            },
            "holding_period": max(exit_index - position.entry_index, 0),
            "profit": float(profit),
            "return": realized_return,
            "pnl": float(pnl),
            "fees": fee_rate * (entry_value + exit_value),
            "balance_after_trade": balance,
            "exit_reason": reason,
        }

        signals.append(trade_record)
        trades += 1
        if profit > 0:
            wins += 1

    def close_all_positions(
        exit_index: int, timestamp: datetime, market_price: float, reason: str
    ) -> None:
        nonlocal open_positions
        remaining = open_positions
        open_positions = []
        for position in remaining:
            exit_price = apply_slippage(market_price, position.direction, "exit")
            register_trade(position, position.remaining_quantity, exit_index, timestamp, exit_price, reason)

    for idx, predicted_close in enumerate(predictions):
        if idx >= len(dataset.sample_indices):
            break

        row_index = dataset.sample_indices[idx]
        current_row = sorted_rows[row_index]
        current_timestamp = _to_datetime(current_row["Date"])
        current_close_price = float(current_row["Close"])
        ticker = current_row.get("Ticker") if isinstance(current_row, dict) else None

        # 期限到達による決済
        remaining_positions: List[Position] = []
        matured_positions: List[Position] = []
        for position in open_positions:
            if row_index >= position.target_exit_index:
                matured_positions.append(position)
            else:
                remaining_positions.append(position)
        open_positions = remaining_positions

        for position in matured_positions:
            exit_index = min(position.target_exit_index, len(sorted_rows) - 1)
            exit_row = sorted_rows[exit_index]
            exit_timestamp = _to_datetime(exit_row["Date"])
            exit_price = apply_slippage(float(exit_row["Close"]), position.direction, "exit")
            register_trade(position, position.remaining_quantity, exit_index, exit_timestamp, exit_price, "horizon")

        if halt_due_to_drawdown:
            close_all_positions(row_index, current_timestamp, current_close_price, halt_reason or "max_drawdown_limit")
            break

        # カスタムエグジット条件の評価
        if exit_condition:
            updated_positions: List[Position] = []
            for position in open_positions:
                decisions = normalize_exit_decisions(exit_condition(position, current_row, row_index))
                for decision in decisions:
                    new_exit_index = decision.get("new_exit_index")
                    if isinstance(new_exit_index, (int, float)):
                        new_index = int(new_exit_index)
                        if new_index <= row_index:
                            new_index = row_index + 1
                        position.target_exit_index = min(new_index, len(sorted_rows) - 1)

                    fraction = decision.get("close_fraction") or decision.get("fraction")
                    quantity_override = decision.get("close_quantity") or decision.get("quantity")
                    close_quantity: float | None = None
                    if isinstance(quantity_override, (int, float)) and not isinstance(quantity_override, bool):
                        close_quantity = float(quantity_override)
                    elif isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
                        close_quantity = position.remaining_quantity * float(fraction)

                    if close_quantity and close_quantity > EPSILON:
                        raw_exit_price = decision.get("exit_price")
                        if isinstance(raw_exit_price, (int, float)):
                            exit_price = float(raw_exit_price)
                        else:
                            exit_price = apply_slippage(current_close_price, position.direction, "exit")
                        reason = str(decision.get("exit_reason", "custom"))
                        register_trade(position, close_quantity, row_index, current_timestamp, exit_price, reason)
                        if position.remaining_quantity <= EPSILON:
                            break
                if position.remaining_quantity > EPSILON:
                    updated_positions.append(position)
            open_positions = updated_positions

        if halt_due_to_drawdown:
            close_all_positions(row_index, current_timestamp, current_close_price, halt_reason or "max_drawdown_limit")
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

        horizon = normalized_exit_horizons.get(direction, forecast_horizon)
        if horizon < 1:
            continue

        target_exit_index = row_index + horizon
        if target_exit_index >= len(sorted_rows):
            continue

        entry_price = apply_slippage(current_close_price, direction, "entry")

        if normalized_allocation == "fixed-fraction":
            trade_value = balance * position_fraction
        else:
            remaining_slots = max(max_open_positions - len(open_positions), 1)
            equal_allocation = balance / remaining_slots if remaining_slots else balance
            trade_value = min(equal_allocation, balance * position_fraction)

        trade_value = min(trade_value, balance)
        if trade_value <= EPSILON or entry_price <= 0:
            continue
        if trade_value < entry_price:
            continue

        if position_size_adjuster is not None:
            context = {
                "direction": direction,
                "predicted_return": predicted_return,
                "balance": balance,
                "row": current_row,
                "index": row_index,
                "open_positions": len(open_positions),
                "max_open_positions": max_open_positions,
                "ticker": ticker,
            }
            adjusted_value = position_size_adjuster(trade_value, context)
            if not isinstance(adjusted_value, (int, float)) or isinstance(adjusted_value, bool):
                raise TypeError("position_size_adjuster は数値を返す必要があります")
            if not math.isfinite(adjusted_value) or adjusted_value <= 0:
                continue
            trade_value = float(adjusted_value)
            trade_value = min(trade_value, balance)
            if trade_value <= EPSILON:
                continue
            if trade_value < entry_price:
                continue

        quantity = trade_value / entry_price
        if quantity <= EPSILON:
            continue

        position = Position(
            direction=direction,
            total_quantity=float(quantity),
            remaining_quantity=float(quantity),
            entry_index=row_index,
            target_exit_index=target_exit_index,
            entry_price=float(entry_price),
            entry_timestamp=current_timestamp,
            predicted_return=float(predicted_return),
            ticker=ticker if isinstance(ticker, str) else None,
        )
        open_positions.append(position)

    if not halt_due_to_drawdown:
        for position in open_positions:
            exit_index = min(position.target_exit_index, len(sorted_rows) - 1)
            exit_row = sorted_rows[exit_index]
            exit_timestamp = _to_datetime(exit_row["Date"])
            exit_price = apply_slippage(float(exit_row["Close"]), position.direction, "exit")
            reason = "horizon" if exit_index == position.target_exit_index else "end_of_data"
            register_trade(position, position.remaining_quantity, exit_index, exit_timestamp, exit_price, reason)

    win_rate = wins / trades if trades else 0.0
    ending_balance = balance
    cumulative_return = total_profit / initial_capital if initial_capital else 0.0

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