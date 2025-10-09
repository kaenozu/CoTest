"""モデル予測に基づくシンプルトレーディング戦略のバックテスト."""

from __future__ import annotations

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
class Position:
    """オープンポジションを表現するデータ構造."""

    direction: str
    quantity: int
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
    open_positions: List[Position] = []
    max_open_positions = 1

    halt_due_to_drawdown = False
    halted = False
    halt_reason: str | None = None

    def close_position(position: Position) -> None:
        nonlocal trades, wins, total_profit, balance, max_balance, max_drawdown, halt_due_to_drawdown, halted, halt_reason

        entry_price = position.entry_price
        exit_price = position.exit_price
        quantity = position.quantity

        if quantity <= 0:
            return

        price_diff = exit_price - entry_price
        if position.direction == "long":
            profit = price_diff * quantity
        else:
            profit = -price_diff * quantity

        if entry_price != 0:
            realized_return = profit / (entry_price * quantity)
        else:
            realized_return = 0.0

        # 手数料とスリッページを計算
        entry_value = quantity * entry_price
        exit_value = quantity * exit_price
        fees = fee_rate * (entry_value + exit_value)

        # 損益を計算
        pnl = (exit_value - entry_value) - fees if position.direction == "long" else (entry_value - exit_value) - fees

        # 残高を更新
        balance += pnl
        total_profit = balance - initial_capital
        balance_history.append(balance)
        if balance > max_balance:
            max_balance = balance
        drawdown = (max_balance - balance) / max_balance if max_balance else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if max_drawdown_limit is not None and max_drawdown > max_drawdown_limit:
            halt_due_to_drawdown = True
            halted = True
            if halt_reason is None:
                halt_reason = "max_drawdown_limit"

        trade_record = {
            "direction": position.direction,
            "quantity": quantity,
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
            "profit": float(profit),
            "return": realized_return,
            "pnl": pnl,
            "fees": fees,
            "balance_after_trade": balance,
        }

        signals.append(trade_record)
        trades += 1
        if profit > 0:
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
            for position in positions_to_force_close:
                position.exit_index = row_index
                position.exit_timestamp = current_timestamp
                position.exit_price = current_close_forced
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
        exit_price = float(exit_row["Close"])

        # トレード価格を計算 (スリッページ)
        entry_price_with_slippage = current_close * (1 + slippage) if direction == "long" else current_close * (1 - slippage)

        # トレード数量を計算
        trade_value = balance * position_fraction
        quantity = int(trade_value / entry_price_with_slippage) if entry_price_with_slippage > 0 else 0
        if quantity <= 0:
            continue

        position = Position(
            direction=direction,
            quantity=quantity,
            entry_index=row_index,
            exit_index=exit_index,
            entry_price=float(current_close),
            exit_price=exit_price,
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
