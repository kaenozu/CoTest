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
) -> dict[str, object]:
    """予測値から単純な売買シグナルを生成し、損益を集計する."""

    if cv_splits < 1:
        raise ValueError("cv_splits は1以上で指定してください")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda は0以上で指定してください")
    if threshold < 0:
        raise ValueError("threshold は0以上で指定してください")

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
    cumulative_return = 0.0
    open_positions: List[Position] = []
    max_open_positions = 1

    def close_position(position: Position) -> None:
        nonlocal trades, wins, cumulative_return

        entry_price = position.entry_price
        exit_price = position.exit_price
        quantity = position.quantity

        price_diff = exit_price - entry_price
        if position.direction == "long":
            profit = price_diff * quantity
        else:
            profit = -price_diff * quantity

        if entry_price != 0:
            realized_return = profit / (entry_price * quantity)
        else:
            realized_return = 0.0

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
        }

        signals.append(trade_record)
        trades += 1
        if profit > 0:
            wins += 1
        cumulative_return += profit

    for idx, predicted_close in enumerate(predictions):
        if idx >= len(dataset.sample_indices):
            break
        row_index = dataset.sample_indices[idx]

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

        entry_row = sorted_rows[row_index]
        exit_row = sorted_rows[exit_index]
        entry_date_value = entry_row["Date"]
        exit_date_value = exit_row["Date"]
        entry_timestamp = _to_datetime(entry_date_value)
        exit_timestamp = _to_datetime(exit_date_value)
        exit_price = float(exit_row["Close"])

        position = Position(
            direction=direction,
            quantity=1,
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

    return {
        "signals": signals,
        "trades": trades,
        "win_rate": win_rate,
        "cumulative_return": float(cumulative_return),
    }
