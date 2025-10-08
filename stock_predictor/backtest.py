"""モデル予測に基づくシンプルトレーディング戦略のバックテスト."""

from __future__ import annotations

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

    halt_due_to_drawdown = False

    for idx, predicted_close in enumerate(predictions):
        if predicted_close is None:
            continue
        current_close = dataset.closes[idx]
        if current_close == 0:
            continue
        actual_close = dataset.targets[idx]
        predicted_return = (predicted_close - current_close) / current_close
        actual_return = (actual_close - current_close) / current_close
        entry_price = current_close
        exit_price = actual_close
        quantity = 0.0
        fees = 0.0
        pnl = 0.0
        executed = False

        if predicted_return > threshold:
            action = "buy"
            entry_price = current_close * (1 + slippage)
            exit_price = actual_close * (1 - slippage)
            if entry_price > 0 and balance > 0:
                trade_value = balance * position_fraction
                quantity = trade_value / entry_price
                entry_value = quantity * entry_price
                exit_value = quantity * exit_price
                fees = fee_rate * (entry_value + exit_value)
                pnl = exit_value - entry_value - fees
                executed = quantity > 0
        elif predicted_return < -threshold:
            action = "sell"
            entry_price = current_close * (1 - slippage)
            exit_price = actual_close * (1 + slippage)
            if entry_price > 0 and balance > 0:
                trade_value = balance * position_fraction
                quantity = trade_value / entry_price
                entry_value = quantity * entry_price
                exit_value = quantity * exit_price
                fees = fee_rate * (entry_value + exit_value)
                pnl = (entry_value - exit_value) - fees
                executed = quantity > 0
        else:
            action = "hold"
            pnl = 0.0

        if executed:
            trades += 1
            if pnl > 0:
                wins += 1
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
        else:
            quantity = 0.0
            fees = 0.0 if action == "hold" else fees
            pnl = 0.0 if action == "hold" else pnl

        row_index = dataset.sample_indices[idx]
        entry_date_value = sorted_rows[row_index]["Date"]
        exit_index = row_index + forecast_horizon
        exit_date_value = (
            sorted_rows[exit_index]["Date"] if exit_index < len(sorted_rows) else None
        )

        signal = {
            "date": entry_date_value,
            "action": action,
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "fees": fees,
            "pnl": pnl,
            "executed": executed,
            "entry_timestamp": _to_datetime(entry_date_value),
            "exit_timestamp": _to_datetime(exit_date_value)
            if exit_date_value is not None
            else _to_datetime(entry_date_value),
        }
        if executed:
            signal["balance_after_trade"] = balance
        signals.append(signal)

        if halt_due_to_drawdown:
            break

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
    }
