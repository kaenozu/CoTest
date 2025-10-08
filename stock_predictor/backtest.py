"""モデル予測に基づくシンプルトレーディング戦略のバックテスト."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .data import FeatureDataset, PriceRow, build_feature_dataset
from .model import LinearModel, _fit_linear_regression, _time_series_splits


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

    for idx, predicted_close in enumerate(predictions):
        if predicted_close is None:
            continue
        current_close = dataset.closes[idx]
        if current_close == 0:
            continue
        actual_close = dataset.targets[idx]
        predicted_return = (predicted_close - current_close) / current_close
        actual_return = (actual_close - current_close) / current_close

        if predicted_return > threshold:
            action = "buy"
            profit = actual_return
        elif predicted_return < -threshold:
            action = "sell"
            profit = -actual_return
        else:
            action = "hold"
            profit = 0.0

        if action != "hold":
            trades += 1
            if profit > 0:
                wins += 1
            cumulative_return += profit

        row_index = dataset.sample_indices[idx]
        signal = {
            "date": sorted_rows[row_index]["Date"],
            "action": action,
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "profit": profit,
        }
        signals.append(signal)

    win_rate = wins / trades if trades else 0.0

    return {
        "signals": signals,
        "trades": trades,
        "win_rate": win_rate,
        "cumulative_return": cumulative_return,
    }
