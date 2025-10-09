"""学習と評価モジュール."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .data import PriceRow, build_feature_dataset


@dataclass
class LinearModel:
    feature_names: List[str]
    coefficients: List[float]

    def predict_one(self, features: Sequence[float]) -> float:
        return sum(coef * value for coef, value in zip(self.coefficients, [1.0, *features]))

    def predict(self, feature_matrix: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_one(row) for row in feature_matrix]

def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]

def _matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    result: List[List[float]] = []
    for row in A:
        new_row: List[float] = []
        for col in zip(*B):
            new_row.append(sum(r * c for r, c in zip(row, col)))
        result.append(new_row)
    return result

def _vector_matmul(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(m * v for m, v in zip(row, vector)) for row in matrix]

def _gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[max_row][i]) < 1e-12:
            raise ValueError("行列が特異です")
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
        pivot = A[i][i]
        factor = 1.0 / pivot
        A[i] = [value * factor for value in A[i]]
        b[i] *= factor
        for j in range(i + 1, n):
            factor = A[j][i]
            if factor == 0:
                continue
            A[j] = [a - factor * b_ for a, b_ in zip(A[j], A[i])]
            b[j] -= factor * b[i]
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))
    return x

def _fit_linear_regression(
    X: Sequence[Sequence[float]], y: Sequence[float], ridge_lambda: float = 1e-6
) -> List[float]:
    X_with_bias = [[1.0, *row] for row in X]
    Xt = _transpose(X_with_bias)
    XtX = _matmul(Xt, X_with_bias)
    for i in range(len(XtX)):
        if i == 0:
            continue
        XtX[i][i] += ridge_lambda
    Xty = _vector_matmul(Xt, y)
    coefficients = _gaussian_elimination([row[:] for row in XtX], list(Xty))
    return coefficients

def _mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

def _root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))

def _walk_forward_splits(n_samples: int, n_splits: int) -> List[Tuple[List[int], List[int]]]:
    if n_splits < 1 or n_samples < 2:
        return []

    segments = n_splits + 1
    base_size = n_samples // segments
    remainder = n_samples % segments

    if base_size == 0:
        splits: List[Tuple[List[int], List[int]]] = []
        for test_start in range(1, min(n_samples, n_splits + 1)):
            train_indices = list(range(test_start))
            test_indices = [test_start]
            if train_indices and test_indices:
                splits.append((train_indices, test_indices))
        return splits

    boundaries: List[Tuple[int, int]] = []
    start = 0
    for i in range(segments):
        size = base_size + (1 if i < remainder else 0)
        if size <= 0:
            continue
        end = min(start + size, n_samples)
        boundaries.append((start, end))
        start = end

    splits: List[Tuple[List[int], List[int]]] = []
    for i in range(len(boundaries) - 1):
        train_end = boundaries[i][1]
        test_start, test_end = boundaries[i + 1]
        train_indices = list(range(train_end))
        test_indices = list(range(test_start, test_end))
        if train_indices and test_indices:
            splits.append((train_indices, test_indices))
    return splits

def train_and_evaluate(
    prices: Sequence[PriceRow],
    forecast_horizon: int = 1,
    lags: Iterable[int] = (1, 2, 3, 5, 10),
    cv_splits: int = 5,
    ridge_lambda: float = 1e-6,
) -> dict[str, object]:
    """履歴データでモデルを学習し、指標を返す."""
    dataset = build_feature_dataset(
        prices,
        forecast_horizon=forecast_horizon,
        lags=lags,
    )
    X = dataset.features
    y = dataset.targets
    feature_names = dataset.feature_names
    if len(X) == 0:
        raise ValueError("学習に利用できるサンプルがありません")
    if cv_splits < 1:
        raise ValueError("cv_splits は1以上で指定してください")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda は0以上で指定してください")

    mae_scores: List[float] = []
    rmse_scores: List[float] = []
    forward_mae = float("nan")
    forward_rmse = float("nan")
    forward_indices: List[int] = []

    splits = _walk_forward_splits(len(X), cv_splits)

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        if not X_train or not X_test:
            continue
        coefficients = _fit_linear_regression(X_train, y_train, ridge_lambda=ridge_lambda)
        model = LinearModel(feature_names, coefficients)
        predictions = model.predict(X_test)
        mae_scores.append(_mean_absolute_error(y_test, predictions))
        rmse_scores.append(_root_mean_squared_error(y_test, predictions))

        if split_idx == len(splits) - 1:
            forward_mae = mae_scores[-1]
            forward_rmse = rmse_scores[-1]
            forward_indices = [dataset.sample_indices[i] for i in test_idx]

    final_coefficients = _fit_linear_regression(X, y, ridge_lambda=ridge_lambda)
    final_model = LinearModel(feature_names, final_coefficients)
    full_predictions = final_model.predict(X)
    mae = _mean_absolute_error(y, full_predictions)
    rmse = _root_mean_squared_error(y, full_predictions)

    cv_rmse = sum(rmse_scores) / len(rmse_scores) if rmse_scores else rmse

    return {
        "model": final_model,
        "mae": mae,
        "rmse": rmse,
        "cv_score": cv_rmse,
        "forward_mae": forward_mae if forward_mae == forward_mae else mae,
        "forward_rmse": forward_rmse if forward_rmse == forward_rmse else rmse,
        "forward_indices": forward_indices,
    }
