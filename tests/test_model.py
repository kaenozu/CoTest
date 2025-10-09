from datetime import date, timedelta

import pytest

from stock_predictor.data import build_feature_dataset
from stock_predictor.model import _fit_linear_regression, train_and_evaluate


def generate_prices(days: int = 40):
    base_date = date(2023, 1, 1)
    data = []
    price = 100.0
    for i in range(days):
        day = base_date + timedelta(days=i)
        price += 0.8
        data.append(
            {
                "Date": day,
                "Open": price - 0.5,
                "High": price + 0.5,
                "Low": price - 1.0,
                "Close": price,
                "Volume": 1000.0 + i * 10,
            }
        )
    return data


def test_train_and_evaluate_returns_metrics():
    prices = generate_prices()

    result = train_and_evaluate(prices, forecast_horizon=1, lags=(1, 2, 3), cv_splits=3)

    assert set(result.keys()) == {
        "model",
        "mae",
        "rmse",
        "cv_score",
        "forward_mae",
        "forward_rmse",
        "forward_indices",
    }
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["cv_score"] >= 0
    assert result["forward_mae"] >= 0
    assert result["forward_rmse"] >= 0
    assert isinstance(result["forward_indices"], list)
    assert all(isinstance(idx, int) for idx in result["forward_indices"])


def test_ridge_lambda_influences_coefficients():
    prices = generate_prices()

    weak_ridge = train_and_evaluate(
        prices,
        forecast_horizon=1,
        lags=(1, 2, 3),
        cv_splits=3,
        ridge_lambda=1e-6,
    )
    strong_ridge = train_and_evaluate(
        prices,
        forecast_horizon=1,
        lags=(1, 2, 3),
        cv_splits=3,
        ridge_lambda=10.0,
    )

    weak_coeffs = weak_ridge["model"].coefficients
    strong_coeffs = strong_ridge["model"].coefficients

    assert len(weak_coeffs) == len(strong_coeffs)
    assert any(abs(a - b) > 1e-6 for a, b in zip(weak_coeffs, strong_coeffs))


def test_ridge_regularization_does_not_shrink_bias():
    X = [[0.0], [0.0], [0.0], [0.0]]
    y = [2.0, 2.0, 2.0, 2.0]

    coefficients = _fit_linear_regression(X, y, ridge_lambda=10.0)

    assert pytest.approx(2.0, rel=1e-6) == coefficients[0]
    assert all(abs(coef) < 1e-9 for coef in coefficients[1:])


def test_train_and_evaluate_ignores_excessive_lags():
    prices = generate_prices(days=8)

    result = train_and_evaluate(prices, forecast_horizon=1, lags=(1, 2, 10), cv_splits=2)

    assert result["model"].feature_names
    assert "lag_10_close" not in result["model"].feature_names
    assert "lag_10_return" not in result["model"].feature_names


def test_train_and_evaluate_walk_forward_uses_future_window():
    prices = generate_prices(days=50)

    result = train_and_evaluate(prices, forecast_horizon=1, lags=(1, 2, 3), cv_splits=3)

    dataset = build_feature_dataset(prices, forecast_horizon=1, lags=(1, 2, 3))

    forward_indices = result["forward_indices"]

    assert forward_indices, "ウォークフォワード検証の対象サンプルが返される"
    assert all(idx in dataset.sample_indices for idx in forward_indices)
    assert dataset.sample_indices[-len(forward_indices) :] == forward_indices


@pytest.fixture
def sample_prices():
    return [
        {
            "Date": date(2023, 1, 1),
            "Open": 10.0,
            "High": 11.0,
            "Low": 9.0,
            "Close": 10.5,
            "Volume": 1000.0,
        },
        {
            "Date": date(2023, 1, 2),
            "Open": 11.0,
            "High": 12.0,
            "Low": 10.0,
            "Close": 11.5,
            "Volume": 1100.0,
        },
        {
            "Date": date(2023, 1, 3),
            "Open": 12.0,
            "High": 13.0,
            "Low": 11.0,
            "Close": 12.5,
            "Volume": 1200.0,
        },
        {
            "Date": date(2023, 1, 4),
            "Open": 13.0,
            "High": 14.0,
            "Low": 12.0,
            "Close": 13.5,
            "Volume": 1300.0,
        },
        {
            "Date": date(2023, 1, 5),
            "Open": 14.0,
            "High": 15.0,
            "Low": 13.0,
            "Close": 14.5,
            "Volume": 1400.0,
        },
        {
            "Date": date(2023, 1, 6),
            "Open": 15.0,
            "High": 16.0,
            "Low": 14.0,
            "Close": 15.5,
            "Volume": 1500.0,
        },
    ]