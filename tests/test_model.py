from datetime import date, timedelta

import pytest

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

    assert set(result.keys()) == {"model", "mae", "rmse", "cv_score"}
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["cv_score"] >= 0


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