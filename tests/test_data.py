from datetime import date

import pytest

from stock_predictor.data import build_feature_matrix, load_price_data


def test_load_price_data_sort_and_columns(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-02,10,11,9,10.5,1000\n"
        "2023-01-01,9,10,8,9.5,1500\n"
    )

    loaded = load_price_data(csv_path)

    assert list(loaded[0].keys()) == [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    assert loaded[0]["Date"] == date(2023, 1, 1)


def test_build_feature_matrix_creates_lag_and_returns_targets(sample_prices):
    X, y, feature_names = build_feature_matrix(sample_prices, forecast_horizon=1, lags=(1, 2))

    assert "lag_1_close" in feature_names
    assert "lag_2_close" in feature_names
    assert len(X) == len(y)
    assert len(X[0]) == len(feature_names)


def test_build_feature_matrix_skips_volume_zscore_when_insufficient_data():
    prices = [
        {
            "Date": date(2023, 1, day),
            "Open": 10.0 + day,
            "High": 11.0 + day,
            "Low": 9.0 + day,
            "Close": 10.5 + day,
            "Volume": 1000.0 + day * 10,
        }
        for day in range(1, 5)
    ]

    X, y, feature_names = build_feature_matrix(
        prices,
        forecast_horizon=1,
        lags=(1,),
        rolling_windows=(3,),
    )

    assert "volume_zscore_5" not in feature_names
    assert len(X) > 0
    assert len(X) == len(y)


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
