import warnings
from datetime import date

import pytest

from types import SimpleNamespace

from stock_predictor.data import (
    build_feature_dataset,
    build_feature_matrix,
    fetch_price_data_from_yfinance,
    load_price_data,
    stream_live_prices,
)
import stock_predictor.data as data_module


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


def test_build_feature_dataset_returns_indices_and_closes(sample_prices):
    dataset = build_feature_dataset(
        sample_prices,
        forecast_horizon=1,
        lags=(1,),
        rolling_windows=(3, 5),
    )

    assert dataset.sample_indices == [4]
    assert dataset.closes == [14.5]
    assert len(dataset.features) == len(dataset.targets) == 1


def test_fetch_price_data_from_yfinance(monkeypatch):
    class DummyFrame:
        def __init__(self, rows):
            self._rows = rows

        @property
        def empty(self):
            return len(self._rows) == 0

        def dropna(self):
            filtered = [
                (idx, values)
                for idx, values in self._rows
                if all(value is not None for value in values.values())
            ]
            return DummyFrame(filtered)

        def iterrows(self):
            yield from self._rows

    rows = [
        (
            date(2023, 1, 1),
            {
                "Open": 10.0,
                "High": 11.0,
                "Low": 9.0,
                "Close": 10.5,
                "Adj Close": 10.4,
                "Volume": 1000,
            },
        ),
        (
            date(2023, 1, 2),
            {
                "Open": 11.0,
                "High": 12.0,
                "Low": 10.0,
                "Close": 11.5,
                "Adj Close": 11.4,
                "Volume": 1100,
            },
        ),
        (
            date(2023, 1, 3),
            {
                "Open": 12.0,
                "High": 13.0,
                "Low": 11.0,
                "Close": None,
                "Adj Close": 12.4,
                "Volume": 1200,
            },
        ),
    ]

    frame = DummyFrame(rows)

    def fake_download(*args, **kwargs):
        assert kwargs["period"] == "30d"
        assert kwargs["interval"] == "1d"
        return frame

    monkeypatch.setattr(
        data_module,
        "yfinance",
        SimpleNamespace(download=fake_download),
    )

    data = fetch_price_data_from_yfinance("AAPL", period="30d", interval="1d")

    assert len(data) == 2
    assert data[0]["Date"] == date(2023, 1, 1)
    assert data[0]["Close"] == 10.5
    assert data[1]["Volume"] == 1100


def test_fetch_price_data_from_yfinance_handles_series_scalars(monkeypatch):
    import pandas as pd

    class DummyFrame:
        def __init__(self, rows):
            self._rows = rows

        @property
        def empty(self):
            return len(self._rows) == 0

        def dropna(self):
            return DummyFrame(self._rows)

        def iterrows(self):
            for idx, values in self._rows:
                yield idx, values.copy()

    rows = [
        (
            date(2024, 1, 1),
            pd.Series(
                {
                    "Open": pd.Series([10.0]),
                    "High": pd.Series([11.0]),
                    "Low": pd.Series([9.0]),
                    "Close": pd.Series([10.5]),
                    "Adj Close": pd.Series([10.4]),
                    "Volume": pd.Series([1000.0]),
                }
            ),
        )
    ]

    frame = DummyFrame(rows)

    def fake_download(*args, **kwargs):
        return frame

    monkeypatch.setattr(
        data_module,
        "yfinance",
        SimpleNamespace(download=fake_download),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = fetch_price_data_from_yfinance("AAPL")

    future_warnings = [item for item in caught if issubclass(item.category, FutureWarning)]
    assert not future_warnings
    assert data[0]["Close"] == pytest.approx(10.5)
    assert data[0]["Volume"] == pytest.approx(1000.0)


def test_fetch_price_data_from_yfinance_with_split_adjustments(monkeypatch):
    class DummyFrame:
        def __init__(self, rows):
            self._rows = rows

        @property
        def empty(self):
            return len(self._rows) == 0

        def dropna(self):
            filtered = [
                (idx, values)
                for idx, values in self._rows
                if all(value is not None for value in values.values())
            ]
            return DummyFrame(filtered)

        def iterrows(self):
            yield from self._rows

    base_rows = [
        (
            date(2023, 1, 1),
            {
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 100.0,
                "Adj Close": 50.0,
                "Volume": 1000,
                "Stock Splits": 0.0,
                "Dividends": 0.0,
            },
        ),
        (
            date(2023, 1, 2),
            {
                "Open": 62.0,
                "High": 64.0,
                "Low": 58.0,
                "Close": 60.0,
                "Adj Close": 60.0,
                "Volume": 1200,
                "Stock Splits": 2.0,
                "Dividends": 0.0,
            },
        ),
        (
            date(2023, 1, 3),
            {
                "Open": 61.5,
                "High": 63.0,
                "Low": 59.0,
                "Close": 61.0,
                "Adj Close": 60.5,
                "Volume": 900,
                "Stock Splits": 0.0,
                "Dividends": 0.5,
            },
        ),
    ]

    adjusted_rows = [
        (
            date(2023, 1, 1),
            {
                "Open": 50.0,
                "High": 55.0,
                "Low": 45.0,
                "Close": 50.0,
                "Adj Close": 50.0,
                "Volume": 1000,
            },
        ),
        (
            date(2023, 1, 2),
            {
                "Open": 62.0,
                "High": 64.0,
                "Low": 58.0,
                "Close": 60.0,
                "Adj Close": 60.0,
                "Volume": 1200,
            },
        ),
        (
            date(2023, 1, 3),
            {
                "Open": 61.5,
                "High": 63.0,
                "Low": 59.0,
                "Close": 60.5,
                "Adj Close": 60.5,
                "Volume": 900,
            },
        ),
    ]

    def fake_download(*args, **kwargs):
        assert kwargs["period"] == "90d"
        assert kwargs["interval"] == "1d"
        auto_adjust = kwargs.get("auto_adjust")
        if auto_adjust:
            return DummyFrame(adjusted_rows)
        return DummyFrame(base_rows)

    monkeypatch.setattr(
        data_module,
        "yfinance",
        SimpleNamespace(download=fake_download),
    )

    unadjusted = fetch_price_data_from_yfinance(
        "TEST", period="90d", interval="1d", adjust="none"
    )
    assert unadjusted[0]["Close"] == 100.0
    assert unadjusted[1]["Close"] == 60.0

    manual_adjusted = fetch_price_data_from_yfinance(
        "TEST", period="90d", interval="1d", adjust="manual"
    )
    assert manual_adjusted[0]["Close"] == pytest.approx(50.0)
    assert manual_adjusted[0]["Open"] == pytest.approx(50.0)
    assert manual_adjusted[2]["Close"] == pytest.approx(60.5)

    auto_adjusted = fetch_price_data_from_yfinance(
        "TEST", period="90d", interval="1d", adjust="auto"
    )
    assert auto_adjusted[0]["Close"] == pytest.approx(50.0)
    assert auto_adjusted[2]["Close"] == pytest.approx(60.5)

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


def test_stream_live_prices_normalizes_payload():
    events = [
        {
            "symbol": "AAPL",
            "timestamp": 1,
            "open": 150.0,
            "high": 151.0,
            "low": 149.5,
            "price": 150.8,
            "volume": 1200,
        },
        {
            "symbol": "AAPL",
            "timestamp": 2,
            "price": 152.3,
            "volume": None,
        },
    ]

    class DummyClient:
        def stream_prices(self, ticker):
            assert ticker == "AAPL"
            for payload in events:
                yield payload

    def fake_to_datetime(ordinal):
        return date(2023, 1, ordinal)

    client = DummyClient()
    rows = list(stream_live_prices(client, "AAPL", limit=2, timestamp_converter=fake_to_datetime))

    assert len(rows) == 2
    assert rows[0]["Date"] == date(2023, 1, 1)
    assert rows[0]["Close"] == 150.8
    assert rows[0]["Volume"] == 1200.0
    assert rows[0]["Open"] == 150.0
    assert rows[0]["High"] == 151.0
    assert rows[0]["Low"] == 149.5
    assert rows[1]["Volume"] == 0.0


def test_build_feature_matrix_skips_too_long_lags(sample_prices):
    short_prices = sample_prices[:8]

    X, y, feature_names = build_feature_matrix(
        short_prices, forecast_horizon=1, lags=(1, 2, 10)
    )

    assert len(X) > 0
    assert len(X) == len(y)
    assert "lag_10_close" not in feature_names
    assert "lag_10_return" not in feature_names


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