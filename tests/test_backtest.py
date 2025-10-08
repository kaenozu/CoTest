from datetime import date, datetime, timedelta

import math

import pytest

from stock_predictor import backtest
from stock_predictor.backtest import simulate_trading_strategy


def generate_prices(start_price: float = 100.0, days: int = 12, step: float = 1.0):
    base_date = date(2023, 1, 1)
    prices = []
    price = start_price
    for i in range(days):
        price += step
        prices.append(
            {
                "Date": base_date + timedelta(days=i),
                "Open": price - 0.3,
                "High": price + 0.5,
                "Low": price - 0.7,
                "Close": price,
                "Volume": 1000 + i * 10,
            }
        )
    return prices


@pytest.mark.parametrize("threshold", [0.0, 0.002])
def test_simulate_trading_strategy_returns_metrics(threshold):
    prices = generate_prices(days=15)

    result = simulate_trading_strategy(
        prices,
        forecast_horizon=1,
        lags=(1,),
        rolling_windows=(3, 5),
        cv_splits=3,
        threshold=threshold,
    )

    assert result["trades"] >= 0
    assert 0.0 <= result["win_rate"] <= 1.0
    assert isinstance(result["cumulative_return"], float)

    signals = result["signals"]
    assert isinstance(signals, list)
    assert result["trades"] == len(signals)

    if signals:
        trade = signals[0]
        assert trade["direction"] in {"long", "short"}
        assert trade["quantity"] == 1
        assert "entry" in trade and "exit" in trade
        assert isinstance(trade["entry"]["timestamp"], datetime)
        assert isinstance(trade["exit"]["timestamp"], datetime)
        assert isinstance(trade["entry"]["price"], float)
        assert isinstance(trade["exit"]["price"], float)
        assert isinstance(trade["profit"], float)
        assert math.isclose(trade["quantity"], 1)


def test_simulate_trading_strategy_provides_trade_timing(monkeypatch):
    prices = generate_prices(days=15)

    def fake_predictions(dataset, *_, **__):
        values = []
        for close, idx in zip(dataset.closes, dataset.sample_indices):
            if idx in (4, 5):
                values.append(close * 1.05)
            elif idx in (8, 9):
                values.append(close * 0.95)
            else:
                values.append(close)
        return values

    monkeypatch.setattr(backtest, "_generate_predictions", fake_predictions)

    result = simulate_trading_strategy(
        prices,
        forecast_horizon=2,
        lags=(1,),
        rolling_windows=(),
        cv_splits=3,
        threshold=0.0,
    )

    signals = result["signals"]
    assert len(signals) == 2, "エントリー後は決済まで再エントリーしない想定"

    first = signals[0]
    assert first["direction"] == "long"
    assert first["entry"]["timestamp"] < first["exit"]["timestamp"]
    assert (first["exit"]["timestamp"] - first["entry"]["timestamp"]).days == 2
    assert math.isclose(first["profit"], 2.0, rel_tol=1e-6)

    second = signals[1]
    assert second["direction"] == "short"
    assert second["entry"]["timestamp"] >= first["exit"]["timestamp"]
    assert math.isclose(second["profit"], -2.0, rel_tol=1e-6)

    assert result["trades"] == 2
    assert math.isclose(result["win_rate"], 0.5, rel_tol=1e-6)
    assert math.isclose(result["cumulative_return"], 0.0, abs_tol=1e-9)


def test_backtest_limits_open_positions(monkeypatch):
    prices = generate_prices(days=15)

    def fake_predictions(dataset, *_, **__):
        values = []
        for close, idx in zip(dataset.closes, dataset.sample_indices):
            # 連続でシグナルが出るが同時ポジション上限は1
            if idx in (4, 5, 6, 7):
                values.append(close * 1.05)
            else:
                values.append(close)
        return values

    monkeypatch.setattr(backtest, "_generate_predictions", fake_predictions)

    result = simulate_trading_strategy(
        prices,
        forecast_horizon=2,
        lags=(1,),
        rolling_windows=(),
        cv_splits=3,
        threshold=0.0,
    )

    signals = result["signals"]
    assert len(signals) == 2, "同時ポジション上限により同時保有数が制限される"
    assert result["trades"] == 2

    for previous, current in zip(signals, signals[1:]):
        assert current["entry"]["timestamp"] >= previous["exit"]["timestamp"]
