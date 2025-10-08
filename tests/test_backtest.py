from datetime import date, timedelta

import pytest

from stock_predictor.backtest import simulate_trading_strategy
from stock_predictor.data import build_feature_dataset


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

    assert result["trades"] >= 1
    assert 0.0 <= result["win_rate"] <= 1.0
    assert "signals" in result and result["signals"]
    first_signal = result["signals"][0]
    assert first_signal["action"] in {"buy", "sell", "hold"}
    assert "predicted_return" in first_signal


def test_simulate_trading_strategy_provides_trade_timing():
    prices = generate_prices(days=20)

    result = simulate_trading_strategy(
        prices,
        forecast_horizon=2,
        lags=(1,),
        rolling_windows=(3, 5),
        cv_splits=3,
        threshold=0.0,
    )

    actionable_signals = [s for s in result["signals"] if s["action"] != "hold"]
    assert actionable_signals, "少なくとも1件の売買シグナルが生成される想定"

    signal = actionable_signals[0]
    assert "entry_timestamp" in signal
    assert "exit_timestamp" in signal

    entry = signal["entry_timestamp"]
    exit_ = signal["exit_timestamp"]

    from datetime import datetime

    assert isinstance(entry, datetime)
    assert isinstance(exit_, datetime)
    assert exit_ > entry
    assert (exit_ - entry).days == 2


def test_simulate_trading_strategy_generates_tail_signals():
    prices = generate_prices(days=16)

    dataset = build_feature_dataset(
        prices,
        forecast_horizon=1,
        lags=(1,),
        rolling_windows=(3, 5),
    )

    result = simulate_trading_strategy(
        prices,
        forecast_horizon=1,
        lags=(1,),
        rolling_windows=(3, 5),
        cv_splits=3,
        threshold=0.0,
    )

    assert result["signals"], "シグナルが生成されていること"

    last_signal_date = result["signals"][-1]["date"]
    expected_last_date = prices[dataset.sample_indices[-1]]["Date"]

    assert last_signal_date == expected_last_date
