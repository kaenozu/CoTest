from datetime import date, datetime, timedelta

import math

import pytest

from stock_predictor import backtest
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

    assert result["trades"] >= 0
    assert 0.0 <= result["win_rate"] <= 1.0
    assert isinstance(result["cumulative_return"], float)
    assert "signals" in result
    assert "initial_capital" in result
    assert "ending_balance" in result
    assert "max_drawdown" in result
    assert isinstance(result["balance_history"], list)

    signals = result["signals"]
    assert isinstance(signals, list)
    assert result["trades"] == len(signals)

    if signals:
        trade = signals[0]
        # プルリクエスト #16 のブランチの内容
        assert trade["direction"] in {"long", "short"}
        assert trade["quantity"] == 1
        assert "entry" in trade and "exit" in trade
        assert isinstance(trade["entry"]["timestamp"], datetime)
        assert isinstance(trade["exit"]["timestamp"], datetime)
        assert isinstance(trade["entry"]["price"], float)
        assert isinstance(trade["exit"]["price"], float)
        assert isinstance(trade["profit"], float)
        assert math.isclose(trade["quantity"], 1)
        # メインブランチの最新の状態の内容
        assert trade["action"] in {"buy", "sell", "hold"}
        assert "predicted_return" in trade


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


def test_simulate_trading_strategy_accounts_for_position_and_fees(monkeypatch):
    prices = generate_prices(days=12)

    from stock_predictor.data import build_feature_dataset

    dataset = build_feature_dataset(
        prices,
        forecast_horizon=1,
        lags=(1,),
        rolling_windows=(3,),
    )

    def fake_generate_predictions(dataset_arg, cv_splits, ridge_lambda):
        assert len(dataset_arg.closes) == len(dataset.closes)
        predictions = []
        for idx, close in enumerate(dataset_arg.closes):
            if idx == 0:
                predictions.append(close * 1.02)
            elif idx == 1:
                predictions.append(close * 0.98)
            else:
                predictions.append(close)
        return predictions

    monkeypatch.setattr(
        "stock_predictor.backtest._generate_predictions",
        fake_generate_predictions,
    )

    result = simulate_trading_strategy(
        prices,
        forecast_horizon=1,
        lags=(1,),
        rolling_windows=(3,),
        cv_splits=3,
        threshold=0.005,
        initial_capital=1000.0,
        position_fraction=0.5,
        fee_rate=0.001,
        slippage=0.002,
    )

    trade_signals = [s for s in result["signals"] if s["action"] != "hold"]

    assert len(trade_signals) == 2
    assert result["trades"] == 2
    assert result["win_rate"] == pytest.approx(0.5)

    first_trade, second_trade = trade_signals

    assert first_trade["action"] == "buy"
    assert first_trade["quantity"] == pytest.approx(4.7523999619808)
    assert first_trade["pnl"] == pytest.approx(1.744140290846881)
    assert first_trade["fees"] == pytest.approx(1.002746887178025)

    assert second_trade["action"] == "sell"
    assert second_trade["quantity"] == pytest.approx(4.734677564047184)
    assert second_trade["pnl"] == pytest.approx(-7.760145996828484)

    assert result["initial_capital"] == pytest.approx(1000.0)
    assert result["ending_balance"] == pytest.approx(993.9839942940184)
    assert result["total_profit"] == pytest.approx(-6.016005705981575)
    assert result["balance_history"] == pytest.approx(
        [1000.0, 1001.7441402908469, 993.9839942940184]
    )
    assert result["max_drawdown"] == pytest.approx(0.007746634778991926)