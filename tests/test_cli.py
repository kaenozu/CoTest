from datetime import date

from click.testing import CliRunner

from unittest.mock import Mock

import pytest

from stock_predictor.cli import main


def _dummy_price_rows():
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
    ]


def test_cli_forecast_uses_recommended_ticker_when_missing(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance", fetch_mock
    )

    train_mock = Mock(
        return_value={
            "mae": 0.1,
            "rmse": 0.2,
            "cv_score": 0.3,
        }
    )
    monkeypatch.setattr("stock_predictor.cli.train_and_evaluate", train_mock)

    recommended = ["AAPL", "MSFT", "GOOGL"]
    monkeypatch.setattr(
        "stock_predictor.cli.get_recommended_tickers", lambda *args, **kwargs: recommended
    )

    result = runner.invoke(main, ["forecast"])

    assert result.exit_code == 0
    fetch_mock.assert_called_once_with(
        "AAPL", period="60d", interval="1d", adjust="none"
    )
    assert "自動選択ティッカー: AAPL" in result.output


def test_cli_forecast_runs_with_ticker(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance", fetch_mock
    )

    train_mock = Mock(
        return_value={
            "mae": 0.1,
            "rmse": 0.2,
            "cv_score": 0.3,
        }
    )
    monkeypatch.setattr("stock_predictor.cli.train_and_evaluate", train_mock)

    result = runner.invoke(
        main,
        [
            "forecast",
            "--ticker",
            "AAPL",
            "--horizon",
            "2",
            "--lags",
            "1",
            "--lags",
            "3",
        ],
    )

    assert result.exit_code == 0
    fetch_mock.assert_called_once_with(
        "AAPL", period="60d", interval="1d", adjust="none"
    )
    train_mock.assert_called_once()
    _, kwargs = train_mock.call_args
    assert kwargs["forecast_horizon"] == 2
    assert kwargs["lags"] == (1, 3)


def test_cli_forecast_rejects_csv_argument(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("dummy")

    result = runner.invoke(
        main,
        [
            "forecast",
            str(csv_path),
        ],
    )

    assert result.exit_code != 0
    assert "CSVファイルの直接指定はサポートされません" in result.output


def test_cli_backtest_uses_recommended_ticker_when_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance", fetch_mock
    )

    simulate_mock = Mock(
        return_value={
            "trades": 3,
            "win_rate": 0.5,
            "cumulative_return": 0.08,
            "initial_capital": 1_000_000.0,
            "ending_balance": 1_080_000.0,
            "total_profit": 80_000.0,
            "max_drawdown": 0.05,
            "signals": [],
        }
    )
    monkeypatch.setattr(
        "stock_predictor.cli.simulate_trading_strategy", simulate_mock
    )

    recommended = ["AAPL", "MSFT", "GOOGL"]
    monkeypatch.setattr(
        "stock_predictor.cli.get_recommended_tickers", lambda *args, **kwargs: recommended
    )

    result = runner.invoke(
        main,
        [
            "backtest",
        ],
    )

    assert result.exit_code == 0
    fetch_mock.assert_called_once_with("AAPL", period="60d", interval="1d")
    simulate_mock.assert_called_once()
    assert "自動選択ティッカー: AAPL" in result.output


def test_cli_backtest_runs_with_ticker(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance", fetch_mock
    )

    simulate_mock = Mock(
        return_value={
            "trades": 3,
            "win_rate": 0.5,
            "cumulative_return": 0.08,
            "initial_capital": 1_000_000.0,
            "ending_balance": 1_080_000.0,
            "total_profit": 80_000.0,
            "max_drawdown": 0.05,
            "signals": [],
        }
    )
    monkeypatch.setattr(
        "stock_predictor.cli.simulate_trading_strategy", simulate_mock
    )

    result = runner.invoke(
        main,
        [
            "backtest",
            "--ticker",
            "AAPL",
            "--threshold",
            "0.001",
            "--lags",
            "1",
        ],
    )

    assert result.exit_code == 0
    fetch_mock.assert_called_once_with("AAPL", period="60d", interval="1d")
    simulate_mock.assert_called_once()


def test_cli_backtest_rejects_csv_argument(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("dummy")

    result = runner.invoke(
        main,
        [
            "backtest",
            str(csv_path),
        ],
    )

    assert result.exit_code != 0
    assert "CSVファイルの直接指定はサポートされません" in result.output
