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


def test_cli_forecast_requires_ticker():
    runner = CliRunner()
    result = runner.invoke(main, ["forecast"])

    assert result.exit_code != 0
    assert "--ticker を指定してください" in result.output


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
        "AAPL",
        period="3y",
        interval="1d",
        adjust="none",
        start_date=None,
        end_date=None,
        additional_columns=(),
    )
    train_mock.assert_called_once()
    _, kwargs = train_mock.call_args
    assert kwargs["forecast_horizon"] == 2
    assert kwargs["lags"] == (1, 3)


def test_cli_forecast_accepts_date_range_and_columns(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance",
        fetch_mock,
    )

    train_mock = Mock(return_value={"mae": 0.1, "rmse": 0.2, "cv_score": 0.3})
    monkeypatch.setattr("stock_predictor.cli.train_and_evaluate", train_mock)

    result = runner.invoke(
        main,
        [
            "forecast",
            "--ticker",
            "AAPL",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2023-01-01",
            "--column",
            "Dividends",
            "--column",
            "Stock Splits",
        ],
    )

    assert result.exit_code == 0
    fetch_mock.assert_called_once()
    _, kwargs = fetch_mock.call_args
    assert kwargs["start_date"] == date(2020, 1, 1)
    assert kwargs["end_date"] == date(2023, 1, 1)
    assert kwargs["additional_columns"] == ("Dividends", "Stock Splits")


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
    fetch_mock.assert_called_once_with(
        "AAPL",
        period="3y",
        interval="1d",
        adjust="none",
        start_date=None,
        end_date=None,
        additional_columns=(),
    )
    simulate_mock.assert_called_once()


def test_cli_backtest_accepts_date_range_and_columns(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance",
        fetch_mock,
    )

    simulate_mock = Mock(
        return_value={
            "trades": 1,
            "win_rate": 1.0,
            "cumulative_return": 0.1,
            "initial_capital": 1_000_000.0,
            "ending_balance": 1_100_000.0,
            "total_profit": 100_000.0,
            "max_drawdown": 0.02,
            "signals": [],
        }
    )
    monkeypatch.setattr(
        "stock_predictor.cli.simulate_trading_strategy",
        simulate_mock,
    )

    result = runner.invoke(
        main,
        [
            "backtest",
            "--ticker",
            "AAPL",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2023-01-01",
            "--column",
            "Dividends",
        ],
    )

    assert result.exit_code == 0
    fetch_mock.assert_called_once()
    _, kwargs = fetch_mock.call_args
    assert kwargs["start_date"] == date(2021, 1, 1)
    assert kwargs["end_date"] == date(2023, 1, 1)
    assert kwargs["additional_columns"] == ("Dividends",)


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


def test_cli_forecast_accepts_live_mode(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance", fetch_mock
    )

    class DummyModel:
        def predict_one(self, features):
            return features[0] * 2

    train_mock = Mock(
        return_value={
            "mae": 0.1,
            "rmse": 0.2,
            "cv_score": 0.3,
            "model": DummyModel(),
        }
    )
    monkeypatch.setattr("stock_predictor.cli.train_and_evaluate", train_mock)

    live_rows = [
        {
            "Date": date(2023, 2, 1),
            "Open": 150.0,
            "High": 151.0,
            "Low": 149.0,
            "Close": 150.5,
            "Volume": 1000.0,
        },
        {
            "Date": date(2023, 2, 2),
            "Open": 151.0,
            "High": 152.0,
            "Low": 150.0,
            "Close": 151.5,
            "Volume": 1100.0,
        },
    ]

    def fake_stream_live_prices(client, ticker, limit, **kwargs):
        assert ticker == "AAPL"
        assert limit == 2
        for row in live_rows:
            yield row

    monkeypatch.setattr(
        "stock_predictor.cli.stream_live_prices", fake_stream_live_prices
    )

    def fake_build_latest_feature_row(prices, **kwargs):
        return [float(len(prices))], ["dummy"]

    monkeypatch.setattr(
        "stock_predictor.cli.build_latest_feature_row", fake_build_latest_feature_row
    )

    client = object()

    def fake_resolve_live_client(identifier):
        assert identifier == "dummy.factory"
        return client

    monkeypatch.setattr("stock_predictor.cli._resolve_live_client", fake_resolve_live_client)

    result = runner.invoke(
        main,
        [
            "forecast",
            "--ticker",
            "AAPL",
            "--live",
            "--live-client",
            "dummy.factory",
            "--live-limit",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "===== ライブ予測 =====" in result.output
    assert "最新終値" in result.output


def test_cli_backtest_passes_cv_splits(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    dummy_rows = _dummy_price_rows()

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance", fetch_mock
    )

    simulate_mock = Mock(
        return_value={
            "trades": 5,
            "win_rate": 0.6,
            "cumulative_return": 0.12,
            "initial_capital": 1_000_000.0,
            "ending_balance": 1_120_000.0,
            "total_profit": 120_000.0,
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
            "--cv-splits",
            "4",
        ],
    )

    assert result.exit_code == 0
    simulate_mock.assert_called_once()
    _, kwargs = simulate_mock.call_args
    assert kwargs["cv_splits"] == 4
    assert kwargs["initial_capital"] == 1_000_000.0
    assert kwargs["position_fraction"] == 1.0
    assert kwargs["fee_rate"] == 0.0
    assert kwargs["slippage"] == 0.0
    assert kwargs["max_drawdown_limit"] is None
