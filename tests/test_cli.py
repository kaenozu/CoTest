from datetime import date, timedelta
from pathlib import Path

from click.testing import CliRunner

from unittest.mock import Mock

import pytest

from stock_predictor.cli import main


def create_csv(path: Path, days: int = 60) -> None:
    rows = ["Date,Open,High,Low,Close,Volume"]
    base_date = date(2023, 1, 1)
    price = 100.0
    for i in range(days):
        price += 0.5
        day = base_date + timedelta(days=i)
        rows.append(
            f"{day.isoformat()},{price-0.3:.2f},{price+0.4:.2f},{price-0.8:.2f},{price:.2f},{1000+i*5}"
        )
    path.write_text("\n".join(rows))


def test_cli_runs_and_outputs_metrics(tmp_path):
    csv_path = tmp_path / "prices.csv"
    create_csv(csv_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "forecast",
            str(csv_path),
            "--horizon",
            "1",
            "--lags",
            "1",
            "--lags",
            "2",
            "--lags",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "平均絶対誤差" in result.output
    assert "二乗平均平方根誤差" in result.output


def test_cli_accepts_ridge_option(tmp_path):
    csv_path = tmp_path / "prices.csv"
    create_csv(csv_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "forecast",
            str(csv_path),
            "--ridge",
            "0.1",
        ],
    )

    assert result.exit_code == 0


def test_cli_backtest_outputs_strategy_metrics(tmp_path):
    csv_path = tmp_path / "prices.csv"
    create_csv(csv_path, days=80)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "backtest",
            str(csv_path),
            "--threshold",
            "0.001",
            "--lags",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "トレード回数" in result.output
    assert "初期資金" in result.output
    assert "最大ドローダウン" in result.output


def test_cli_backtest_accepts_ticker(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    dummy_rows = [
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

    fetch_mock = Mock(return_value=dummy_rows)
    monkeypatch.setattr(
        "stock_predictor.cli.fetch_price_data_from_yfinance", fetch_mock
    )

    simulate_mock = Mock(
        return_value={
            "trades": 3,
            "win_rate": 0.5,
            "cumulative_return": 0.08,
            "signals": [],
        }
    )
    monkeypatch.setattr(
        "stock_predictor.cli.simulate_trading_strategy", simulate_mock
    )

    load_mock = Mock(side_effect=AssertionError("load_price_data should not be called"))
    monkeypatch.setattr("stock_predictor.cli.load_price_data", load_mock)

    result = runner.invoke(
        main,
        [
            "backtest",
            "--ticker",
            "AAPL",
            "--period",
            "30d",
            "--interval",
            "1h",
            "--threshold",
            "0.001",
        ],
    )

    assert result.exit_code == 0
    fetch_mock.assert_called_once_with("AAPL", period="30d", interval="1h")
    simulate_mock.assert_called_once()
    args, _ = simulate_mock.call_args
    assert args[0] == dummy_rows


def test_cli_fetches_data_from_yfinance(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    dummy_rows = [
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
            "--period",
            "60d",
            "--interval",
            "1d",
        ],
    )

    assert result.exit_code == 0
    fetch_mock.assert_called_once_with(
        "AAPL", period="60d", interval="1d", adjust="none"
    )
    train_mock.assert_called_once()
    _, kwargs = train_mock.call_args
    assert kwargs["cv_splits"] == 5


def test_cli_backtest_rejects_csv_and_ticker(tmp_path):
    csv_path = tmp_path / "prices.csv"
    create_csv(csv_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "backtest",
            str(csv_path),
            "--ticker",
            "AAPL",
        ],
    )

    assert result.exit_code != 0
    assert "CSVパスまたは--tickerのいずれか一方を指定してください" in result.output


def test_cli_accepts_cv_splits_option_for_forecast(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    csv_path = tmp_path / "prices.csv"
    create_csv(csv_path)

    runner = CliRunner()

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
            str(csv_path),
            "--cv-splits",
            "7",
        ],
    )

    assert result.exit_code == 0
    train_mock.assert_called_once()
    _, kwargs = train_mock.call_args
    assert kwargs["cv_splits"] == 7


def test_cli_backtest_accepts_cv_splits_option(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    csv_path = tmp_path / "prices.csv"
    create_csv(csv_path, days=80)

    runner = CliRunner()

    simulate_mock = Mock(
        return_value={
            "trades": 5,
            "win_rate": 0.6,
            "cumulative_return": 0.12,
            "initial_capital": 1_000_000.0,
            "ending_balance": 1_120_000.0,
            "total_profit": 120_000.0,
            "balance_history": [1_000_000.0, 1_120_000.0],
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
            str(csv_path),
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


def test_cli_forecast_streams_live_data(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    csv_path = tmp_path / "prices.csv"
    create_csv(csv_path, days=10)

    runner = CliRunner()

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
            str(csv_path),
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