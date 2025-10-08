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
    assert "累積リターン" in result.output


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
    fetch_mock.assert_called_once_with("AAPL", period="60d", interval="1d")
    train_mock.assert_called_once()