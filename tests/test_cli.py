from datetime import date, timedelta
from pathlib import Path

from click.testing import CliRunner

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
