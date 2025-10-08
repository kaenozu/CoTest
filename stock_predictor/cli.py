"""株価予測CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click

from .data import load_price_data
from .model import train_and_evaluate


@click.group()
def main() -> None:
    """株価予測ツール."""


@main.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--horizon", default=1, show_default=True, type=int, help="予測する営業日数")
@click.option(
    "--lags",
    type=int,
    multiple=True,
    help="特徴量に含める終値のラグ(複数指定可)",
)
@click.option(
    "--ridge",
    default=1e-6,
    show_default=True,
    type=float,
    help="リッジ回帰の正則化係数",
)
def forecast(csv_path: Path, horizon: int, lags: Tuple[int, ...], cv_splits: int, ridge: float) -> None:
    """CSVから学習し翌日以降の終値を予測する."""
    data = load_price_data(csv_path)

    effective_lags = lags or (1, 2, 3, 5, 10)

    result = train_and_evaluate(
        data,
        forecast_horizon=horizon,
        lags=effective_lags,
        cv_splits=cv_splits,
        ridge_lambda=ridge,
    )

    click.echo("===== 予測評価結果 =====")
    click.echo(f"使用データ: {csv_path}")
    click.echo(f"予測ホライゾン: {horizon} 日")
    click.echo(f"使用ラグ: {', '.join(str(l) for l in effective_lags)}")
    click.echo(f"平均絶対誤差(MAE): {result['mae']:.4f}")
    click.echo(f"二乗平均平方根誤差(RMSE): {result['rmse']:.4f}")
    click.echo(f"CV平均RMSE: {result['cv_score']:.4f}")