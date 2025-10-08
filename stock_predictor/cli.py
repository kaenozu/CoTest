"""株価予測CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click

from .backtest import simulate_trading_strategy
from .data import fetch_price_data_from_yfinance, load_price_data
from .model import train_and_evaluate


@click.group()
def main() -> None:
    """株価予測ツール."""


@main.command()
@click.argument(
    "csv_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
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
@click.option(
    "--cv-splits",
    default=5,
    show_default=True,
    type=int,
    help="交差検証の分割数",
)
@click.option("--ticker", type=str, help="yfinanceから取得するティッカー")
@click.option(
    "--period",
    type=str,
    default="60d",
    show_default=True,
    help="yfinanceから取得する期間",
)
@click.option(
    "--interval",
    type=str,
    default="1d",
    show_default=True,
    help="yfinanceから取得する足種",
)
def forecast(
    csv_path: Path | None,
    horizon: int,
    lags: Tuple[int, ...],
    ridge: float,
    cv_splits: int,
    ticker: str | None,
    period: str,
    interval: str,
) -> None:
    """CSVまたはyfinanceから学習し翌日以降の終値を予測する."""

    if (csv_path is None) == (ticker is None):
        raise click.UsageError("CSVパスまたは--tickerのいずれか一方を指定してください")

    if csv_path is not None:
        if ticker is not None:
            raise click.UsageError("CSV入力時は--tickerを同時指定できません")
        if period != "60d" or interval != "1d":
            raise click.UsageError("CSV入力時は--period/--intervalを指定できません")
        data = load_price_data(csv_path)
        source_label = str(csv_path)
    else:
        data = fetch_price_data_from_yfinance(ticker, period=period, interval=interval)
        source_label = f"{ticker} ({period}, {interval})"

    effective_lags = lags or (1, 2, 3, 5, 10)

    result = train_and_evaluate(
        data,
        forecast_horizon=horizon,
        lags=effective_lags,
        cv_splits=cv_splits,
        ridge_lambda=ridge,
    )

    click.echo("===== 予測評価結果 =====")
    click.echo(f"使用データ: {source_label}")
    click.echo(f"予測ホライゾン: {horizon} 日")
    click.echo(f"使用ラグ: {', '.join(str(l) for l in effective_lags)}")
    click.echo(f"平均絶対誤差(MAE): {result['mae']:.4f}")
    click.echo(f"二乗平均平方根誤差(RMSE): {result['rmse']:.4f}")
    click.echo(f"CV平均RMSE: {result['cv_score']:.4f}")


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
@click.option(
    "--threshold",
    default=0.0,
    show_default=True,
    type=float,
    help="売買シグナルとする予測リターンの閾値(比率)",
)
@click.option(
    "--cv-splits",
    default=5,
    show_default=True,
    type=int,
    help="交差検証の分割数",
)
def backtest(
    csv_path: Path,
    horizon: int,
    lags: Tuple[int, ...],
    ridge: float,
    threshold: float,
    cv_splits: int,
) -> None:
    """予測値を用いたシンプルトレード戦略をバックテストする."""

    data = load_price_data(csv_path)
    effective_lags = lags or (1, 2, 3, 5, 10)

    result = simulate_trading_strategy(
        data,
        forecast_horizon=horizon,
        lags=effective_lags,
        cv_splits=cv_splits,
        ridge_lambda=ridge,
        threshold=threshold,
    )

    click.echo("===== バックテスト結果 =====")
    click.echo(f"使用データ: {csv_path}")
    click.echo(f"予測ホライゾン: {horizon} 日")
    click.echo(f"使用ラグ: {', '.join(str(l) for l in effective_lags)}")
    click.echo(f"トレード回数: {result['trades']}")
    click.echo(f"勝率: {result['win_rate'] * 100:.2f}%")
    click.echo(f"累積リターン: {result['cumulative_return'] * 100:.2f}%")

    preview = result["signals"][:10]
    if preview:
        click.echo("--- シグナル一覧(最大10件) ---")
        for signal in preview:
            date = signal["date"]
            click.echo(
                f"{date}: {signal['action']} | 予測リターン {signal['predicted_return'] * 100:.2f}%"
                f" / 実現リターン {signal['actual_return'] * 100:.2f}%"
            )