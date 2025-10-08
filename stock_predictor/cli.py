"""株価予測CLI."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Tuple

import click

from .backtest import simulate_trading_strategy
from .data import (
    build_latest_feature_row,
    fetch_price_data_from_yfinance,
    load_price_data,
    stream_live_prices,
)
from .model import train_and_evaluate


@click.group()
def main() -> None:
    """株価予測ツール."""


def _resolve_live_client(identifier: str):
    """`module:attribute` または `module.attribute` 形式からクライアントを解決する."""

    if not identifier:
        raise click.BadParameter("クライアントを識別する文字列を指定してください")
    module_name: str
    attr_name: str
    if ":" in identifier:
        module_name, attr_name = identifier.split(":", 1)
    elif "." in identifier:
        module_name, attr_name = identifier.rsplit(".", 1)
    else:
        raise click.BadParameter(
            "module:factory または module.factory 形式で指定してください",
            param_hint="--live-client",
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - importエラー経路の保持
        raise click.BadParameter(
            f"モジュール {module_name} をインポートできません",
            param_hint="--live-client",
        ) from exc
    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise click.BadParameter(
            f"{module_name} に {attr_name} が見つかりません",
            param_hint="--live-client",
        ) from exc

    client = factory() if callable(factory) else factory
    if not hasattr(client, "stream_prices"):
        raise click.BadParameter(
            "ライブクライアントは stream_prices メソッドを実装している必要があります",
            param_hint="--live-client",
        )
    return client


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
@click.option(
    "--live",
    is_flag=True,
    help="ライブ価格を購読し最新データに基づく予測を表示する",
)
@click.option(
    "--live-client",
    type=str,
    help="ライブ価格取得クライアントのimportパス(module:factory形式)",
)
@click.option(
    "--live-limit",
    type=int,
    default=5,
    show_default=True,
    help="ライブデータを受信する件数(0で無制限)",
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
    live: bool,
    live_client: str | None,
    live_limit: int,
) -> None:
    """CSVまたはyfinanceから学習し翌日以降の終値を予測する."""

    if live_limit < 0:
        raise click.BadParameter("--live-limit には0以上を指定してください")

    if csv_path is None and ticker is None:
        raise click.UsageError("CSVパスまたは--tickerのいずれかを指定してください")
    if csv_path is not None and ticker is not None and not live:
        raise click.UsageError("ライブ利用時以外はCSVと--tickerを同時指定できません")

    if csv_path is not None:
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

    if not live and live_client is None:
        return

    if ticker is None:
        raise click.UsageError("ライブ予測には--tickerを指定してください")
    if live_client is None:
        raise click.UsageError("ライブ予測には--live-clientを指定してください")

    model = result.get("model")
    if model is None or not hasattr(model, "predict_one"):
        raise click.ClickException("推論に使用するモデルを取得できませんでした")

    client = _resolve_live_client(live_client)
    history = list(data)
    stream_limit = None if live_limit == 0 else live_limit

    click.echo("===== ライブ予測 =====")
    for latest in stream_live_prices(client, ticker, limit=stream_limit):
        history.append(latest)
        try:
            feature_row, _ = build_latest_feature_row(
                history,
                forecast_horizon=horizon,
                lags=effective_lags,
            )
        except ValueError:
            click.echo(f"{latest['Date']}: 十分な履歴がないため予測をスキップします")
            continue
        prediction = model.predict_one(feature_row)
        close_price = float(latest["Close"])
        click.echo(
            f"{latest['Date']}: 最新終値 {close_price:.2f} / 予測終値 {prediction:.2f}"
        )


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