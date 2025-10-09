"""株価予測CLI."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence, Tuple

import click

from .backtest import CostModel, simulate_trading_strategy
from .data import (
    build_latest_feature_row,
    fetch_price_data_from_yfinance,
    PriceRow,
    stream_live_prices,
)
from .portfolio import optimize_ticker_combinations
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


def _parse_fee_tiers(raw_values: Sequence[str]) -> list[tuple[float, float]]:
    tiers: list[tuple[float, float]] = []
    for raw in raw_values:
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            threshold_str, rate_str = raw.split(":", 1)
        elif "," in raw:
            threshold_str, rate_str = raw.split(",", 1)
        else:
            raise click.BadParameter(
                "--fee-tier は '閾値:料率' 形式で指定してください",
                param_hint="--fee-tier",
            )
        try:
            threshold = float(threshold_str)
            rate = float(rate_str)
        except ValueError as exc:  # pragma: no cover - 例外経路の保持
            raise click.BadParameter(
                "--fee-tier の値を数値に変換できません",
                param_hint="--fee-tier",
            ) from exc
        tiers.append((threshold, rate))
    return tiers


@main.command(context_settings={"allow_extra_args": True})
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
    "--adjust",
    type=click.Choice(["none", "auto", "manual"]),
    default="none",
    show_default=True,
    help="yfinance取得時の価格調整モード",
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
@click.pass_context
def forecast(
    ctx: click.Context,
    horizon: int,
    lags: Tuple[int, ...],
    ridge: float,
    cv_splits: int,
    ticker: str | None,
    period: str,
    interval: str,
    adjust: str,
    live: bool,
    live_client: str | None,
    live_limit: int,
) -> None:
    """CSVまたはyfinanceから学習し翌日以降の終値を予測する."""

    if live_limit < 0:
        raise click.BadParameter("--live-limit には0以上を指定してください")

    if ctx.args:
        raise click.UsageError("CSVファイルの直接指定はサポートされません。--ticker を指定してください")

    if ticker is None:
        raise click.UsageError("--ticker を指定してください")

    data = fetch_price_data_from_yfinance(
        ticker,
        period=period,
        interval=interval,
        adjust=adjust,
    )
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


@main.command(context_settings={"allow_extra_args": True})
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
@click.option(
    "--initial-capital",
    default=1_000_000.0,
    show_default=True,
    type=float,
    help="バックテスト開始時の資金",
)
@click.option(
    "--position-fraction",
    default=1.0,
    show_default=True,
    type=float,
    help="各トレードで投入する資金割合(0-1)",
)
@click.option(
    "--fee-rate",
    default=0.0,
    show_default=True,
    type=float,
    help="売買時にかかる手数料率",
)
@click.option(
    "--slippage",
    default=0.0,
    show_default=True,
    type=float,
    help="約定価格に上乗せするスリッページ(比率)",
)
@click.option(
    "--slippage-long",
    type=float,
    help="ロング方向のスリッページ率(指定時は--slippageを上書き)",
)
@click.option(
    "--slippage-short",
    type=float,
    help="ショート方向のスリッページ率(指定時は--slippageを上書き)",
)
@click.option(
    "--liquidity-slippage",
    default=0.0,
    show_default=True,
    type=float,
    help="出来高比率に応じて加算するスリッページ係数",
)
@click.option(
    "--fixed-fee",
    default=0.0,
    show_default=True,
    type=float,
    help="1回の往復取引ごとに発生する固定手数料",
)
@click.option(
    "--fee-tier",
    type=str,
    multiple=True,
    help="閾値:料率 形式で段階的手数料を指定 (例: --fee-tier 100000:0.001)",
)
@click.option(
    "--tick-size",
    type=float,
    help="発注時に適用する価格刻み",
)
@click.option(
    "--max-drawdown-limit",
    type=float,
    help="許容最大ドローダウン(比率)。超過で取引停止",
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
@click.pass_context
def backtest(
    ctx: click.Context,
    horizon: int,
    lags: Tuple[int, ...],
    ridge: float,
    threshold: float,
    cv_splits: int,
    initial_capital: float,
    position_fraction: float,
    fee_rate: float,
    slippage: float,
    slippage_long: float | None,
    slippage_short: float | None,
    liquidity_slippage: float,
    fixed_fee: float,
    fee_tier: Tuple[str, ...],
    tick_size: float | None,
    max_drawdown_limit: float | None,
    ticker: str | None,
    period: str,
    interval: str,
) -> None:
    """予測値を用いたシンプルトレード戦略をバックテストする."""

    if ctx.args:
        raise click.UsageError("CSVファイルの直接指定はサポートされません。--ticker を指定してください")

    if ticker is None:
        raise click.UsageError("--ticker を指定してください")

    data = fetch_price_data_from_yfinance(ticker, period=period, interval=interval)
    source_label = f"{ticker} ({period}, {interval})"

    effective_lags = lags or (1, 2, 3, 5, 10)

    directional_slippage: float | dict[str, float]
    if slippage_long is not None or slippage_short is not None:
        directional_slippage = {
            "long": slippage_long if slippage_long is not None else slippage,
            "short": slippage_short if slippage_short is not None else slippage,
        }
    else:
        directional_slippage = slippage

    parsed_fee_tiers = _parse_fee_tiers(fee_tier) if fee_tier else None
    fee_tiers = parsed_fee_tiers or None

    try:
        model = CostModel(
            fee_rate=fee_rate,
            fee_tiers=fee_tiers,
            fixed_fee=fixed_fee,
            slippage=directional_slippage,
            liquidity_slippage=liquidity_slippage,
            tick_size=tick_size,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc))

    result = simulate_trading_strategy(
        data,
        forecast_horizon=horizon,
        lags=effective_lags,
        cv_splits=cv_splits,
        ridge_lambda=ridge,
        threshold=threshold,
        initial_capital=initial_capital,
        position_fraction=position_fraction,
        fee_rate=fee_rate,
        slippage=slippage,
        cost_model=model,
        max_drawdown_limit=max_drawdown_limit,
    )

    click.echo("===== バックテスト結果 =====")
    click.echo(f"使用データ: {source_label}")
    click.echo(f"予測ホライゾン: {horizon} 日")
    click.echo(f"使用ラグ: {', '.join(str(l) for l in effective_lags)}")
    click.echo(f"トレード回数: {result['trades']}")
    click.echo(f"勝率: {result['win_rate'] * 100:.2f}%")
    click.echo(f"初期資金: {result['initial_capital']:.2f}")
    click.echo(f"最終残高: {result['ending_balance']:.2f}")
    click.echo(f"累積損益: {result['total_profit']:.2f}")
    click.echo(f"累積リターン: {result['cumulative_return'] * 100:.2f}%")
    click.echo(f"最大ドローダウン: {result['max_drawdown'] * 100:.2f}%")

    preview = result["signals"][:10]
    if preview:
        click.echo("--- トレード一覧(最大10件) ---")
        for trade in preview:
            entry = trade["entry"]
            exit_ = trade["exit"]
            entry_time = entry["timestamp"]
            exit_time = exit_["timestamp"]
            predicted = entry.get("predicted_return", 0.0) * 100
            realized = trade.get("return", 0.0) * 100
            profit = trade.get("profit", 0.0)
            click.echo(
                f"{entry_time.date()} -> {exit_time.date()}: {trade['direction']}"
                f" | 予測 {predicted:.2f}% / 実現 {realized:.2f}%"
                f" | 損益 {profit:.2f}"
            )


@main.command("backtest-portfolio")
@click.argument(
    "tickers_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--combination-size", default=2, show_default=True, type=int, help="同時に評価するティッカー数")
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
@click.option(
    "--initial-capital",
    default=1_000_000.0,
    show_default=True,
    type=float,
    help="バックテスト開始時の資金",
)
@click.option(
    "--position-fraction",
    default=1.0,
    show_default=True,
    type=float,
    help="各トレードで投入する資金割合(0-1)",
)
@click.option(
    "--fee-rate",
    default=0.0,
    show_default=True,
    type=float,
    help="売買時にかかる手数料率",
)
@click.option(
    "--slippage",
    default=0.0,
    show_default=True,
    type=float,
    help="約定価格に上乗せするスリッページ(比率)",
)
@click.option(
    "--max-drawdown-limit",
    type=float,
    help="許容最大ドローダウン(比率)。超過で取引停止",
)
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
def backtest_portfolio(
    tickers_file: Path,
    combination_size: int,
    horizon: int,
    lags: Tuple[int, ...],
    ridge: float,
    threshold: float,
    cv_splits: int,
    initial_capital: float,
    position_fraction: float,
    fee_rate: float,
    slippage: float,
    max_drawdown_limit: float | None,
    period: str,
    interval: str,
) -> None:
    """ティッカーリストから最適なポートフォリオ組み合わせを探索する."""

    if combination_size < 1:
        raise click.BadParameter("--combination-size は1以上で指定してください")

    raw_text = tickers_file.read_text(encoding="utf-8")
    tickers = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not tickers:
        raise click.ClickException("ティッカーリストファイルに有効なティッカーがありません")
    if combination_size > len(tickers):
        raise click.BadParameter(
            "--combination-size がティッカー数を超えています", param_hint="--combination-size"
        )

    price_map: dict[str, Sequence[PriceRow]] = {}
    for ticker in tickers:
        price_map[ticker] = fetch_price_data_from_yfinance(
            ticker,
            period=period,
            interval=interval,
        )

    effective_lags = lags or (1, 2, 3, 5, 10)
    backtest_kwargs = {
        "forecast_horizon": horizon,
        "lags": tuple(effective_lags),
        "cv_splits": cv_splits,
        "ridge_lambda": ridge,
        "threshold": threshold,
        "initial_capital": initial_capital,
        "position_fraction": position_fraction,
        "fee_rate": fee_rate,
        "slippage": slippage,
        "max_drawdown_limit": max_drawdown_limit,
    }

    result = optimize_ticker_combinations(
        price_map,
        combination_size,
        backtest_kwargs=backtest_kwargs,
    )

    click.echo("===== ポートフォリオ最適化 =====")
    click.echo(f"候補ティッカー: {', '.join(tickers)}")
    click.echo(f"組み合わせサイズ: {combination_size}")

    best_combination = result.get("best_combination", ())
    best_metrics = result.get("best_metrics") or {}
    if best_combination:
        label = ", ".join(best_combination)
        click.echo(f"最適組み合わせ: {label}")
        click.echo(
            "ポートフォリオ累積リターン: "
            f"{float(best_metrics.get('cumulative_return', 0.0)) * 100:.2f}%"
        )
        click.echo(
            "ポートフォリオ累積損益: "
            f"{float(best_metrics.get('total_profit', 0.0)):.2f}"
        )
    else:
        click.echo("最適組み合わせを特定できませんでした")

    click.echo("--- 個別ティッカー成績 ---")
    per_ticker = result.get("per_ticker_results", {})
    for ticker in tickers:
        ticker_result = per_ticker.get(ticker)
        if not ticker_result:
            continue
        cumulative = float(ticker_result.get("cumulative_return", 0.0)) * 100
        profit = float(ticker_result.get("total_profit", 0.0))
        click.echo(
            f"{ticker}: 累積リターン {cumulative:.2f}% / 累積損益 {profit:.2f}"
        )

    ranking = result.get("ranking", [])
    if ranking:
        click.echo("--- 上位組み合わせ ---")
        for idx, entry in enumerate(ranking[:10], start=1):
            combo_label = ", ".join(entry["tickers"])
            cumulative = float(entry.get("cumulative_return", 0.0)) * 100
            profit = float(entry.get("total_profit", 0.0))
            click.echo(
                f"{idx}. {combo_label} (累積リターン {cumulative:.2f}%, 累積損益 {profit:.2f})"
            )
