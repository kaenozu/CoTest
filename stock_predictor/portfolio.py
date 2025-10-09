"""ポートフォリオ最適化ロジック."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from .backtest import simulate_portfolio_trading_strategy, simulate_trading_strategy
from .data import PriceRow


def optimize_ticker_combinations(
    price_series_by_ticker: Mapping[str, Sequence[PriceRow]],
    combination_size: int,
    *,
    backtest_kwargs: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """ティッカー集合から累積リターン最大の組み合わせを探索する."""

    if combination_size < 1:
        raise ValueError("combination_size は1以上で指定してください")

    tickers = [ticker for ticker in price_series_by_ticker.keys()]
    if not tickers:
        raise ValueError("最適化対象のティッカーがありません")
    if combination_size > len(tickers):
        raise ValueError("combination_size がティッカー数を超えています")

    params = dict(backtest_kwargs or {})

    per_ticker_results: MutableMapping[str, Dict[str, Any]] = {}
    for ticker in tickers:
        series = price_series_by_ticker[ticker]
        per_ticker_results[ticker] = simulate_trading_strategy(series, **params)

    ranking: list[Dict[str, Any]] = []
    for combo in combinations(tickers, combination_size):
        combo_price_map = {ticker: price_series_by_ticker[ticker] for ticker in combo}
        portfolio_metrics = simulate_portfolio_trading_strategy(combo_price_map, **params)
        entry = dict(portfolio_metrics)
        entry["tickers"] = combo
        ranking.append(entry)

    ranking.sort(
        key=lambda item: (
            float(item.get("cumulative_return", 0.0)),
            float(item.get("total_profit", 0.0)),
        ),
        reverse=True,
    )

    best_metrics = ranking[0] if ranking else None
    best_combination: Tuple[str, ...] = best_metrics.get("tickers", tuple()) if best_metrics else tuple()

    return {
        "best_combination": best_combination,
        "best_metrics": best_metrics,
        "ranking": ranking,
        "per_ticker_results": dict(per_ticker_results),
    }
