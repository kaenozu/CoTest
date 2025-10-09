"""ポートフォリオ最適化ロジック."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from .backtest import simulate_trading_strategy
from .data import PriceRow


@dataclass(frozen=True)
class CombinationMetrics:
    """組み合わせごとの集計指標."""

    tickers: Tuple[str, ...]
    initial_capital: float
    ending_balance: float
    total_profit: float
    cumulative_return: float


def _aggregate_metrics(
    per_ticker: Mapping[str, Mapping[str, Any]],
    tickers: Tuple[str, ...],
) -> CombinationMetrics:
    total_initial = 0.0
    total_profit = 0.0
    total_ending = 0.0
    for ticker in tickers:
        result = per_ticker[ticker]
        total_initial += float(result.get("initial_capital", 0.0))
        total_profit += float(result.get("total_profit", 0.0))
        total_ending += float(result.get("ending_balance", 0.0))
    cumulative = total_profit / total_initial if total_initial else 0.0
    return CombinationMetrics(
        tickers=tickers,
        initial_capital=total_initial,
        ending_balance=total_ending,
        total_profit=total_profit,
        cumulative_return=cumulative,
    )


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

    ranking: list[CombinationMetrics] = []
    for combo in combinations(tickers, combination_size):
        metrics = _aggregate_metrics(per_ticker_results, combo)
        ranking.append(metrics)

    ranking.sort(
        key=lambda item: (item.cumulative_return, item.total_profit),
        reverse=True,
    )

    best_metrics = ranking[0] if ranking else None
    best_combination: Tuple[str, ...] = best_metrics.tickers if best_metrics else tuple()

    # dict形式で返却
    ranking_dicts = [
        {
            "tickers": metrics.tickers,
            "initial_capital": metrics.initial_capital,
            "ending_balance": metrics.ending_balance,
            "total_profit": metrics.total_profit,
            "cumulative_return": metrics.cumulative_return,
        }
        for metrics in ranking
    ]

    return {
        "best_combination": best_combination,
        "best_metrics": ranking_dicts[0] if ranking_dicts else None,
        "ranking": ranking_dicts,
        "per_ticker_results": dict(per_ticker_results),
    }
