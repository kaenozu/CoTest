"""推奨ティッカー集合を提供するユニバースモジュール."""

from __future__ import annotations

from typing import Dict, List

DEFAULT_UNIVERSE = "default"

_UNIVERSES: Dict[str, List[str]] = {
    DEFAULT_UNIVERSE: [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK-B",
        "UNH",
        "JPM",
        "V",
        "JNJ",
        "HD",
        "PG",
        "MA",
    ],
    "dividend_focus": [
        "KO",
        "PEP",
        "PG",
        "JNJ",
        "HD",
        "WMT",
        "MCD",
        "CVX",
        "XOM",
        "T",
    ],
    "innovation_growth": [
        "AAPL",
        "MSFT",
        "NVDA",
        "ADBE",
        "CRM",
        "NFLX",
        "AMD",
        "ASML",
        "INTU",
        "SHOP",
    ],
}


def get_recommended_tickers(
    universe: str = DEFAULT_UNIVERSE, *, limit: int | None = None
) -> list[str]:
    """ユニバース名に対応する推奨ティッカーを取得する."""

    if limit is not None and limit < 1:
        raise ValueError("limit は1以上で指定してください")

    tickers = _UNIVERSES.get(universe)
    if tickers is None:
        raise ValueError(f"未知のユニバースです: {universe}")

    result = list(tickers)
    if limit is not None:
        result = result[:limit]
    return result


__all__ = ["DEFAULT_UNIVERSE", "get_recommended_tickers"]
