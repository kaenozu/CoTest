import pytest

from stock_predictor import universe


def test_get_recommended_tickers_returns_default_list():
    tickers = universe.get_recommended_tickers()

    assert tickers
    assert all(isinstance(t, str) for t in tickers)
    assert len(tickers) >= 10
    # デフォルトユニバースの先頭はS&P大型株の代表例とする
    assert tickers[:5] == ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_get_recommended_tickers_honors_limit(limit: int):
    tickers = universe.get_recommended_tickers(limit=limit)

    assert len(tickers) == limit


def test_get_recommended_tickers_unknown_universe():
    with pytest.raises(ValueError):
        universe.get_recommended_tickers("unknown")
