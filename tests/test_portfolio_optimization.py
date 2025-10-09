from datetime import date

from datetime import date

from click.testing import CliRunner

import pytest

from stock_predictor.cli import main
from stock_predictor.data import PriceRow


def _build_price_rows(*closes: float) -> list[PriceRow]:
    rows: list[PriceRow] = []
    for idx, close in enumerate(closes, start=1):
        rows.append(
            {
                "Date": date(2023, 1, idx),
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": 1_000.0,
            }
        )
    return rows


def _make_price_rows(value: float) -> list[PriceRow]:
    return [
        {
            "Date": date(2023, 1, 1),
            "Open": value,
            "High": value,
            "Low": value,
            "Close": value,
            "Volume": 1000.0,
        },
        {
            "Date": date(2023, 1, 2),
            "Open": value + 1,
            "High": value + 1,
            "Low": value + 1,
            "Close": value + 1,
            "Volume": 1000.0,
        },
    ]


def test_optimize_ticker_combinations_selects_highest_cumulative_return(monkeypatch: pytest.MonkeyPatch):
    price_map = {
        "AAA": _make_price_rows(100.0),
        "BBB": _make_price_rows(120.0),
        "CCC": _make_price_rows(140.0),
    }

    simulate_results = {
        "AAA": {
            "initial_capital": 100_000.0,
            "ending_balance": 107_000.0,
            "total_profit": 7_000.0,
            "cumulative_return": 0.07,
            "signals": [],
        },
        "BBB": {
            "initial_capital": 100_000.0,
            "ending_balance": 102_000.0,
            "total_profit": 2_000.0,
            "cumulative_return": 0.02,
            "signals": [],
        },
        "CCC": {
            "initial_capital": 100_000.0,
            "ending_balance": 111_000.0,
            "total_profit": 11_000.0,
            "cumulative_return": 0.11,
            "signals": [],
        },
    }

    call_order: list[str] = []
    id_to_ticker = {id(rows): ticker for ticker, rows in price_map.items()}

    from stock_predictor import portfolio

    def fake_simulate(prices, **kwargs):
        ticker = id_to_ticker[id(prices)]
        call_order.append(ticker)
        return simulate_results[ticker]

    monkeypatch.setattr(portfolio, "simulate_trading_strategy", fake_simulate)

    combo_metrics = {
        ("AAA", "BBB"): {
            "signals": [],
            "trades": 2,
            "win_rate": 0.5,
            "cumulative_return": 0.05,
            "initial_capital": 200_000.0,
            "ending_balance": 210_000.0,
            "total_profit": 10_000.0,
            "balance_history": [200_000.0, 210_000.0],
            "max_drawdown": 0.0,
            "halted": False,
            "halt_reason": None,
            "per_ticker_breakdown": {},
        },
        ("AAA", "CCC"): {
            "signals": [],
            "trades": 2,
            "win_rate": 1.0,
            "cumulative_return": 0.09,
            "initial_capital": 200_000.0,
            "ending_balance": 218_000.0,
            "total_profit": 18_000.0,
            "balance_history": [200_000.0, 218_000.0],
            "max_drawdown": 0.0,
            "halted": False,
            "halt_reason": None,
            "per_ticker_breakdown": {},
        },
        ("BBB", "CCC"): {
            "signals": [],
            "trades": 2,
            "win_rate": 0.5,
            "cumulative_return": 0.065,
            "initial_capital": 200_000.0,
            "ending_balance": 213_000.0,
            "total_profit": 13_000.0,
            "balance_history": [200_000.0, 213_000.0],
            "max_drawdown": 0.0,
            "halted": False,
            "halt_reason": None,
            "per_ticker_breakdown": {},
        },
    }

    def fake_portfolio_simulation(price_map_subset, **kwargs):
        tickers = tuple(sorted(price_map_subset.keys()))
        return combo_metrics[tickers]

    monkeypatch.setattr(
        portfolio,
        "simulate_portfolio_trading_strategy",
        fake_portfolio_simulation,
    )

    result = portfolio.optimize_ticker_combinations(price_map, 2)

    assert call_order.count("AAA") == 1
    assert call_order.count("BBB") == 1
    assert call_order.count("CCC") == 1

    assert result["best_combination"] == ("AAA", "CCC")
    assert result["best_metrics"]["cumulative_return"] == pytest.approx(0.09)
    assert result["best_metrics"]["total_profit"] == pytest.approx(18_000.0)
    assert result["best_metrics"]["initial_capital"] == pytest.approx(200_000.0)
    assert result["best_metrics"]["ending_balance"] == pytest.approx(218_000.0)

    ranking = result["ranking"]
    assert [entry["tickers"] for entry in ranking] == [
        ("AAA", "CCC"),
        ("BBB", "CCC"),
        ("AAA", "BBB"),
    ]
    assert ranking[1]["cumulative_return"] == pytest.approx(0.065)
    assert ranking[2]["total_profit"] == pytest.approx(10_000.0)

    per_ticker = result["per_ticker_results"]
    assert per_ticker["AAA"]["cumulative_return"] == 0.07
    assert per_ticker["BBB"]["total_profit"] == 2_000.0
    assert per_ticker["CCC"] is simulate_results["CCC"]


def test_optimize_ticker_combinations_respects_portfolio_capital(monkeypatch: pytest.MonkeyPatch):
    price_map = {
        "AAA": _build_price_rows(100.0, 110.0, 120.0),
        "BBB": _build_price_rows(200.0, 220.0, 240.0),
    }

    def fake_generate_predictions(dataset, cv_splits, ridge_lambda):
        predictions = [None] * len(dataset.features)
        if predictions:
            predictions[0] = dataset.closes[0] * 1.1
        return predictions

    monkeypatch.setattr(
        "stock_predictor.backtest._generate_predictions",
        fake_generate_predictions,
    )

    from stock_predictor import portfolio

    result = portfolio.optimize_ticker_combinations(
        price_map,
        2,
        backtest_kwargs={
            "forecast_horizon": 1,
            "lags": (1,),
            "rolling_windows": (2,),
            "cv_splits": 2,
            "ridge_lambda": 0.0,
            "threshold": 0.0,
            "initial_capital": 100_000.0,
            "position_fraction": 1.0,
            "fee_rate": 0.0,
            "slippage": 0.0,
        },
    )

    best_metrics = result["best_metrics"]
    assert best_metrics["initial_capital"] == pytest.approx(100_000.0)
    assert best_metrics["ending_balance"] == pytest.approx(109_090.90909090909)
    assert best_metrics["total_profit"] == pytest.approx(9_090.909090909088)
    assert best_metrics["cumulative_return"] == pytest.approx(0.0909090909)
    assert best_metrics["trades"] == 1
    assert [signal["ticker"] for signal in best_metrics["signals"]] == ["AAA"]


def test_optimize_ticker_combinations_accounts_for_mixed_returns(
    monkeypatch: pytest.MonkeyPatch,
):
    price_map = {
        "AAA": _build_price_rows(100.0, 110.0, 120.0),
        "BBB": _build_price_rows(90.0, 81.0, 72.9),
    }

    def fake_generate_predictions(dataset, cv_splits, ridge_lambda):
        predictions = [None] * len(dataset.features)
        if predictions:
            predictions[0] = dataset.closes[0] * 1.1
        return predictions

    monkeypatch.setattr(
        "stock_predictor.backtest._generate_predictions",
        fake_generate_predictions,
    )

    from stock_predictor import portfolio

    result = portfolio.optimize_ticker_combinations(
        price_map,
        2,
        backtest_kwargs={
            "forecast_horizon": 1,
            "lags": (1,),
            "rolling_windows": (2,),
            "cv_splits": 2,
            "ridge_lambda": 0.0,
            "threshold": 0.0,
            "initial_capital": 100_000.0,
            "position_fraction": 0.5,
            "fee_rate": 0.0,
            "slippage": 0.0,
        },
    )

    best_metrics = result["best_metrics"]
    assert best_metrics["ending_balance"] == pytest.approx(102_045.45454545454)
    assert best_metrics["total_profit"] == pytest.approx(2_045.454545454544)
    assert best_metrics["cumulative_return"] == pytest.approx(0.0204545454)
    assert best_metrics["trades"] == 2
    tickers = {signal["ticker"] for signal in best_metrics["signals"]}
    assert tickers == {"AAA", "BBB"}
    breakdown = best_metrics["per_ticker_breakdown"]
    assert breakdown["AAA"]["total_profit"] == pytest.approx(4_545.454545454546)
    assert breakdown["BBB"]["total_profit"] == pytest.approx(-2_500.0)


def test_cli_backtest_portfolio_reads_file_and_outputs_optimal_combo(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAA\nBBB\nCCC\n")

    data_map = {
        "AAA": _make_price_rows(100.0),
        "BBB": _make_price_rows(120.0),
        "CCC": _make_price_rows(140.0),
    }

    def fake_fetch(ticker, *, period, interval):
        return data_map[ticker]

    monkeypatch.setattr("stock_predictor.cli.fetch_price_data_from_yfinance", fake_fetch)

    captured: dict[str, object] = {}

    def fake_optimize(price_map, combination_size, backtest_kwargs=None):
        captured["price_map"] = price_map
        captured["combination_size"] = combination_size
        captured["backtest_kwargs"] = backtest_kwargs or {}
        return {
            "best_combination": ("AAA", "CCC"),
            "best_metrics": {
                "cumulative_return": 0.09,
                "total_profit": 18_000.0,
                "initial_capital": 200_000.0,
                "ending_balance": 218_000.0,
                "trades": 4,
                "win_rate": 0.75,
                "per_ticker_breakdown": {
                    "AAA": {
                        "capital_committed": 100_000.0,
                        "total_profit": 7_000.0,
                        "trades": 2,
                        "win_rate": 1.0,
                    },
                    "BBB": {
                        "capital_committed": 80_000.0,
                        "total_profit": 2_000.0,
                        "trades": 1,
                        "win_rate": 1.0,
                    },
                    "CCC": {
                        "capital_committed": 120_000.0,
                        "total_profit": 9_000.0,
                        "trades": 1,
                        "win_rate": 1.0,
                    },
                },
                "signals": [],
            },
            "ranking": [
                {
                    "tickers": ("AAA", "CCC"),
                    "cumulative_return": 0.09,
                    "total_profit": 18_000.0,
                },
                {
                    "tickers": ("BBB", "CCC"),
                    "cumulative_return": 0.065,
                    "total_profit": 13_000.0,
                },
            ],
            "per_ticker_results": {
                "AAA": {
                    "cumulative_return": 0.07,
                    "total_profit": 7_000.0,
                },
                "BBB": {
                    "cumulative_return": 0.02,
                    "total_profit": 2_000.0,
                },
                "CCC": {
                    "cumulative_return": 0.11,
                    "total_profit": 11_000.0,
                },
            },
        }

    monkeypatch.setattr(
        "stock_predictor.cli.optimize_ticker_combinations", fake_optimize
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "backtest-portfolio",
            str(tickers_file),
            "--combination-size",
            "2",
            "--horizon",
            "1",
            "--threshold",
            "0.01",
            "--lags",
            "1",
            "--lags",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "最適組み合わせ: AAA, CCC" in result.output
    assert "ポートフォリオ累積リターン: 9.00%" in result.output
    assert "ポートフォリオ累積損益: 18000.00" in result.output
    assert "最終残高: 218000.00" in result.output
    assert "トレード回数: 4 / 勝率: 75.00%" in result.output
    assert "--- ポートフォリオ内訳 ---" in result.output
    assert "AAA: 投下資金 100000.00 / 累積損益 7000.00 / トレード 2 / 勝率 100.00%" in result.output
    assert "AAA: 累積リターン 7.00% / 累積損益 7000.00" in result.output

    assert captured["combination_size"] == 2
    assert captured["price_map"] == data_map
    backtest_kwargs = captured["backtest_kwargs"]
    assert backtest_kwargs["forecast_horizon"] == 1
    assert backtest_kwargs["threshold"] == 0.01
    assert backtest_kwargs["lags"] == (1, 2)
    assert backtest_kwargs["cv_splits"] == 5
    assert backtest_kwargs["ridge_lambda"] == pytest.approx(1e-6)
