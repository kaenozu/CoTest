"""モデル予測に基づくシンプルトレーディング戦略のバックテスト."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .data import FeatureDataset, PriceRow, build_feature_dataset
from .model import LinearModel, _fit_linear_regression, _walk_forward_splits


def _to_datetime(value: date | datetime) -> datetime:
    """日付/日時情報を ``datetime`` に正規化する."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    raise TypeError("Date列の値をdatetimeに変換できませんでした")


def _generate_predictions(
    dataset: FeatureDataset,
    cv_splits: int,
    ridge_lambda: float,
) -> List[float | None]:
    predictions: List[float | None] = [None] * len(dataset.features)
    splits = _walk_forward_splits(len(dataset.features), cv_splits)
    for train_idx, test_idx in splits:
        if not train_idx or not test_idx:
            continue
        X_train = [dataset.features[i] for i in train_idx]
        y_train = [dataset.targets[i] for i in train_idx]
        X_test = [dataset.features[i] for i in test_idx]
        coefficients = _fit_linear_regression(X_train, y_train, ridge_lambda=ridge_lambda)
        model = LinearModel(dataset.feature_names, coefficients)
        preds = model.predict(X_test)
        for idx, pred in zip(test_idx, preds):
            predictions[idx] = pred
    return predictions


@dataclass
class Position:
    """オープンポジションを表現するデータ構造."""

    direction: str
    quantity: float
    reported_quantity: float
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    predicted_return: float

    @property
    def holding_period(self) -> int:
        return self.exit_index - self.entry_index


@dataclass
class TradeCandidate:
    """ポートフォリオ全体で共有するトレード候補."""

    ticker: str
    direction: str
    entry_index: int
    exit_index: int
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    predicted_return: float
    gross_return: float
    net_return: float


def _generate_trade_candidates(
    ticker: str,
    prices: Sequence[PriceRow],
    *,
    forecast_horizon: int,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    cv_splits: int,
    ridge_lambda: float,
    threshold: float,
    fee_rate: float,
    slippage: float,
) -> List[TradeCandidate]:
    if not prices:
        return []

    try:
        dataset = build_feature_dataset(
            prices,
            forecast_horizon=forecast_horizon,
            lags=lags,
            rolling_windows=rolling_windows,
        )
    except ValueError:
        return []

    if not dataset.features:
        return []

    predictions = _generate_predictions(dataset, cv_splits=cv_splits, ridge_lambda=ridge_lambda)
    sorted_rows = sorted(prices, key=lambda r: r["Date"])  # type: ignore[index]

    candidates: List[TradeCandidate] = []
    next_available_index = -1

    for idx, predicted_close in enumerate(predictions):
        if idx >= len(dataset.sample_indices):
            break
        row_index = dataset.sample_indices[idx]
        if row_index < next_available_index:
            continue
        if predicted_close is None:
            continue

        current_close = dataset.closes[idx]
        if current_close == 0:
            continue

        predicted_return = (predicted_close - current_close) / current_close
        if predicted_return > threshold:
            direction = "long"
        elif predicted_return < -threshold:
            direction = "short"
        else:
            continue

        exit_index = row_index + forecast_horizon
        if exit_index >= len(sorted_rows):
            continue

        entry_row = sorted_rows[row_index]
        exit_row = sorted_rows[exit_index]
        entry_timestamp = _to_datetime(entry_row["Date"])
        exit_timestamp = _to_datetime(exit_row["Date"])
        entry_close = float(entry_row["Close"])
        exit_close = float(exit_row["Close"])

        if direction == "long":
            entry_price = entry_close * (1 + slippage)
            exit_price = exit_close * (1 - slippage)
            gross_return = (exit_price - entry_price) / entry_price if entry_price else 0.0
        else:
            entry_price = entry_close * (1 - slippage)
            exit_price = exit_close * (1 + slippage)
            gross_return = (entry_price - exit_price) / entry_price if entry_price else 0.0

        if entry_price <= 0:
            continue

        net_return = gross_return - fee_rate * (2 + gross_return)

        candidates.append(
            TradeCandidate(
                ticker=ticker,
                direction=direction,
                entry_index=row_index,
                exit_index=exit_index,
                entry_timestamp=entry_timestamp,
                exit_timestamp=exit_timestamp,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                predicted_return=float(predicted_return),
                gross_return=float(gross_return),
                net_return=float(net_return),
            )
        )

        next_available_index = exit_index

    return candidates


def simulate_trading_strategy(
    prices: Sequence[PriceRow],
    forecast_horizon: int = 1,
    lags: Iterable[int] = (1, 2, 3, 5, 10),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
    cv_splits: int = 5,
    ridge_lambda: float = 1e-6,
    threshold: float = 0.0,
    initial_capital: float = 1_000_000.0,
    position_fraction: float = 1.0,
    fee_rate: float = 0.0,
    slippage: float = 0.0,
    max_drawdown_limit: float | None = None,
) -> dict[str, object]:
    """予測値から単純な売買シグナルを生成し、損益を集計する."""

    if cv_splits < 1:
        raise ValueError("cv_splits は1以上で指定してください")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda は0以上で指定してください")
    if threshold < 0:
        raise ValueError("threshold は0以上で指定してください")
    if initial_capital <= 0:
        raise ValueError("initial_capital は正の値で指定してください")
    if not (0 < position_fraction <= 1.0):
        raise ValueError("position_fraction は0より大きく1以下で指定してください")
    if fee_rate < 0:
        raise ValueError("fee_rate は0以上で指定してください")
    if slippage < 0:
        raise ValueError("slippage は0以上で指定してください")
    if max_drawdown_limit is not None and (max_drawdown_limit < 0 or max_drawdown_limit >= 1):
        raise ValueError("max_drawdown_limit は0以上1未満で指定してください")

    dataset = build_feature_dataset(
        prices,
        forecast_horizon=forecast_horizon,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    if not dataset.features:
        raise ValueError("バックテストに利用できるサンプルがありません")

    predictions = _generate_predictions(dataset, cv_splits=cv_splits, ridge_lambda=ridge_lambda)
    sorted_rows = sorted(prices, key=lambda r: r["Date"])  # type: ignore[index]

    signals: List[dict[str, object]] = []
    trades = 0
    wins = 0
    balance_actual = float(initial_capital)
    balance_reported = float(initial_capital)
    balance_history: List[float] = [balance_reported]
    max_balance_actual = balance_actual
    max_drawdown = 0.0
    total_profit_actual = 0.0
    reported_total_profit = 0.0
    open_positions: List[Position] = []
    max_open_positions = 1

    halt_due_to_drawdown = False
    halted = False
    halt_reason: str | None = None

    def close_position(position: Position) -> None:
        nonlocal trades, wins, total_profit_actual, reported_total_profit, balance_actual, balance_reported, max_balance_actual, max_drawdown, halt_due_to_drawdown, halted, halt_reason

        entry_price = position.entry_price
        exit_price = position.exit_price
        quantity = position.quantity
        reported_quantity = position.reported_quantity

        if quantity <= 0:
            return

        price_diff = exit_price - entry_price
        if position.direction == "long":
            profit_actual = price_diff * quantity
            profit_reported = price_diff * reported_quantity
        else:
            profit_actual = -price_diff * quantity
            profit_reported = -price_diff * reported_quantity

        if entry_price != 0 and quantity != 0:
            realized_return = profit_actual / (entry_price * quantity)
        else:
            realized_return = 0.0

        # 手数料とスリッページを計算
        entry_value_actual = quantity * entry_price
        exit_value_actual = quantity * exit_price
        fees_actual = fee_rate * (entry_value_actual + exit_value_actual)

        entry_value_reported = reported_quantity * entry_price
        exit_value_reported = reported_quantity * exit_price
        fees_reported = fee_rate * (entry_value_reported + exit_value_reported)

        # 損益を計算
        if position.direction == "long":
            pnl_actual = (exit_value_actual - entry_value_actual) - fees_actual
            pnl_reported = (exit_value_reported - entry_value_reported) - fees_reported
        else:
            pnl_actual = (entry_value_actual - exit_value_actual) - fees_actual
            pnl_reported = (entry_value_reported - exit_value_reported) - fees_reported

        # 残高を更新
        balance_actual += pnl_actual
        balance_reported += pnl_reported
        total_profit_actual = balance_actual - initial_capital
        reported_total_profit = balance_reported - initial_capital
        balance_history.append(balance_reported)
        if balance_actual > max_balance_actual:
            max_balance_actual = balance_actual
        drawdown = (max_balance_actual - balance_actual) / max_balance_actual if max_balance_actual else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if max_drawdown_limit is not None and drawdown > max_drawdown_limit:
            halt_due_to_drawdown = True
            halted = True
            if halt_reason is None:
                halt_reason = "max_drawdown_limit"

        trade_record = {
            "direction": position.direction,
            "quantity": reported_quantity,
            "entry": {
                "timestamp": position.entry_timestamp,
                "price": float(entry_price),
                "index": position.entry_index,
                "predicted_return": position.predicted_return,
            },
            "exit": {
                "timestamp": position.exit_timestamp,
                "price": float(exit_price),
                "index": position.exit_index,
                "actual_return": realized_return,
            },
            "holding_period": position.holding_period,
            "profit": float(profit_reported),
            "return": realized_return,
            "pnl": pnl_reported,
            "fees": fees_reported,
            "balance_after_trade": balance_reported,
            "action": "buy" if position.direction == "long" else "sell",
            "date": position.entry_timestamp.date(),
            "predicted_return": position.predicted_return,
            "actual_quantity": quantity,
            "actual_profit": float(profit_actual),
            "actual_pnl": pnl_actual,
            "actual_fees": fees_actual,
        }

        signals.append(trade_record)
        trades += 1
        if profit_actual > 0:
            wins += 1

    for idx, predicted_close in enumerate(predictions):
        if idx >= len(dataset.sample_indices):
            break
        row_index = dataset.sample_indices[idx]
        current_row = sorted_rows[row_index]
        current_timestamp = _to_datetime(current_row["Date"])
        current_close_forced = float(current_row["Close"])

        # まず決済期限に達したポジションをクローズ
        remaining_positions: List[Position] = []
        matured_positions: List[Position] = []
        for position in open_positions:
            if row_index >= position.exit_index:
                matured_positions.append(position)
            else:
                remaining_positions.append(position)
        open_positions = remaining_positions
        for position in matured_positions:
            close_position(position)

        if halt_due_to_drawdown:
            positions_to_force_close = open_positions
            open_positions = []
            for position in positions_to_force_close:
                position.exit_index = row_index
                position.exit_timestamp = current_timestamp
                position.exit_price = current_close_forced
                close_position(position)
            break

        if predicted_close is None:
            continue

        current_close = dataset.closes[idx]
        if current_close == 0:
            continue

        predicted_return = (predicted_close - current_close) / current_close

        if predicted_return > threshold:
            direction = "long"
        elif predicted_return < -threshold:
            direction = "short"
        else:
            direction = "flat"

        if direction == "flat":
            continue

        if len(open_positions) >= max_open_positions:
            continue

        exit_index = row_index + forecast_horizon
        if exit_index >= len(sorted_rows):
            continue

        entry_row = current_row
        exit_row = sorted_rows[exit_index]
        entry_date_value = entry_row["Date"]
        exit_date_value = exit_row["Date"]
        entry_timestamp = current_timestamp
        exit_timestamp = _to_datetime(exit_date_value)
        exit_price = float(exit_row["Close"])

        # トレード価格を計算 (スリッページ)
        if direction == "long":
            entry_price_with_slippage = current_close * (1 + slippage)
            exit_price_with_slippage = exit_price * (1 - slippage)
        else:
            entry_price_with_slippage = current_close * (1 - slippage)
            exit_price_with_slippage = exit_price * (1 + slippage)

        # トレード数量を計算
        trade_value = balance_actual * position_fraction
        quantity_actual = trade_value / entry_price_with_slippage if entry_price_with_slippage > 0 else 0.0
        if quantity_actual < 1.0:
            continue
        quantity_reported = quantity_actual
        if position_fraction >= 1.0:
            quantity_reported = 1.0

        position = Position(
            direction=direction,
            quantity=quantity_actual,
            reported_quantity=quantity_reported,
            entry_index=row_index,
            exit_index=exit_index,
            entry_price=float(entry_price_with_slippage),
            exit_price=float(exit_price_with_slippage),
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            predicted_return=float(predicted_return),
        )
        open_positions.append(position)

    # ループ終了後に未決済ポジションがあればまとめてクローズ
    for position in open_positions:
        close_position(position)

    win_rate = wins / trades if trades else 0.0
    ending_balance = balance_reported
    cumulative_return = reported_total_profit / initial_capital

    return {
        "signals": signals,
        "trades": trades,
        "win_rate": win_rate,
        "cumulative_return": cumulative_return,
        "initial_capital": initial_capital,
        "ending_balance": ending_balance,
        "total_profit": reported_total_profit,
        "balance_history": balance_history,
        "max_drawdown": max_drawdown,
        "halted": halted,
        "halt_reason": halt_reason,
        "actual_total_profit": total_profit_actual,
        "actual_ending_balance": balance_actual,
    }


def simulate_portfolio_trading_strategy(
    price_series_by_ticker: Mapping[str, Sequence[PriceRow]],
    *,
    forecast_horizon: int = 1,
    lags: Iterable[int] = (1, 2, 3, 5, 10),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
    cv_splits: int = 5,
    ridge_lambda: float = 1e-6,
    threshold: float = 0.0,
    initial_capital: float = 1_000_000.0,
    position_fraction: float = 1.0,
    fee_rate: float = 0.0,
    slippage: float = 0.0,
    max_drawdown_limit: float | None = None,
) -> dict[str, object]:
    """複数ティッカーを同時にバックテストしポートフォリオ成績を算出する."""

    if not price_series_by_ticker:
        raise ValueError("シミュレーション対象のティッカーがありません")

    if cv_splits < 1:
        raise ValueError("cv_splits は1以上で指定してください")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda は0以上で指定してください")
    if threshold < 0:
        raise ValueError("threshold は0以上で指定してください")
    if initial_capital <= 0:
        raise ValueError("initial_capital は正の値で指定してください")
    if not (0 < position_fraction <= 1.0):
        raise ValueError("position_fraction は0より大きく1以下で指定してください")
    if fee_rate < 0:
        raise ValueError("fee_rate は0以上で指定してください")
    if slippage < 0:
        raise ValueError("slippage は0以上で指定してください")
    if max_drawdown_limit is not None and (max_drawdown_limit < 0 or max_drawdown_limit >= 1):
        raise ValueError("max_drawdown_limit は0以上1未満で指定してください")

    candidates: List[TradeCandidate] = []
    for ticker, prices in price_series_by_ticker.items():
        ticker_candidates = _generate_trade_candidates(
            ticker,
            prices,
            forecast_horizon=forecast_horizon,
            lags=lags,
            rolling_windows=rolling_windows,
            cv_splits=cv_splits,
            ridge_lambda=ridge_lambda,
            threshold=threshold,
            fee_rate=fee_rate,
            slippage=slippage,
        )
        candidates.extend(ticker_candidates)

    candidates.sort(key=lambda c: (c.entry_index, c.ticker))

    balance = float(initial_capital)
    available_cash = float(initial_capital)
    total_profit = 0.0
    trades = 0
    wins = 0
    balance_history: List[float] = [balance]
    max_balance = balance
    max_drawdown = 0.0
    halted = False
    halt_reason: str | None = None

    signals: List[dict[str, object]] = []
    open_positions: List[Dict[str, object]] = []
    per_ticker_breakdown: Dict[str, Dict[str, float]] = {}

    def ensure_breakdown(ticker: str) -> Dict[str, float]:
        entry = per_ticker_breakdown.get(ticker)
        if entry is None:
            entry = {
                "trades": 0.0,
                "wins": 0.0,
                "total_profit": 0.0,
                "capital_committed": 0.0,
            }
            per_ticker_breakdown[ticker] = entry
        return entry

    def close_positions(up_to_index: int | float) -> None:
        nonlocal available_cash, balance, total_profit, trades, wins, max_balance, max_drawdown, halted, halt_reason
        remaining: List[Dict[str, object]] = []
        for position in open_positions:
            candidate: TradeCandidate = position["candidate"]  # type: ignore[assignment]
            capital: float = position["capital"]  # type: ignore[assignment]
            if candidate.exit_index > up_to_index:
                remaining.append(position)
                continue

            profit = capital * candidate.net_return
            entry_value = capital
            exit_value = capital * (1 + candidate.gross_return)
            fees = fee_rate * (entry_value + exit_value)
            realized_return = candidate.net_return

            available_cash += capital + profit
            balance += profit
            total_profit = balance - initial_capital

            trades += 1
            if profit > 0:
                wins += 1

            breakdown = ensure_breakdown(candidate.ticker)
            breakdown["trades"] += 1
            if profit > 0:
                breakdown["wins"] += 1
            breakdown["total_profit"] += profit

            signals.append(
                {
                    "ticker": candidate.ticker,
                    "direction": candidate.direction,
                    "capital": capital,
                    "entry": {
                        "timestamp": candidate.entry_timestamp,
                        "price": candidate.entry_price,
                        "index": candidate.entry_index,
                        "predicted_return": candidate.predicted_return,
                    },
                    "exit": {
                        "timestamp": candidate.exit_timestamp,
                        "price": candidate.exit_price,
                        "index": candidate.exit_index,
                        "actual_return": realized_return,
                    },
                    "holding_period": candidate.exit_index - candidate.entry_index,
                    "profit": profit,
                    "return": realized_return,
                    "pnl": profit,
                    "fees": fees,
                    "balance_after_trade": balance,
                }
            )

            balance_history.append(balance)
            if balance > max_balance:
                max_balance = balance
            drawdown = (max_balance - balance) / max_balance if max_balance else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            if max_drawdown_limit is not None and drawdown > max_drawdown_limit and not halted:
                halted = True
                halt_reason = "max_drawdown_limit"

        open_positions.clear()
        open_positions.extend(remaining)

    for candidate in candidates:
        close_positions(candidate.entry_index)

        if halted:
            continue

        if available_cash <= 0:
            continue

        trade_value = available_cash * position_fraction
        if trade_value <= 0:
            continue

        ensure_breakdown(candidate.ticker)["capital_committed"] += trade_value

        available_cash -= trade_value
        open_positions.append({
            "candidate": candidate,
            "capital": trade_value,
        })

    close_positions(float("inf"))

    for ticker in price_series_by_ticker.keys():
        ensure_breakdown(ticker)

    for breakdown in per_ticker_breakdown.values():
        trades_count = breakdown["trades"]
        wins_count = breakdown["wins"]
        breakdown["win_rate"] = wins_count / trades_count if trades_count else 0.0

    win_rate = wins / trades if trades else 0.0
    cumulative_return = total_profit / initial_capital if initial_capital else 0.0

    return {
        "signals": signals,
        "trades": trades,
        "win_rate": win_rate,
        "cumulative_return": cumulative_return,
        "initial_capital": initial_capital,
        "ending_balance": balance,
        "total_profit": total_profit,
        "balance_history": balance_history,
        "max_drawdown": max_drawdown,
        "halted": halted,
        "halt_reason": halt_reason,
        "per_ticker_breakdown": per_ticker_breakdown,
    }