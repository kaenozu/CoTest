"""データ読み込みと特徴量生成モジュール."""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Literal, Sequence, Tuple

try:  # pragma: no cover - importガード
    import yfinance
except ModuleNotFoundError:  # pragma: no cover
    yfinance = None  # type: ignore[assignment]

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

PriceRow = dict[str, float | date | datetime]


@dataclass
class LiveStreamState:
    """ライブストリームの状態を保持し再接続や欠損補完を制御する."""

    max_retries: int = 3
    retry_attempt: int = 0
    last_row: PriceRow | None = None

    def reset_retry(self) -> None:
        self.retry_attempt = 0

    def register_retry(self) -> int:
        self.retry_attempt += 1
        return self.retry_attempt

    def can_retry(self) -> bool:
        return self.retry_attempt <= self.max_retries

    def remember(self, row: PriceRow) -> None:
        self.last_row = row


@dataclass
class FeatureDataset:
    """特徴量生成の結果をまとめたデータ構造."""

    features: List[List[float]]
    targets: List[float]
    feature_names: List[str]
    sample_indices: List[int]
    closes: List[float]


def _ensure_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        if isinstance(converted, datetime):
            return converted.date()
    if hasattr(value, "date"):
        converted = value.date()
        if isinstance(converted, date):
            return converted
    raise ValueError("日付インデックスを解釈できませんでした")


def load_price_data(path: str | Path) -> List[PriceRow]:
    """株価CSVを読み込み、日付順に整形する."""
    file_path = Path(path)
    with file_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError("ヘッダー行が存在しません")
        if reader.fieldnames != REQUIRED_COLUMNS:
            missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise ValueError(f"必要な列が不足しています: {', '.join(sorted(missing))}")
        rows: List[PriceRow] = []
        for row in reader:
            rows.append(
                {
                    "Date": datetime.strptime(row["Date"], "%Y-%m-%d").date(),
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                    "Volume": float(row["Volume"]),
                }
            )
    rows.sort(key=lambda r: r["Date"])  # type: ignore[index]
    return rows


AdjustMode = Literal["none", "auto", "manual"]


def fetch_price_data_from_yfinance(
    ticker: str,
    period: str = "60d",
    interval: str = "1d",
    adjust: AdjustMode = "none",
) -> List[PriceRow]:
    """yfinanceから株価データを取得し、PriceRow形式に変換する."""

    if not ticker:
        raise ValueError("tickerを指定してください")

    if yfinance is None:
        raise RuntimeError("yfinanceがインストールされていません")

    if adjust not in {"none", "auto", "manual"}:
        raise ValueError("adjust は 'none'・'auto'・'manual' のいずれかで指定してください")

    auto_adjust = adjust == "auto"

    try:
        downloaded = yfinance.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=auto_adjust,
            actions=True,
        )
    except Exception as exc:  # pragma: no cover - 例外経路を簡潔に確保
        raise RuntimeError("yfinanceからのデータ取得に失敗しました") from exc

    if getattr(downloaded, "empty", True):
        raise ValueError("指定条件で取得できる価格データがありません")

    cleaned = downloaded.dropna()
    rows: List[PriceRow] = []
    for index, values in cleaned.iterrows():
        try:
            row_date = _ensure_date(index)
            open_price = float(values["Open"])
            high_price = float(values["High"])
            low_price = float(values["Low"])
            close_price = float(values["Close"])
            volume = float(values["Volume"])
        except (KeyError, TypeError, ValueError):
            continue

        adj_close = None
        if "Adj Close" in values and values["Adj Close"] is not None:
            try:
                adj_close = float(values["Adj Close"])
            except (TypeError, ValueError):
                adj_close = None

        if adjust == "manual" and adj_close is not None:
            if close_price == 0:
                factor = 1.0
            else:
                factor = adj_close / close_price
            if not math.isfinite(factor) or factor <= 0:
                factor = 1.0
            open_price *= factor
            high_price *= factor
            low_price *= factor
            close_price = adj_close
        elif adjust == "auto" and adj_close is not None:
            close_price = adj_close

        rows.append(
            {
                "Date": row_date,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
        )

    rows.sort(key=lambda r: r["Date"])  # type: ignore[index]

    if not rows:
        raise ValueError("有効な価格データが取得できませんでした")

    return rows


def stream_live_prices(
    client: Any,
    ticker: str,
    *,
    limit: int | None = None,
    timestamp_key: str = "timestamp",
    timestamp_converter: Callable[[Any], Any] | None = None,
    state: LiveStreamState | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> Iterator[PriceRow]:
    """外部クライアントからライブ株価を購読しPriceRow形式で返す."""

    if not ticker:
        raise ValueError("tickerを指定してください")
    if client is None or not hasattr(client, "stream_prices"):
        raise ValueError("クライアントはstream_pricesメソッドを持つ必要があります")

    converter = timestamp_converter or (lambda value: value)
    count = 0
    stream_state = state or LiveStreamState()

    def notify(event: str, payload: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event, payload)

    while True:
        try:
            source = iter(client.stream_prices(ticker))
        except Exception as exc:  # pragma: no cover - 起動時の例外経路の確保
            attempt = stream_state.register_retry()
            if not stream_state.can_retry():
                raise
            notify("reconnect", {"attempt": attempt, "error": str(exc)})
            continue

        while True:
            try:
                payload = next(source)
            except StopIteration:
                return
            except Exception as exc:
                attempt = stream_state.register_retry()
                if not stream_state.can_retry():
                    raise
                notify("reconnect", {"attempt": attempt, "error": str(exc)})
                break

            if payload is None:
                notify("drop", {"reason": "empty_payload"})
                continue

            raw_timestamp = payload.get(timestamp_key)
            if raw_timestamp is None:
                notify("drop", {"reason": "missing_timestamp"})
                continue

            timestamp = converter(raw_timestamp)
            try:
                row_date = _ensure_date(timestamp)
            except ValueError:
                notify("drop", {"reason": "invalid_timestamp", "value": raw_timestamp})
                continue

            price_value = payload.get("price", payload.get("close"))
            try:
                close_price = float(price_value)
            except (TypeError, ValueError):
                notify("drop", {"reason": "invalid_price", "value": price_value})
                continue

            open_price = payload.get("open")
            high_price = payload.get("high")
            low_price = payload.get("low")
            volume = payload.get("volume")
            has_volume_key = "volume" in payload

            missing_fields: list[str] = []
            if stream_state.last_row is not None:
                prev_close = float(stream_state.last_row["Close"])
                if open_price is None:
                    open_price = prev_close
                    missing_fields.append("open")
                if high_price is None:
                    high_price = max(prev_close, close_price)
                    missing_fields.append("high")
                if low_price is None:
                    low_price = min(prev_close, close_price)
                    missing_fields.append("low")
                if volume is None and not has_volume_key:
                    volume = stream_state.last_row.get("Volume", 0.0)
                    missing_fields.append("volume")

            if open_price is None:
                open_price = close_price
            if high_price is None:
                high_price = max(open_price, close_price)
            if low_price is None:
                low_price = min(open_price, close_price)
            if volume is None:
                volume = 0.0

            try:
                open_value = float(open_price)
                high_value = float(high_price)
                low_value = float(low_price)
                volume_value = float(volume)
            except (TypeError, ValueError):
                notify("drop", {"reason": "invalid_numeric", "payload": payload})
                continue

            row: PriceRow = {
                "Date": row_date,
                "Open": open_value,
                "High": high_value,
                "Low": low_value,
                "Close": close_price,
                "Volume": volume_value,
            }

            stream_state.remember(row)
            stream_state.reset_retry()

            if missing_fields:
                notify("fill", {"fields": missing_fields, "timestamp": row_date})

            yield row

            count += 1
            if limit is not None and count >= limit:
                return


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    out: List[float] = []
    cumulative = 0.0
    for i, value in enumerate(values):
        cumulative += value
        if i >= window:
            cumulative -= values[i - window]
        if i + 1 >= window:
            out.append(cumulative / window)
        else:
            out.append(float("nan"))
    return out


def _exponential_moving_average(values: Sequence[float], span: int) -> List[float]:
    alpha = 2 / (span + 1)
    ema: List[float] = []
    prev = values[0]
    for value in values:
        prev = prev + alpha * (value - prev)
        ema.append(prev)
    return ema


def _rolling_std(values: Sequence[float], window: int) -> List[float]:
    import math

    out: List[float] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(float("nan"))
            continue
        segment = values[i + 1 - window : i + 1]
        mean = sum(segment) / window
        variance = sum((x - mean) ** 2 for x in segment) / window
        out.append(math.sqrt(variance))
    return out


def _calculate_feature_columns(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
    forecast_horizon: int,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> tuple[List[List[float]], List[str]]:
    feature_names: List[str] = []
    feature_columns: List[List[float]] = []

    lags_sorted = sorted(set(int(lag) for lag in lags))
    max_valid_lag = max(len(closes) - forecast_horizon - 1, 0)
    for lag in lags_sorted:
        if lag <= 0:
            raise ValueError("ラグは正の整数で指定してください")
        if lag > max_valid_lag:
            warnings.warn(
                f"利用可能な履歴({max_valid_lag})を超えるラグ {lag} はスキップします",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        shifted_close = [float("nan")] * lag + list(closes[:-lag])
        feature_columns.append(shifted_close)
        feature_names.append(f"lag_{lag}_close")

        pct_changes = [float("nan")] * lag
        for i in range(lag, len(closes)):
            prev = closes[i - lag]
            pct_changes.append((closes[i] - prev) / prev if prev != 0 else 0.0)
        feature_columns.append(pct_changes)
        feature_names.append(f"lag_{lag}_return")

    daily_return = [0.0]
    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        current = closes[i]
        daily_return.append((current - prev_close) / prev_close if prev_close != 0 else 0.0)
    feature_columns.append(daily_return)
    feature_names.append("daily_return")

    price_range = [
        (highs[i] - lows[i]) / closes[i] if closes[i] != 0 else 0.0 for i in range(len(closes))
    ]
    feature_columns.append(price_range)
    feature_names.append("price_range")

    for window in sorted(set(int(w) for w in rolling_windows)):
        if window <= 1:
            raise ValueError("移動窓幅は2以上で指定してください")
        if window > len(closes):
            continue
        feature_columns.append(_moving_average(closes, window))
        feature_names.append(f"sma_{window}")
        feature_columns.append(_exponential_moving_average(closes, window))
        feature_names.append(f"ema_{window}")
        feature_columns.append(_rolling_std(daily_return, window))
        feature_names.append(f"volatility_{window}")

    window = 5
    if len(volumes) >= window:
        volume_z = []
        for i in range(len(volumes)):
            if i + 1 < window:
                volume_z.append(float("nan"))
                continue
            segment = volumes[i + 1 - window : i + 1]
            mean = sum(segment) / window
            variance = sum((x - mean) ** 2 for x in segment) / window
            std = variance ** 0.5
            volume_z.append((volumes[i] - mean) / std if std != 0 else 0.0)
        feature_columns.append(volume_z)
        feature_names.append("volume_zscore_5")

    return feature_columns, feature_names


def build_feature_dataset(
    prices: Sequence[PriceRow],
    forecast_horizon: int = 1,
    lags: Iterable[int] = (1, 2, 3, 5, 10),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
) -> FeatureDataset:
    """特徴量行列とターゲットを生成する."""
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon は1以上である必要があります")

    sorted_rows = sorted(prices, key=lambda r: r["Date"])  # type: ignore[index]
    closes = [float(row["Close"]) for row in sorted_rows]
    highs = [float(row["High"]) for row in sorted_rows]
    lows = [float(row["Low"]) for row in sorted_rows]
    volumes = [float(row["Volume"]) for row in sorted_rows]

    feature_columns, feature_names = _calculate_feature_columns(
        closes,
        highs,
        lows,
        volumes,
        forecast_horizon,
        lags,
        rolling_windows,
    )

    targets: List[float] = []
    for i in range(len(closes)):
        future_index = i + forecast_horizon
        if future_index < len(closes):
            targets.append(closes[future_index])
        else:
            targets.append(float("nan"))

    matrix: List[List[float]] = []
    cleaned_targets: List[float] = []
    sample_indices: List[int] = []
    closes_for_samples: List[float] = []
    for idx in range(len(closes)):
        row = [col[idx] for col in feature_columns]
        if any(value != value for value in row):  # NaNチェック
            continue
        target = targets[idx]
        if target != target:
            continue
        matrix.append(row)
        cleaned_targets.append(target)
        sample_indices.append(idx)
        closes_for_samples.append(closes[idx])

    return FeatureDataset(matrix, cleaned_targets, feature_names, sample_indices, closes_for_samples)


def build_feature_matrix(
    prices: Sequence[PriceRow],
    forecast_horizon: int = 1,
    lags: Iterable[int] = (1, 2, 3, 5, 10),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
) -> Tuple[List[List[float]], List[float], List[str]]:
    """従来互換のインターフェースで特徴量を返す."""

    dataset = build_feature_dataset(
        prices,
        forecast_horizon=forecast_horizon,
        lags=lags,
        rolling_windows=rolling_windows,
    )
    return dataset.features, dataset.targets, dataset.feature_names


def build_latest_feature_row(
    prices: Sequence[PriceRow],
    forecast_horizon: int = 1,
    lags: Iterable[int] = (1, 2, 3, 5, 10),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
) -> Tuple[List[float], List[str]]:
    """最新レコードから推論用特徴量ベクトルを構築する."""

    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon は1以上である必要があります")
    if not prices:
        raise ValueError("価格データがありません")

    sorted_rows = sorted(prices, key=lambda r: r["Date"])  # type: ignore[index]
    closes = [float(row["Close"]) for row in sorted_rows]
    highs = [float(row["High"]) for row in sorted_rows]
    lows = [float(row["Low"]) for row in sorted_rows]
    volumes = [float(row["Volume"]) for row in sorted_rows]

    feature_columns, feature_names = _calculate_feature_columns(
        closes,
        highs,
        lows,
        volumes,
        forecast_horizon,
        lags,
        rolling_windows,
    )

    latest_index = len(sorted_rows) - 1
    row = [column[latest_index] for column in feature_columns]
    if any(value != value for value in row):
        raise ValueError("最新データから特徴量を生成できませんでした")

    return row, feature_names
