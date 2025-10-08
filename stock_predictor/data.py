"""データ読み込みと特徴量生成モジュール."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - importガード
    import yfinance
except ModuleNotFoundError:  # pragma: no cover
    yfinance = None  # type: ignore[assignment]

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

PriceRow = dict[str, float | date]


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


def fetch_price_data_from_yfinance(
    ticker: str,
    period: str = "60d",
    interval: str = "1d",
) -> List[PriceRow]:
    """yfinanceから株価データを取得し、PriceRow形式に変換する."""

    if not ticker:
        raise ValueError("tickerを指定してください")

    if yfinance is None:
        raise RuntimeError("yfinanceがインストールされていません")

    try:
        downloaded = yfinance.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
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

    feature_names: List[str] = []
    feature_columns: List[List[float]] = []

    lags_sorted = sorted(set(int(lag) for lag in lags))
    for lag in lags_sorted:
        if lag <= 0:
            raise ValueError("ラグは正の整数で指定してください")
        shifted_close = [float("nan")] * lag + closes[:-lag]
        feature_columns.append(shifted_close)
        feature_names.append(f"lag_{lag}_close")

        pct_changes = [float("nan")] * lag
        for i in range(lag, len(closes)):
            prev = closes[i - lag]
            pct_changes.append((closes[i] - prev) / prev if prev != 0 else 0.0)
        feature_columns.append(pct_changes)
        feature_names.append(f"lag_{lag}_return")

    # 1日の値動き
    daily_return = [0.0] + [
        (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] != 0 else 0.0
        for i in range(1, len(closes))
    ]
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

    # 出来高のZスコア(5日)
    window = 5
    if len(volumes) < window:
        volume_z = [float("nan")] * len(volumes)
    else:
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

    # ターゲット
    targets: List[float] = []
    for i in range(len(closes)):
        future_index = i + forecast_horizon
        if future_index < len(closes):
            targets.append(closes[future_index])
        else:
            targets.append(float("nan"))

    # 有効サンプルのみ残す
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