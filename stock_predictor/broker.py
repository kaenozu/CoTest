"""ブローカー連携のための共通ユーティリティ."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Python<3.11用フォールバック
    import tomli as tomllib  # type: ignore


class BrokerConfigurationError(RuntimeError):
    """ブローカーの設定に関する例外."""


class BrokerClient(Protocol):
    """ブローカークライアントが満たすべきインターフェース."""

    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        **kwargs: Any,
    ) -> str:
        ...

    def stream_order_states(self, order_id: str) -> Iterable[Dict[str, Any]]:
        ...


class Notifier(Protocol):
    """通知チャネルに必要なインターフェース."""

    def notify(self, message: str, *, level: str = "info") -> None:
        ...


@dataclass
class ConsoleNotifier:
    """ログ経由で通知を行う単純な実装."""

    logger: logging.Logger

    def notify(self, message: str, *, level: str = "info") -> None:
        level = (level or "info").lower()
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)


def load_runtime_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """運用設定ファイルを読み込み辞書形式で返す.

    `path` が省略された場合は既定の `ops/trading.toml` を探し、
    見つからなければ空の設定を返す。
    """

    candidates: Iterable[Path]
    if path is not None:
        candidates = (path,)
    else:
        candidates = (Path("ops/trading.toml"), Path("config/trading.toml"))

    config_data: Dict[str, Any] = {}
    for candidate in candidates:
        if candidate.exists():
            config_data = tomllib.loads(candidate.read_text())
            break

    logging_cfg = config_data.get("logging", {}) if isinstance(config_data, dict) else {}
    notifications_cfg = config_data.get("notifications", {}) if isinstance(config_data, dict) else {}
    broker_cfg = config_data.get("broker", {}) if isinstance(config_data, dict) else {}

    return {
        "logging": dict(logging_cfg),
        "notifications": dict(notifications_cfg),
        "broker": dict(broker_cfg),
    }


def get_operation_logger(settings: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """運用向けのロガーを初期化して返す."""

    settings = settings or {}
    logger = logging.getLogger("stock_predictor.operations")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    level_name = str(settings.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    return logger


def build_notifier(settings: Optional[Dict[str, Any]] = None, *, logger: Optional[logging.Logger] = None) -> Notifier:
    """通知チャネルを構築する."""

    logger = logger or get_operation_logger(settings)
    return ConsoleNotifier(logger=logger)


def _split_identifier(identifier: str) -> tuple[str, str]:
    if ":" in identifier:
        return identifier.split(":", 1)
    if "." in identifier:
        return identifier.rsplit(".", 1)
    raise BrokerConfigurationError(
        "ブローカーは module:attribute または module.attribute 形式で指定してください"
    )


def load_broker_client(identifier: str, settings: Optional[Dict[str, Any]] = None) -> BrokerClient:
    """指定された識別子からブローカークライアントを構築する."""

    if not identifier:
        raise BrokerConfigurationError("ブローカー識別子を指定してください")

    module_name, attr = _split_identifier(identifier)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - import経路の保持
        raise BrokerConfigurationError(
            f"モジュール {module_name} を読み込めませんでした"
        ) from exc

    try:
        factory = getattr(module, attr)
    except AttributeError as exc:
        raise BrokerConfigurationError(
            f"{module_name} に {attr} が見つかりません"
        ) from exc

    settings = settings or {}

    client = factory(**settings) if callable(factory) else factory
    missing = [
        name
        for name in ("place_order", "stream_order_states")
        if not hasattr(client, name)
    ]
    if missing:
        raise BrokerConfigurationError(
            "ブローカークライアントには place_order と stream_order_states が必要です"
        )
    return client  # type: ignore[return-value]
