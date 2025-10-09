# 運用ガイド

このドキュメントは、`stock-predictor` CLI のライブ注文ワークフローを運用する際の手順をまとめたものです。

## 概要

- `trade` サブコマンドがブローカーAPIを介した注文送信と状態監視を担当します。
- ブローカー接続に関する設定や通知先は TOML 形式の設定ファイルで管理します。
- 異常が検出された場合はロガー経由で通知され、CLIは非0終了コードを返します。

## 設定ファイル

既定では `ops/trading.toml` が読み込まれます。別ファイルを利用する場合は `--config` オプションでパスを指定してください。

```toml
[logging]
# ログレベルを指定します。INFO/DEBUG/WARNING/ERROR などが利用できます。
level = "INFO"

[notifications]
# 通知チャネル定義。標準実装ではロガー経由で通知します。
channel = "operations"

[broker]
# ブローカー固有の設定。ファクトリー関数にキーワード引数として渡されます。
base_url = "https://paper-trade.example.com"
api_key = "YOUR_API_KEY"
```

## ブローカークライアントの実装要件

- `place_order(ticker: str, side: str, quantity: float, order_type: str = "market", **kwargs) -> str`
  - ブローカーに注文を送信し、注文IDを返却します。
- `stream_order_states(order_id: str) -> Iterable[dict]`
  - 指定した注文の状態更新を逐次返します。
  - 辞書には少なくとも `status` キーが含まれている必要があります。
  - `status` が `rejected`/`cancelled`/`failed` の場合は `reason` を併せて返すことを推奨します。

## CLI 実行例

```bash
stock-predictor trade \
  --broker trading.client:build_paper_client \
  --ticker AAPL \
  --side buy \
  --quantity 10 \
  --config ops/trading.toml
```

実行時には以下のような挙動が期待されます。

1. 設定ファイルを読み込みロガーと通知チャネルを初期化します。
2. `--broker` で指定されたファクトリーからクライアントを生成します。
3. `place_order` で注文を送信し、IDを表示します。
4. `stream_order_states` を用いて状態を監視し、`filled` などの状態を逐次表示します。
5. `rejected` などの異常状態が検出された場合、通知を発行して処理を中断します。

## エラー処理

- 注文送信に失敗した場合は通知を出しつつ `ClickException` により失敗を報告します。
- 状態監視中に異常が検出された場合はその時点で通知を発行し、CLIは非0終了コードで終了します。
- テストでは `rejected` ステータスを受信した際にエラーログと通知が発火することを検証しています (`tests/test_cli.py`)。

## ログと通知

- `stock_predictor.operations` ロガーにより標準出力へ運用ログを出力します。
- 標準の `ConsoleNotifier` はロガーを利用した通知のみを提供します。外部サービス連携が必要な場合は `Notifier` プロトコルを満たす別実装を用意してください。
