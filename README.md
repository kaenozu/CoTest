# Stock Predictor CLI

株価CSVまたはyfinance経由で取得したデータを用いて終値を予測するCLIツールです。

## インストール

```bash
pip install -e .
```

## 使い方

### CSVファイルを利用する場合

```bash
stock-predictor forecast data/prices.csv --horizon 1 --lags 1 --lags 2 --lags 5
```

### yfinanceから自動取得する場合

- `--ticker` を指定するとCSVパスは指定できません。
- `--period` と `--interval` は yfinance のパラメータに準拠します。既定値は `60d` と `1d` です。
- `--adjust` で価格調整方法を切り替えられます。`auto` は yfinance の自動調整、`manual` は取得後に分割・配当を反映、既定値の `none` は未調整値を使用します。

```bash
stock-predictor forecast --ticker AAPL --period 60d --interval 1d --adjust manual
```

### 複数ティッカーのポートフォリオ最適化

1行1ティッカーで列挙したテキストファイルを用意すると、`backtest-portfolio` サブコマンドで組み合わせ探索が行えます。候補のティッカーすべてについてバックテストを実行し、累積リターンが最大になる組み合わせとランキングを表示します。

```text
AAPL
MSFT
GOOG
AMZN
```

```bash
stock-predictor backtest-portfolio tickers.txt \
  --combination-size 2 \
  --horizon 1 \
  --threshold 0.005 \
  --period 60d \
  --interval 1d
```

`--lags` や `--ridge` などのバックテスト向けオプションも利用できます。結果には最適組み合わせ、各ティッカーの個別損益、ポートフォリオ全体の累積リターン/損益が含まれます。

### ブローカー連携によるライブ注文

`trade` サブコマンドはブローカーAPIを呼び出して注文を送信し、状態更新を監視します。`--broker` には `module:factory` または `module.factory` 形式でファクトリーを指定してください。

```bash
stock-predictor trade \
  --broker trading.client:build_paper_client \
  --ticker AAPL \
  --side buy \
  --quantity 5 \
  --config ops/trading.toml
```

設定ファイルの詳細は [`docs/operations.md`](docs/operations.md) を参照してください。異常な状態 (`rejected`, `failed` など) が検出されると通知を発行し非0終了コードで終了します。

### 共通オプション

- `--horizon`: 予測ホライゾン(日数)
- `--lags`: 終値ラグ(複数指定可)。未指定時は `(1, 2, 3, 5, 10)`
- `--cv-splits`: クロスバリデーション分割数
- `--ridge`: リッジ回帰の正則化係数

## 開発

テストは `pytest` で実行できます。

```bash
pytest
```
