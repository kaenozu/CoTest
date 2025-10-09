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

- `forecast` / `backtest` サブコマンドは、`--ticker` を省略するとアプリが用意した推奨ユニバースから自動でティッカーを選びます。
- `--universe` で利用するユニバースを、`--auto-index` で候補リスト内のインデックス(0始まり)を変更できます。
- `--period` と `--interval` は yfinance のパラメータに準拠します。既定値は `60d` と `1d` です。
- `--adjust` で価格調整方法を切り替えられます。`auto` は yfinance の自動調整、`manual` は取得後に分割・配当を反映、既定値の `none` は未調整値を使用します。

```bash
# 推奨ティッカーを利用して予測
stock-predictor forecast --universe default --auto-index 0

# ティッカーを明示的に指定する場合
stock-predictor forecast --ticker AAPL --period 60d --interval 1d --adjust manual
```

### 複数ティッカーのポートフォリオ最適化

`backtest-portfolio` サブコマンドは以下の2通りで候補ティッカーを準備できます。

1. `--tickers-file` に1行1ティッカー形式のテキストファイルを指定する。
2. `--universe` と `--limit` を指定し、推奨ユニバースから必要数だけ自動取得する。

候補のティッカーすべてについてバックテストを実行し、累積リターンが最大になる組み合わせとランキングを表示します。

```text
AAPL
MSFT
GOOG
AMZN
```

```bash
# ファイル指定
stock-predictor backtest-portfolio --tickers-file tickers.txt --combination-size 2

# 推奨ユニバースから自動取得
stock-predictor backtest-portfolio --universe default --limit 5 --combination-size 3
```

`--lags` や `--ridge` などのバックテスト向けオプションも利用できます。結果には最適組み合わせ、各ティッカーの個別損益、ポートフォリオ全体の累積リターン/損益が含まれます。

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
