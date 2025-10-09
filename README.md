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
- `--period` と `--interval` は yfinance のパラメータに準拠します。既定値は `3y` と `1d` です。
- `--start-date` と `--end-date` を指定すると期間を明示できます(未指定側は `--period` と組み合わせ可能)。
- `--column` で `Dividends` や `Stock Splits` など追加のデータ列を取得できます(複数指定可)。
- `--adjust` で価格調整方法を切り替えられます。`auto` は yfinance の自動調整、`manual` は取得後に分割・配当を反映、既定値の `none` は未調整値を使用します。

```bash
stock-predictor forecast --ticker AAPL --period 3y --interval 1d --column Dividends
stock-predictor forecast --ticker AAPL --start-date 2021-01-01 --end-date 2023-01-01 --adjust manual
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
  --period 3y \
  --interval 1d
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
