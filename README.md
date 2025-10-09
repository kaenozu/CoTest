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

### 共通オプション

- `--horizon`: 予測ホライゾン(日数)
- `--lags`: 終値ラグ(複数指定可)。未指定時は `(1, 2, 3, 5, 10)`
- `--cv-splits`: クロスバリデーション分割数
- `--ridge`: リッジ回帰の正則化係数

### バックテスト時のコストモデル

`backtest` サブコマンドでは以下のオプションを組み合わせて、段階的な手数料や方向別スリッページ、ティックサイズなどを表現できます。

- `--fee-rate`: 取引金額に比例する基本手数料率。
- `--fee-tier`: `閾値:料率` 形式で段階的手数料を複数指定。(例: `--fee-tier 100000:0.0005`)
- `--fixed-fee`: 各トレード(往復)ごとに加算する固定手数料。
- `--slippage`: 方向に依らない基本スリッページ率。
- `--slippage-long` / `--slippage-short`: ロング・ショート個別のスリッページ。未指定側は `--slippage` の値を利用。
- `--liquidity-slippage`: 取引数量と出来高の比率に応じて追加されるスリッページ係数。
- `--tick-size`: 約定価格を丸める最小刻み。

```bash
stock-predictor backtest --ticker AAPL --threshold 0.002 \
  --fee-rate 0.001 --fee-tier 100000:0.0005 --fixed-fee 20 \
  --slippage 0.0005 --slippage-short 0.001 --liquidity-slippage 0.05 \
  --tick-size 0.05
```

## 開発

テストは `pytest` で実行できます。

```bash
pytest
```
