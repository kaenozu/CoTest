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

```bash
stock-predictor forecast --ticker AAPL --period 60d --interval 1d
```

- `--ticker` を指定するとCSVパスは指定できません。
- `--period` と `--interval` は yfinance のパラメータに準拠します。既定値は `60d` と `1d` です。

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
