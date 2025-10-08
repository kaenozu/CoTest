from pathlib import Path


def test_spec_mentions_backtest_signals():
    spec_path = Path("SPEC.md")
    assert spec_path.exists(), "SPEC.md が存在しません"

    content = spec_path.read_text(encoding="utf-8")

    assert "backtest" in content.lower(), "SPECにbacktestコマンドの記述がありません"
    assert "buy" in content.lower(), "SPECに売買シグナル(Buy)の説明が不足しています"
    assert "sell" in content.lower(), "SPECに売買シグナル(Sell)の説明が不足しています"
    assert "hold" in content.lower(), "SPECに売買シグナル(Hold)の説明が不足しています"
