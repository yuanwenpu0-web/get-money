from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
import fit_readiness as readiness


def ms(year, month, day):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def test_global_history_shorter_than_calendar_requirement_blocks_validation():
    result = readiness.history_gate(ms(2025, 9, 20), ms(2026, 11, 13))
    assert result["status"] == "BLOCKED"
    assert result["required_first_ms"] == ms(2025, 8, 13)


def test_global_history_alone_cannot_pass_per_symbol_quality():
    result = readiness.history_gate(ms(2025, 1, 1), ms(2026, 11, 13))
    assert result["status"] == "NEEDS_SYMBOL_QUALITY_CHECK"


def test_pool_counterexamples_preserve_raw_labels_without_promoting_them():
    cells = [{"key": "a|x", "scope": "major", "interval": "1h",
              "pooled_mean_net_pct": -0.12, "verdict_cell": "candidate"},
             {"key": "b|x", "scope": "liq_q1", "interval": "30m",
              "pooled_mean_net_pct": 0.3, "verdict_cell": "candidate"}]
    result = readiness.pool_counterexamples(cells)
    assert len(result) == 1
    assert result[0]["key"] == "a|x"
    assert result[0]["raw_label"] == "candidate"
    assert result[0]["audit_status"] == "EXPLORATORY_ONLY"
    assert cells[0]["verdict_cell"] == "candidate"
