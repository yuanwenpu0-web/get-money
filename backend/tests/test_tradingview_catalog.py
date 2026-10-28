import hashlib
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import tradingview_strategies as catalog


def test_catalog_discovers_sources_without_index_and_hides_non_code_matches(tmp_path, monkeypatch):
    (tmp_path / "project").mkdir()
    source = b'//@version=5\r\nstrategy("SMA Example", overlay=true)\r\nstrategy.entry("L", strategy.long)\r\n'
    (tmp_path / "project" / "example.pine").write_bytes(source)
    (tmp_path / "project" / "Code").write_text('strategy(\n "No Extension")\n', encoding="utf8")
    (tmp_path / "project" / "indicator.pine").write_text(
        '// strategy("comment")\n/*\nstrategy("block")\n*/\nindicator("strategy name")\n', encoding="utf8")
    (tmp_path / "project" / "string.pine").write_text('s = "strategy(hello)"\nindicator("test")', encoding="utf8")
    (tmp_path / "project" / "bad.pine").write_bytes(b"\xff\x00")
    monkeypatch.setattr(catalog, "DATA_DIR", tmp_path)
    result = catalog.list_tradingview_strategies()
    assert result["total"] == 2
    row = next(item for item in result["strategies"] if item["title"] == "SMA Example")
    assert row["path"] == "data/project/example.pine"
    assert row["version"] == "5"
    assert row["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert row["execution_status"] == "not_migrated"
    assert "source" not in row
    assert catalog.get_tradingview_strategy(row["id"])["source"] == source.decode()
    assert any(item["title"] == "No Extension" for item in result["strategies"])
    with pytest.raises(HTTPException) as exc:
        catalog.get_tradingview_strategy("invalid")
    assert exc.value.status_code == 404


def test_catalog_reports_only_research_support_for_existing_ports(tmp_path, monkeypatch):
    relative = "pine-crypto-collection/eterna/strategies/trend-following/sma-crossover/strategy.pine"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('strategy("SMA")', encoding="utf8")
    digest = hashlib.sha256(b'strategy("SMA")').hexdigest()
    monkeypatch.setitem(catalog._PORTS, relative, ("sma_crossover", digest))
    item = catalog._catalog(tmp_path)[0]
    assert item["execution_status"] == "entry_filter_and_research"
    assert item["local_filter_code"] == "sma_crossover"
    path.write_text('strategy("Changed")', encoding="utf8")
    assert catalog._catalog(tmp_path)[0]["execution_status"] == "not_migrated"


def test_catalog_does_not_expose_symlink_outside_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.pine"
    outside.write_text('strategy("outside")', encoding="utf8")
    try:
        (root / "escape.pine").symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    assert catalog._catalog(root) == []


def test_catalog_rejects_paths_resolving_outside_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    link = root / "escape.pine"
    link.write_text('strategy("inside")', encoding="utf8")
    outside = tmp_path / "outside.pine"
    outside.write_text('strategy("outside")', encoding="utf8")
    resolve = Path.resolve

    def simulate_link(path, *args, **kwargs):
        if path == link:
            return resolve(outside, *args, **kwargs)
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", simulate_link)
    assert catalog._catalog(root) == []
