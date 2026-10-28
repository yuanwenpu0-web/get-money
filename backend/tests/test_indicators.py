import json

from fastapi.testclient import TestClient

from app.api import indicators
from app.main import app


def test_catalog_selects_indicators_and_returns_original_source(tmp_path, monkeypatch):
    standard = tmp_path / "pine-collection"
    (standard / "indicators" / "trend").mkdir(parents=True)
    source = '// @version=6\nindicator("示例")\nplot(close)\n'
    (standard / "indicators" / "trend" / "sample.pine").write_text(source, encoding="utf-8")
    (standard / "LICENSE").write_text("Copyright and MIT license", encoding="utf-8")
    (standard / "index.json").write_text(json.dumps([
        {"path": "indicators/trend/sample.pine", "kind": "indicator", "title": "Sample", "description_zh": "描述"},
        {"path": "../outside.pine", "kind": "indicator"},
    ]), encoding="utf-8")
    (tmp_path / "outside.pine").write_text('indicator("outside")', encoding="utf-8")
    crypto = tmp_path / "pine-crypto-collection"
    (crypto / "project").mkdir(parents=True)
    (crypto / "project" / "old.js").write_text('// strategy("comment")\nstudy("old")', encoding="utf-8")
    (crypto / "project" / "strategy.pine").write_text('strategy("strategy")', encoding="utf-8")
    (crypto / "index.json").write_text(json.dumps([
        {"local_path": "project/old.js", "collection": "project", "title_zh": "旧指标", "license": "来源声明"},
        {"local_path": "project/strategy.pine"},
    ], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(indicators, "DATA_DIR", tmp_path)
    client = TestClient(app)
    response = client.get("/api/indicators")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    item = next(item for item in data["items"] if item["title"] == "Sample")
    assert item["category"] == "trend"
    assert item["version"] == "6"
    assert "source" not in item
    detail = client.get(f"/api/indicators/{item['id']}").json()
    assert detail["source"] == source
    assert detail["license"] == "MIT"
    assert detail["license_text"] == "Copyright and MIT license"
    assert client.get("/api/indicators/nonexistent").status_code == 404
