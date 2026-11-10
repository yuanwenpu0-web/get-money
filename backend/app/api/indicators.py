"""Catalog of the third-party Pine indicator collections.

The local indexes and source files are the catalog data. Strategies are
intentionally excluded; Pine source is never executed by this application.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import DATA_DIR
from ..catalog_indicators import list_indicators as list_python_indicators
from ..indicator_classifications import CRYPTO_CLASSIFICATIONS
from ..pagination import DEFAULT_PAGE_SIZE, PageQuery, PageSizeQuery, offset_of, page_metadata

router = APIRouter(prefix="/indicators", tags=["indicators"])
_COLLECTIONS = ("pine-collection", "pine-crypto-collection")
_CATALOG_CACHE: list[dict] | None = None
_CATALOG_SIGNATURE: tuple | None = None
_DECLARATION = re.compile(r"(?m)^\s*(indicator|study|strategy)\s*\(")
_VERSION = re.compile(r"(?m)^\s*//\s*@version\s*=\s*(\d+)")

# The first collection explicitly groups scripts by function. Other collections
# group by upstream author, so do not infer an indicator's purpose from that.
_TYPES = {
    "trend_direction": ("趋势方向", "根据价格、均线或趋势状态识别方向", "趋势识别与方向过滤"),
    "trend_strength": ("趋势强度", "衡量趋势强弱或持续状态", "过滤弱趋势及震荡"),
    "momentum": ("动量", "衡量价格变化速度或动量摆动", "动量变化及反转观察"),
    "mean_reversion": ("均值回归", "衡量价格或振荡值偏离及回归", "震荡与极端偏离观察"),
    "volatility": ("波动率", "衡量价格波动幅度或通道宽度", "波动状态与风险观察"),
    "money_flow": ("资金流与成交量", "结合成交量与价格估计交易活动或资金流", "量价确认与资金流观察"),
    "relative_strength": ("相对强弱", "比较价格或资产的相对表现", "强弱比较与筛选"),
    "market_structure": ("市场结构", "识别价格摆动、区间或结构形态", "支撑阻力及结构变化观察"),
    "composite": ("综合状态", "组合多个条件或特征得到状态", "多条件过滤与市场状态观察"),
}


def _safe_file(root: Path, relative: str) -> Path | None:
    """Resolve only files within their collection, including across symlinks."""
    if not isinstance(relative, str) or not relative:
        return None
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


def _declaration(source: str) -> str | None:
    # Old indexes omit kind. Only a top-level Pine declaration identifies a kind.
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    code = re.sub(r"//[^\n]*", "", code)
    match = _DECLARATION.search(code)
    return match.group(1) if match else None


def _license_text(root: Path, relative: str) -> str | None:
    # Crypto collection licenses belong to each upstream project, not its parent.
    project = Path(relative).parts[0] if relative else ""
    license_dir = root / project if root.name == "pine-crypto-collection" else root
    if not license_dir.resolve().is_relative_to(root.resolve()):
        return None
    candidates = sorted(p for p in license_dir.glob("LICENSE*") if p.is_file())
    for candidate in candidates:
        safe = _safe_file(root, candidate.relative_to(root).as_posix())
        if safe:
            return safe.read_text(encoding="utf-8-sig", errors="replace")
    return None


def _catalog(data_dir: Path) -> list[dict]:
    """解析两个 Pine 集合的索引，并读出每个源码文件（版本号、哈希、python 绑定）。

    这里是全站唯一「每次请求都读文件」的热点：156 个源码文件逐份读盘 + 算 sha256，
    单次约 0.6s。目录是随仓库提交的只读数据，用索引文件的 mtime+size 做签名缓存，
    命中时 0ms；目录被替换（重新拉取集合）后签名变化会自动重建。
    """
    global _CATALOG_CACHE, _CATALOG_SIGNATURE
    signature = _catalog_signature(data_dir)
    if _CATALOG_CACHE is not None and signature == _CATALOG_SIGNATURE:
        return _CATALOG_CACHE
    items = _build_catalog(data_dir)
    _CATALOG_CACHE, _CATALOG_SIGNATURE = items, signature
    return items


def _catalog_signature(data_dir: Path) -> tuple:
    """目录签名：索引内容 + 每个源码文件的 (mtime, size)。

    只 stat 不读内容，156 个文件的 stat 约十几毫秒；比每次读源码 + 算 sha256（约 0.6s）
    便宜两个数量级，同时能捕捉「索引换了」和「源码被改了」两种变化。
    """
    parts: list[tuple] = [("root", str(data_dir.resolve()))]
    for collection in _COLLECTIONS:
        root = data_dir / collection
        index = root / "index.json"
        try:
            entries = json.loads(index.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            parts.append((collection, None))
            continue
        parts.append((collection, len(entries) if isinstance(entries, list) else 0))
        for entry in entries if isinstance(entries, list) else ():
            relative = entry.get("local_path") or entry.get("path") or ""
            try:
                stat = (root / relative).stat()
                parts.append((relative, stat.st_mtime_ns, stat.st_size))
            except OSError:
                parts.append((relative, None, None))
    return tuple(parts)


def _build_catalog(data_dir: Path) -> list[dict]:
    results = []
    bindings = {(item.collection, item.path, item.source_sha256): item.name
                for item in list_python_indicators()}
    for collection in _COLLECTIONS:
        root = data_dir / collection
        index = root / "index.json"
        if not index.is_file():
            continue
        entries = json.loads(index.read_text(encoding="utf-8-sig"))
        for entry in entries:
            relative = entry.get("local_path") or entry.get("path")
            path = _safe_file(root, relative)
            if path is None:
                continue
            source = path.read_text(encoding="utf-8-sig", errors="replace")
            kind = entry.get("kind")
            if kind != "indicator" and not (not kind and _declaration(source) in ("indicator", "study")):
                continue
            version = _VERSION.search(source)
            parts = Path(relative).parts
            category = parts[1] if collection == "pine-collection" and len(parts) > 1 else entry.get("collection", "")
            classification = (_TYPES.get(category) if collection == "pine-collection"
                              else CRYPTO_CLASSIFICATIONS.get(Path(relative).as_posix()))
            results.append({
                "id": hashlib.sha256(f"{collection}/{relative}".encode("utf-8")).hexdigest(),
                "python_method": bindings.get((collection, Path(relative).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())),
                "title": entry.get("title_zh") or entry.get("title") or path.stem,
                "description": entry.get("description_zh") or entry.get("original_description") or "",
                "category": category,
                "indicator_type": classification[0] if classification else "待分类",
                "calculation_summary": classification[1] if classification else "请查看指标描述与源码确认计算逻辑",
                "use_case": classification[2] if classification else "需结合指标具体定义确认",
                "collection": collection,
                "source_url": entry.get("source_url") or "",
                "original_title": entry.get("title") or "",
                "original_description": entry.get("original_description") or "",
                "notes": entry.get("notes") or entry.get("implementation_note_zh") or "",
                "source_commit": entry.get("source_commit") or "",
                "sha256": entry.get("sha256") or "",
                "license": entry.get("license") or ("MIT" if collection == "pine-collection" else "未注明"),
                "version": version.group(1) if version else None,
                "source": source,
                "license_text": _license_text(root, relative),
            })
    return sorted(results, key=lambda item: (item["title"].casefold(), item["id"]))


@router.get("")
def list_indicators(
    q: str = "",
    collection: str = "",
    category: str = "",
    indicator_type: str = "",
    page: PageQuery = 1,
    page_size: PageSizeQuery = DEFAULT_PAGE_SIZE,
):
    """指标目录分页：筛选与切片都在服务端，响应体从 180KB 降到几 KB。

    `facets` 回传全量去重后的来源/分类/类型，前端下拉不再依赖「已拉到全部行」。
    """
    items = _catalog(DATA_DIR)
    facets = {
        "collections": sorted({item["collection"] for item in items if item["collection"]}),
        "categories": sorted({item["category"] for item in items if item["category"]}),
        "indicator_types": sorted({item["indicator_type"] for item in items if item["indicator_type"]}),
    }
    query = q.strip().casefold()
    filtered = [
        item for item in items
        if (not collection or item["collection"] == collection)
        and (not category or item["category"] == category)
        and (not indicator_type or item["indicator_type"] == indicator_type)
        and (
            not query
            or any(
                query in str(item.get(key) or "").casefold()
                for key in ("title", "original_title", "description", "category", "collection")
            )
        )
    ]
    page_rows = filtered[offset_of(page, page_size):offset_of(page, page_size) + page_size]
    return {
        "items": [{k: v for k, v in item.items() if k not in ("source", "license_text")} for item in page_rows],
        "facets": facets,
        "catalog_total": len(items),
        **page_metadata(len(filtered), page, page_size),
    }


@router.get("/research/latest")
def indicator_research():
    """Return the checked-in experiment, not a live or universal ranking."""
    report = Path(__file__).resolve().parents[3] / "docs" / "indicator-evaluation-current.json"
    if not report.is_file():
        raise HTTPException(status_code=404, detail="暂无指标评估报告")
    data = json.loads(report.read_text(encoding="utf-8"))
    return {key: data[key] for key in ("config", "dataset", "grid", "summary", "singles_all", "combos_all", "limitations")}


@router.get("/research/altcoin-50")
def altcoin_50_research():
    """Return the 50-altcoin experiment with its original data-quality record."""
    report = Path(__file__).resolve().parents[3] / "docs" / "altcoin-50-independent.json"
    if not report.is_file():
        raise HTTPException(status_code=404, detail="暂无50山寨币评估报告")
    return json.loads(report.read_text(encoding="utf-8"))


@router.get("/{indicator_id}")
def get_indicator(indicator_id: str):
    for item in _catalog(DATA_DIR):
        if item["id"] == indicator_id:
            return item
    raise HTTPException(status_code=404, detail="指标不存在")
