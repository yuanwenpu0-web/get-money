"""Read-only catalog of the third-party Pine indicator collections.

The checked-in indexes and source files are the canonical imported data. Strategies
are intentionally excluded; presence in this catalog does not imply compatibility
with the local Pine compiler.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import DATA_DIR

router = APIRouter(prefix="/indicators", tags=["indicators"])
_COLLECTIONS = ("pine-collection", "pine-crypto-collection")
_DECLARATION = re.compile(r"(?m)^\s*(indicator|study|strategy)\s*\(")
_VERSION = re.compile(r"(?m)^\s*//\s*@version\s*=\s*(\d+)")


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
    results = []
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
            results.append({
                "id": hashlib.sha256(f"{collection}/{relative}".encode("utf-8")).hexdigest(),
                "title": entry.get("title_zh") or entry.get("title") or path.stem,
                "description": entry.get("description_zh") or entry.get("original_description") or "",
                "category": category,
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
def list_indicators():
    items = _catalog(DATA_DIR)
    return {"items": [{k: v for k, v in item.items() if k not in ("source", "license_text")} for item in items],
            "total": len(items)}


@router.get("/{indicator_id}")
def get_indicator(indicator_id: str):
    for item in _catalog(DATA_DIR):
        if item["id"] == indicator_id:
            return item
    raise HTTPException(status_code=404, detail="指标不存在")
