"""Read-only inventory of Pine strategy sources; no executable claims implied."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from ..config import DATA_DIR
from ..db import get_db
from ..backtest.tradingview_research import run_research

router = APIRouter(prefix="/tradingview-strategies", tags=["tradingview-strategies"])


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=2, max_length=24, pattern=r"^[A-Za-z0-9]+$")
    interval: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = "15m"
    days: int = Field(default=30, ge=1, le=180)
_DECL = re.compile(r"(?m)^[ \t]*strategy\s*\(")
_VERSION = re.compile(r"(?m)^\s*//\s*@version\s*=\s*(\d+)")
_TITLE = re.compile(r"\s*([\"'])(.*?)\1", re.DOTALL)
_PORTS = {
    "pine-crypto-collection/eterna/strategies/trend-following/sma-crossover/strategy.pine": (
        "sma_crossover", "fa092377267200b3aded1e23c4fb060d351c7d51c342305eef98b32f9cd0c4b9"),
    "pine-crypto-collection/eterna/strategies/trend-following/donchian-channel/strategy.pine": (
        "donchian_channel", "b81ff3d19e651305c1a89ded653e5b7332e3c782e2a4de849d3bd03944ff69e0"),
}
_RESEARCH_PORTS = {
    "pine-crypto-collection/eterna/strategies/trend-following/ichimoku-cloud/strategy.pine":
        ("ichimoku_cloud", "eff28c0b19ab2014677ff7239ab53024de6cd3d7567c8009061ef39505a068e6"),
    "pine-crypto-collection/eterna/strategies/trend-following/chandelier-exit/strategy.pine":
        ("chandelier_exit", "d686d4dfcbd730a4c18dab6552598995503d03e5b18cb37f6a1702b643ee4174"),
    "pine-crypto-collection/eterna/strategies/momentum/stoch_rsi_crossover.pine":
        ("stoch_rsi_crossover", "71f000b054e504ca8428ab6805ffd0f501df5dfed3d10e17bc553e75fe13ce8e"),
    "pine-crypto-collection/dheeraj_eth/BB_EMA_ETHUSDT.pine":
        ("bb_ema_eth", "d5684ddd5d425e52308ccac95824dd9094e42af71dce65283f8e7dfeb8464014"),
    "pine-crypto-collection/eterna/strategies/trend-following/ema-ribbon/strategy.pine":
        ("ema_ribbon", "f4220dbbe009bf84c3a3bcc1ad8c382995434a9908e8edf184efc4effcbefd33"),
    "pine-crypto-collection/eterna/strategies/trend-following/keltner-channel/strategy.pine":
        ("keltner_channel", "8cb171c062837a554a1730cff11bf9e64a958046692bf5b471d41aa14f87b9bc"),
    "pine-crypto-collection/eterna/strategies/trend-following/linear-regression-channel/strategy.pine":
        ("linear_regression_channel", "fdc77b6f32696fc373ecc979c366ac49f43025da165968d8be483d2e7fa6b18a"),
    "pine-crypto-collection/eterna/strategies/momentum/cci_strategy.pine":
        ("cci_strategy", "1a452670426b06d22f16f023abf8f4c77a59ca7704f775590111c357ff6862e2"),
    "pine-crypto-collection/eterna/strategies/momentum/awesome_oscillator.pine":
        ("awesome_oscillator", "89da2199a00933b7f552765788535914fada294a65368446e5d95a3ffe359fe3"),
    "pine-crypto-collection/eterna/strategies/momentum/williams_r_reversal.pine":
        ("williams_r_reversal", "e40ca5f9c1f4672a58958e10365da971a1aa69d35992ac1a22f97a11dec395b6"),
}


def _mask_comments_and_strings(source: str) -> str:
    """Hide comments and strings while preserving offsets and line boundaries."""
    chars = list(source)
    i, mode = 0, None
    while i < len(source):
        ch = source[i]
        following = source[i:i + 2]
        if mode is None:
            if following in ("//", "/*"):
                mode = "line" if following == "//" else "block"
                chars[i:i + 2] = "  "
                i += 2
                continue
            if ch in ("'", '"'):
                mode = ch
                chars[i] = " "
        elif mode == "line":
            if ch == "\n":
                mode = None
            else:
                chars[i] = " "
        elif mode == "block":
            if following == "*/":
                chars[i:i + 2] = "  "
                mode = None
                i += 2
                continue
            if ch != "\n":
                chars[i] = " "
        else:
            if ch == "\\" and i + 1 < len(source):
                chars[i] = " "
                if source[i + 1] != "\n":
                    chars[i + 1] = " "
                i += 2
                continue
            if ch == mode:
                mode = None
            if ch != "\n":
                chars[i] = " "
        i += 1
    return "".join(chars)


def _catalog(data_dir: Path, *, include_source: bool = False) -> list[dict]:
    root = data_dir.resolve()
    if not root.is_dir():
        return []
    items = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in ("", ".pine", ".pinescript", ".txt", ".js"):
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            raw = resolved.read_bytes()
            source = raw.decode("utf-8-sig")
        except (UnicodeError, OSError):
            continue
        declaration = _DECL.search(_mask_comments_and_strings(source))
        if not declaration:
            continue
        relative = path.relative_to(root).as_posix()
        title = _TITLE.match(source[declaration.end():])
        version = _VERSION.search(source)
        port = _PORTS.get(relative)
        # Line ending changes do not change Pine semantics; any substantive source
        # edit invalidates the port association until it is reviewed again.
        canonical = source.replace("\r\n", "\n").replace("\r", "\n")
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        local_code = port[0] if port and digest == port[1] else None
        research_port = _RESEARCH_PORTS.get(relative)
        research_code = research_port[0] if research_port and digest == research_port[1] else local_code
        item = {
            "id": hashlib.sha256(relative.encode("utf-8")).hexdigest(),
            "title": title.group(2) if title else path.stem,
            "path": "data/" + relative,
            "version": version.group(1) if version else None,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "execution_status": ("entry_filter_and_research" if local_code else
                                 "research_backtest" if research_code else "not_migrated"),
            "local_filter_code": local_code,
            "research_code": research_code,
        }
        if include_source:
            item["source"] = source
        items.append(item)
    return sorted(items, key=lambda item: (item["title"].casefold(), item["path"]))


@router.get("")
def list_tradingview_strategies():
    rows = _catalog(DATA_DIR)
    return {"strategies": rows, "total": len(rows)}


@router.get("/{strategy_id}")
def get_tradingview_strategy(strategy_id: str):
    for item in _catalog(DATA_DIR, include_source=True):
        if item["id"] == strategy_id:
            return item
    raise HTTPException(status_code=404, detail="TradingView 策略不存在")


@router.post("/{strategy_id}/backtest")
def backtest_tradingview_strategy(strategy_id: str, payload: ResearchRequest,
                                  db: Session = Depends(get_db)):
    item = get_tradingview_strategy(strategy_id)
    code = item.get("research_code") or item.get("local_filter_code")
    if not code:
        raise HTTPException(status_code=400, detail="该策略尚未实现本地研究回测")
    try:
        return run_research(db, code, payload.symbol, payload.interval, payload.days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
