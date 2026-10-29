"""币种级参数覆盖：把「每个币种跑自己的最优参数」落成一张可审计的表。

为什么需要这一层
----------------
`策略 × 池 × 周期` 绑定只能给**一池币**共用一套参数。但同一套「吸收阈值 + 止损
宽度 + 目标 R」在 BTC 和某个日均振幅 15% 的小币上根本不是一件事：小币的 delta
百分位阈值、ATR 相对成本、可用的目标 R 全都不一样。参数寻优一旦按币种跑，就必须
有地方存放逐币结果，否则最强的那部分收益会被池级折中抹平。

层级
----
    全局设置（app_state） < 策略默认（spec.defaults） < 绑定 params < 本表

也就是说覆盖是「在绑定之上打补丁」：绑定换了周期/池，币种覆盖仍然生效（因为键里
带 interval）；删除覆盖即回到绑定参数，不需要改任何别的东西。

约束
----
- 只允许覆盖策略参数键；`fee_rate` / `slippage` 由全局设置或回测请求决定，
  `resolve_params` 会拒绝（撮合成本不能按币种差异化，否则回测读数不可比）。
- 写入前一律走 `resolve_params` 全量校验，非法值当场报错，不留脏数据。
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..models import SymbolParam
from ..state import all_state
from .registry import has_spec
from .rules import resolve_params

MAX_NOTE = 200


def _clean(params: object) -> dict:
    if not isinstance(params, dict):
        raise ValueError("币种参数必须是 JSON 对象")
    return {key: value for key, value in params.items() if value is not None}


def validate_symbol_params(strategy: str, params: object, interval: str) -> dict:
    """校验一组币种覆盖参数；非法键/值/成本覆盖一律抛 ValueError。"""
    if not has_spec(strategy):
        raise ValueError(f"未知策略: {strategy}")
    return resolve_params(strategy, {}, _clean(params), interval=interval)


def load_map(
    db: Session,
    strategy: str | None = None,
    interval: str | None = None,
) -> dict[tuple[str, str, str], dict]:
    """一次性取出覆盖表 → `{(strategy, symbol, interval): params}`。

    巡检/寻优一次要处理几百个币种，逐条查库会把 SQL 往返变成瓶颈，所以入口只有
    这个批量读取；单条读取只是它的语法糖。
    """
    query = db.query(SymbolParam)
    if strategy:
        query = query.filter(SymbolParam.strategy_code == strategy)
    if interval:
        query = query.filter(SymbolParam.interval == interval)
    result: dict[tuple[str, str, str], dict] = {}
    for row in query.all():
        try:
            params = json.loads(row.params or "{}")
        except ValueError:
            continue
        if isinstance(params, dict) and params:
            result[(row.strategy_code, row.symbol, row.interval)] = params
    return result


def load_symbol_map(db: Session, strategy: str, interval: str) -> dict[str, dict]:
    """单个「策略 × 周期」下的 `{symbol: params}`，供绑定解析直接查。"""
    return {
        symbol: params
        for (code, symbol, iv), params in load_map(db, strategy, interval).items()
        if code == strategy and iv == interval
    }


def get_override(db: Session, strategy: str, symbol: str, interval: str) -> dict:
    row = (
        db.query(SymbolParam)
        .filter(
            SymbolParam.strategy_code == strategy,
            SymbolParam.symbol == symbol,
            SymbolParam.interval == interval,
        )
        .first()
    )
    if row is None:
        return {}
    try:
        params = json.loads(row.params or "{}")
    except ValueError:
        return {}
    return params if isinstance(params, dict) else {}


def merged_params(
    db: Session,
    strategy: str,
    symbol: str,
    interval: str,
    *,
    overrides: dict | None = None,
    state: dict | None = None,
) -> dict:
    """`(策略, 币种, 周期)` 的最终生效参数（四层合并完毕）。

    `overrides` 传绑定参数层，缺省只叠策略配置——两种情况都会再叠币种覆盖。
    """
    layer: dict = {}
    if overrides:
        layer.update(overrides)
    layer.update(get_override(db, strategy, symbol, interval))
    return resolve_params(
        strategy,
        state if state is not None else all_state(db),
        layer or None,
        interval=interval,
    )


def set_overrides(
    db: Session,
    strategy: str,
    interval: str,
    items: dict[str, dict],
    *,
    source: str = "manual",
    run_id: int = 0,
    note: str = "",
    base_params: dict | None = None,
) -> int:
    """批量写入/更新覆盖，返回写入条数。

    先全量校验再落库：一条坏数据不能把整批应用卡在半途。

    `base_params` 传上线实际生效的那层参数（绑定 + 全局），校验按「该层 + 覆盖」
    合并后的值判断。只拿覆盖层单独校验会漏掉跨字段约束——例如覆盖只写了
    `max_hold_bars=20`，它本身合法，但和全局 `min_hold_bars=30` 一起就非法，
    等到巡检时才炸等于把问题推给了运行时。
    """
    if not items:
        return 0
    if not has_spec(strategy):
        raise ValueError(f"未知策略: {strategy}")
    state = all_state(db)
    cleaned: dict[str, dict] = {}
    for symbol, params in items.items():
        code = str(symbol or "").strip().upper()
        if not code:
            raise ValueError("币种代码不能为空")
        payload = _clean(params)
        if not payload:
            raise ValueError(f"{code}: 参数为空，删除覆盖请用 DELETE")
        if {"fee_rate", "slippage"} & set(payload):
            raise ValueError(
                f"{code}: 手续费和滑点请在全局设置配置，不能按币种覆盖撮合成本"
            )
        # 校验底座 = 线上实际生效的那层（base）+ 本次覆盖；成本键只存在于 base，
        # resolve_params 不接受把它当覆盖传进来
        layer = {
            key: value
            for key, value in {**(base_params or {}), **payload}.items()
            if key not in ("fee_rate", "slippage")
        }
        try:
            resolve_params(strategy, state, layer, interval=interval)
        except ValueError as exc:
            raise ValueError(f"{code}: {exc}") from exc
        cleaned[code] = payload

    existing = {
        row.symbol: row
        for row in db.query(SymbolParam)
        .filter(
            SymbolParam.strategy_code == strategy,
            SymbolParam.interval == interval,
            SymbolParam.symbol.in_(list(cleaned)),
        )
        .all()
    }
    for symbol, payload in cleaned.items():
        row = existing.get(symbol)
        if row is None:
            db.add(
                SymbolParam(
                    strategy_code=strategy,
                    symbol=symbol,
                    interval=interval,
                    params=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    source=source,
                    run_id=int(run_id or 0),
                    note=note[:MAX_NOTE],
                )
            )
        else:
            row.params = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            row.source = source
            row.run_id = int(run_id or 0)
            row.note = note[:MAX_NOTE]
    db.commit()
    return len(cleaned)


def delete_overrides(
    db: Session, strategy: str, interval: str, symbols: list[str]
) -> int:
    """删除指定币种的覆盖（回到绑定/策略默认）。"""
    codes = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not codes:
        return 0
    removed = (
        db.query(SymbolParam)
        .filter(
            SymbolParam.strategy_code == strategy,
            SymbolParam.interval == interval,
            SymbolParam.symbol.in_(codes),
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return int(removed or 0)


def describe(
    db: Session,
    strategy: str | None = None,
    interval: str | None = None,
    *,
    diff_only: bool = False,
) -> dict:
    """覆盖表概览：总量、按来源分布、明细（默认只给「与绑定不同」的键）。"""
    query = db.query(SymbolParam)
    if strategy:
        query = query.filter(SymbolParam.strategy_code == strategy)
    if interval:
        query = query.filter(SymbolParam.interval == interval)
    rows = query.order_by(SymbolParam.symbol.asc()).all()
    items = []
    for row in rows:
        try:
            params = json.loads(row.params or "{}")
        except ValueError:
            params = {}
        items.append(
            {
                "id": row.id,
                "strategy": row.strategy_code,
                "symbol": row.symbol,
                "interval": row.interval,
                "params": params if isinstance(params, dict) else {},
                "source": row.source,
                "run_id": int(row.run_id or 0),
                "note": row.note or "",
                "updated_at": row.updated_at.isoformat() if row.updated_at else "",
            }
        )
    sources: dict[str, int] = {}
    for item in items:
        sources[item["source"] or "manual"] = sources.get(item["source"] or "manual", 0) + 1
    return {"total": len(items), "sources": sources, "items": items}
