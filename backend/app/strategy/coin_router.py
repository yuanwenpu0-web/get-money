"""币级路由：按 symbol → (strategy, params) 映射，让每个币/类型只跑最适合的策略。

设计原则：
1. 最小侵入 — 路由器只做「查表返回 (strategy, params)」，不改策略本身逻辑；
   调用方（runner / engine）拿到路由结果后照常走 evaluate_signals / simulate。
2. 两级路由 — 先查 symbol 精确路由（覆盖 WLD/SOL 等 holdout 验证过的），
   再查 coin_type 族群路由（L2→strong_pullback、DeFi→mtf_resonance 等），
   都没有则返回 None（调用方自行 fallback 到默认策略 / 手动指定策略）。
3. 数据驱动 — 路由表可由 walk-forward A/B 脚本自动生成，也可手工维护；
   路由器本身不做任何训练/搜索，只做查表。
4. 反过拟合 — 路由表必须来自「训练期正 edge 且 holdout ≥1 个窗口也正」的币，
   否则宁可不路由（让调用方 fallback），也不要拟合噪声。

用法：
    from app.strategy.coin_router import get_route, build_default_routes

    # 查表
    route = get_route("WLDUSDT")  # -> {"strategy": "trend_follow", "params": {...}}
    route = get_route("DOODUSDT") # -> None（所有策略都差，不拉路由）

    # 从 walk-forward 结果自动生成路由表
    build_default_routes("docs/strong-pullback-exit-ab-extended.json")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ============================================================
# 1. 币种类型注册表（维护族群知识）
# ============================================================
# 原则：只有在 walk-forward 中某类型 × 策略组合稳定优于其他，
# 才把该类型纳入族群路由。否则类型注册但 default_route 留空。

COIN_TYPES: dict[str, str] = {
    # --- L1 基础公链（BTC/ETH/SOL/XRP 等）---
    "BTCUSDT": "L1", "ETHUSDT": "L1", "BNBUSDT": "L1", "SOLUSDT": "L1",
    "XRPUSDT": "L1", "ADAUSDT": "L1", "AVAXUSDT": "L1", "DOTUSDT": "L1",
    "LINKUSDT": "L1", "LTCUSDT": "L1", "BCHUSDT": "L1", "XLMUSDT": "L1",
    "DOGEUSDT": "Meme",  # 独立，不和 L1 混
    # --- L2 ---
    "ARBUSDT": "L2", "OPUSDT": "L2", "BASEUSDT": "L2", "ERAUSDT": "L2",
    "STRKUSDT": "L2", "TIAUSDT": "L2",
    # --- DeFi ---
    "UNIUSDT": "DeFi", "AAVEUSDT": "DeFi", "INJUSDT": "DeFi", "HUMAUSDT": "DeFi",
    "JCTUSDT": "DeFi", "WLDUSDT": "DeFi", "ENAUSDT": "DeFi", "QNTUSDT": "DeFi",
    "FILUSDT": "DeFi",
    # --- Meme / Alt ---
    "TRUMPUSDT": "Meme", "BULLAUSDT": "Meme", "DOODUSDT": "Meme",
    "QUSDT": "Alt", "CAPUSDT": "Alt", "PUMPUSDT": "Alt", "BANKUSDT": "Alt",
    "CHIPUSDT": "Alt", "IRYSUSDT": "Alt", "AINUSDT": "Alt", "BRUSDT": "Alt",
    "DOSUSDT": "Alt", "LITUSDT": "Alt", "NILUSDT": "Alt", "ORDERUSDT": "Alt",
    "KATUSDT": "Alt", "PENGUUSDT": "Meme", "AGTUSDT": "Alt", "ASTERUSDT": "Alt",
    "APRUSDT": "Alt", "ARIAUSDT": "Alt", "ZROUSDT": "Alt", "TAGUSDT": "Alt",
    "ONDOUSDT": "Alt", "BEATUSDT": "Alt", "SOONUSDT": "Meme",
    # --- Privacy / Commodity ---
    "XMRUSDT": "Privacy", "DASHUSDT": "Privacy",
    "PAXGUSDT": "Commodity", "XAUTUSDT": "Commodity",
    # --- AI ---
    "TAOUSDT": "AI", "SUIUSDT": "L1", "APTUSDT": "L1", "NEARUSDT": "L1",
}

# ============================================================
# 2. 族群路由规则（类型 → strategy）
# ============================================================
# 来源：2026-10-27 walk-forward 8×60d（56 币 × 近一年）族群效应。
# 纪律：该类型下 ≥70% 币种策略均值 > 0 **且** 该策略均值显著高于其他。
# 8×60d 结论：**没有任何类型的单一策略能稳定优于其他策略** —
# - Alt→trend_follow 只帮 BR/NIL 2 币，毁 AIN/APR/ASTER/LIT 16 币 → 族群分化太大
# - DeFi→mtf_resonance 只帮 ARIA/QNT/JCT/BEAT 4 币，毁 AAVE/FIL/INJ/UNI 4 币 → 族群分化太大
# - L1/Meme 全部已是 strong_pullback（无路由增益）
# 结论：**族群路由全部移除**，只保留 holdout 验证过的精确 symbol 路由 + 黑名单。

TYPE_ROUTES: dict[str, str] = {
    # 空 — 无家族类型满足"≥70% 币策略均值 > 0 且显著优于其他"
}

# ============================================================
# 3. 精确 symbol 路由（holdout 验证通过的）
# ============================================================
# 来源：2026-10-27 walk-forward 8×60d，训练 7 窗 → 选策略 → 第 8 窗 OOS。
# 纪律：训练均值 > 0 **且** holdout ret > 0.001 **且** positions ≥ 3。
# 训练均值 ≤ 0 但 holdout 正的属于"单次幸运"，不纳入（如 JCT/TAO/AIN/INJ/ORDER/AAVE）。

SYMBOL_ROUTES: dict[str, dict] = {
    # --- 训练均值 > 0 **且** holdout > 0（确认真实 edge）---
    "ARIAUSDT": {
        "strategy": "mtf_resonance",
        "evidence": "8wf 56coin: train=+0.92%, holdout=+0.61% (pos≥3); Alt 族首选",
    },
    "QNTUSDT": {
        "strategy": "mtf_resonance",
        "evidence": "8wf 56coin: train=+0.72%, holdout=+1.53% (pos≥3); DeFi",
    },
    "JCTUSDT": {
        "strategy": "mtf_resonance",
        "evidence": "8wf 56coin: train=+0.05%, holdout=+4.37% (pos≥3); DeFi",
    },
    "BRUSDT": {
        "strategy": "trend_follow",
        "evidence": "8wf 56coin: train=+0.40%, holdout=+1.49% (pos≥3); Alt",
    },
    "SOLUSDT": {
        "strategy": "strong_pullback",
        "evidence": "8wf 56coin: train=+0.55%, holdout=+1.49% (pos≥3); L1",
    },
    # 纪律：ENA/HUMA 8×60d holdout 分别 −0.20% / −0.19% → 从精确路由移除
    # 训练正但 holdout 负 = 训练集过拟合，不路由实盘资金
}

# ============================================================
# 4. 拉黑列表（8×60d walk-forward：最优策略 mean ≤ -0.1% 且 8/8 窗口无正收益）
# ============================================================
BLACKLIST: set[str] = {
    "PAXGUSDT",   # Commodity, strong_pullback 0/8 窗口全负 mean=-0.19%
    "XAUTUSDT",   # Commodity, strong_pullback 0/8 窗口全负 mean=-0.19%
    "BANKUSDT",   # Alt, strong_pullback 0/8 窗口全负 mean=-0.27%
    "KATUSDT",    # Alt, strong_pullback 0/8 窗口全负 mean=-0.35%
    "IRYSUSDT",   # Alt, strong_pullback 0/8 窗口全负 mean=-0.50%
    # NEARUSDT 最优 mtf_resonance mean=+0.040%（略正）→ 移出黑名单，让 fallback 处理
    # DOODUSDT 最优 strong_pullback mean=-0.401% 但 holdout +1.65% → 不拉黑（holdout 有 edge）
}


# ============================================================
# 5. 策略级币种宇宙（"策略绑定到哪些币"）
# ============================================================
# 与上文的「路由」（每个币选最优策略）正交：这里限定**某个策略只在哪些币上运行**。
# 用途：吸收反转的判定依赖 volume 与影线结构，中小币种的量价噪声会让 c1（|delta| 进入滚动
# 95 分位）与 c5（单边支配度）失真，因此把它的运行范围钉在高流动性主流币上。
#
# 主流币 = 长期成交额居前、01:00 级分钟数据连续、且在 U 本位永续上深度充足的白名单。
# 说明：本白名单**只约束运行范围，不是路由决策**——在 walk-forward 拿出 holdout 为正的
# 证据之前，SYMBOL_ROUTES 里不会写入 absorption_reversal（路由纪律见第 3 节）。

MAJOR_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "SUIUSDT",
    "TRXUSDT", "NEARUSDT", "APTUSDT", "BCHUSDT",
)

STRATEGY_UNIVERSE: dict[str, frozenset[str]] = {
    "absorption_reversal": frozenset(MAJOR_SYMBOLS),
}


def symbol_allowed(code: str, symbol: str) -> bool:
    """策略 → 币种的运行范围检查：未登记宇宙的策略不限制（返回 True）。"""
    universe = STRATEGY_UNIVERSE.get(code)
    return universe is None or symbol.upper() in universe


def get_route(symbol: str) -> Optional[dict]:
    """查表返回 (strategy, params) 或 None。

    路由优先级：
    1. 黑名单 → 直接返回 None（所有策略都差）
    2. 精确 symbol 路由（holdout 验证过的）→ 返回
    3. 族群 type 路由 → 返回
    4. 都没有 → None（调用方 fallback）
    """
    symbol = symbol.upper()
    if symbol in BLACKLIST:
        return None
    # 1. 精确路由
    if symbol in SYMBOL_ROUTES:
        route = SYMBOL_ROUTES[symbol]
        return {"strategy": route["strategy"], "params": route.get("params", {})}
    # 2. 族群路由
    coin_type = COIN_TYPES.get(symbol)
    if coin_type and coin_type in TYPE_ROUTES:
        return {"strategy": TYPE_ROUTES[coin_type], "params": {}}
    return None


def get_coin_type(symbol: str) -> str | None:
    """返回币种类型标签，或 None（未知）。"""
    return COIN_TYPES.get(symbol.upper())

def all_routes() -> dict[str, dict]:
    """返回当前所有 symbol→route 的完整映射（给 UI / 运维用）。"""
    out = {}
    for sym in set(list(SYMBOL_ROUTES.keys()) + list(COIN_TYPES.keys())):
        if sym in BLACKLIST:
            out[sym] = None
        elif sym in SYMBOL_ROUTES:
            r = SYMBOL_ROUTES[sym]
            out[sym] = {"strategy": r["strategy"], "params": r.get("params", {})}
        else:
            ct = COIN_TYPES.get(sym)
            if ct and ct in TYPE_ROUTES:
                out[sym] = {"strategy": TYPE_ROUTES[ct], "params": {}}
            else:
                out[sym] = None
    return out


def save_routes(path: str | Path) -> None:
    """把当前路由表导出为 JSON（审计 / 版本控制）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coin_types": COIN_TYPES,
        "type_routes": TYPE_ROUTES,
        "symbol_routes": SYMBOL_ROUTES,
        "blacklist": sorted(BLACKLIST),
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_routes(path: str | Path) -> None:
    """从 JSON 加载路由表（运行时覆盖硬编码）。"""
    global COIN_TYPES, TYPE_ROUTES, SYMBOL_ROUTES, BLACKLIST
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    COIN_TYPES = payload.get("coin_types", COIN_TYPES)
    TYPE_ROUTES = payload.get("type_routes", TYPE_ROUTES)
    SYMBOL_ROUTES = payload.get("symbol_routes", SYMBOL_ROUTES)
    BLACKLIST = set(payload.get("blacklist", BLACKLIST))
