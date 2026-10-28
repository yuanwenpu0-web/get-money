import json

from sqlalchemy.orm import Session

from ..models import StrategyConfig

DEFAULT_STRATEGIES = [
    {
        "code": "absorption_reversal",
        "name": "吸收反转（Absorption Strat 本地适配）",
        "description": (
            "放量单边主动成交（|delta| 位于滚动百分位之上）却推不出价格、在摆动极值留下拒绝影线 → "
            "进攻方被困：买方被吸收做空、卖方被吸收做多。默认等待后续 3 根内收盘穿越吸收 bar 实体确认，"
            "期间打穿吸收极值或超时作废。结构止损（极值 ∓ 0.25 ATR）、2R 固定目标、1R 保本、30 根时间止损，"
            "名义仓位 30%（脚本 percent_of_equity=30）、1 倍杠杆，多空双向"
        ),
        "params": {
            "delta_window": 500,
            "delta_pct": 95.0,
            "body_mult": 0.25,
            "extreme_lookback": 20,
            "wick_mult": 0.20,
            "dominance_pct": 20.0,
            "abs_gap_bars": 10,
            "legacy_gap_logic": False,
            "conf_mode": 1,
            "conf_window_bars": 3,
            "direction_mode": 0,
            "trend_mode": 0,
            "trend_len": 200,
            "session_start_hour": 0.0,
            "session_end_hour": 24.0,
            "sl_buffer_atr": 0.25,
            "target_r": 2.0,
            "be_trigger_r": 1.0,
            "min_sl_atr": 0.30,
            "max_sl_atr": 3.00,
            "max_hold_bars": 30,
            "min_hold_bars": 0,
            "max_leverage": 1,
            "base_position_pct": 0.30,
            "max_position_pct": 0.30,
            "risk_per_trade_pct": 0.02,
            "cooldown_bars": 0,
            "trailing_enabled": False,
        },
        # 默认**不启用**：docs/absorption-reversal-strategy.md 三窗 × 主流币 15m 检验
        # 池化 88/106 笔的每笔期望 95% CI 跨 0（不可判定），且放宽灵敏度后转负。
        # 移植已完整落地、可手动开启观察，但在补齐 1m 历史、做出样本足够的判据之前，
        # 不参与自动巡检的资金竞争。
        "enabled": 0,
    },
]


ALLOWED_CODES = tuple(item["code"] for item in DEFAULT_STRATEGIES)


def seed_strategies(db: Session) -> None:
    """按 DEFAULT_STRATEGIES 补齐缺失行，并把已下线的策略强制停用。

    老库里可能残留其它策略的 enabled=1 记录；那些代码本仓库已删除，
    若继续参与巡检会在参数解析阶段直接抛错，所以这里统一置 0。
    """
    rows = db.query(StrategyConfig).all()
    existing = {row.code: row for row in rows}
    for row in rows:
        if row.code not in ALLOWED_CODES and int(row.enabled or 0) != 0:
            row.enabled = 0
    for item in DEFAULT_STRATEGIES:
        if item["code"] not in existing:
            db.add(
                StrategyConfig(
                    code=item["code"],
                    name=item["name"],
                    description=item["description"],
                    enabled=item.get("enabled", 1),
                    params=json.dumps(item["params"], ensure_ascii=False),
                )
            )
    db.commit()
