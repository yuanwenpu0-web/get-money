import json

from sqlalchemy.orm import Session

from ..models import StrategyConfig

DEFAULT_STRATEGIES = [
    {
        "code": "ml_ensemble",
        "name": "ML 趋势共振",
        "description": "模型方向优势 + 趋势一致性确认，过滤低波动成本损耗，按单笔风险限制仓位",
        "params": {"ml_weight": 0.6, "min_score": 0.55, "require_model": False},
    },
    {
        "code": "trend_follow",
        "name": "SuperTrend 趋势跟随",
        "description": "ADX 趋势过滤 + SuperTrend 与快慢 EMA 同向确认，减少震荡追单",
        "params": {"min_score": 0.55, "adx_filter": 18},
    },
    {
        "code": "mean_reversion",
        "name": "RSI 均值回归",
        "description": "震荡环境中的 RSI 极值与布林带确认，强趋势时暂停逆势入场",
        "params": {"rsi_low": 30, "rsi_high": 70, "min_score": 0.55},
    },
    {
        "code": "funding_extreme",
        "name": "资金费率极值",
        "description": "资金费率 z-score 极值逆向入场：负费率（空头付费、拥挤做空）开多，正费率开空，仅在低 ADX 震荡市执行",
        "params": {"z_threshold": 2.0, "max_adx": 25.0, "min_score": 0.55, "funding_filter": False},
    },
    {
        "code": "mtf_resonance",
        "name": "多周期趋势共振",
        "description": "4h/1d 更高周期趋势给 1h 信号做上下文确认，仅大/中/小周期同向且更高周期 ADX 足够时才开仓，降低震荡市假信号（迭代11实验策略）",
        "params": {"adx_filter": 22.0, "htf_adx_filter": 20.0, "min_score": 0.55},
    },
    {
        "code": "strong_pullback",
        "name": "Martin Luk 强势回调（本地适配）",
        "description": (
            "4h 强势趋势（ema21>ema50 且 ADX 达标）下，等 15m 首次回调至 EMA9/锚定 VWAP，"
            "出现收复 EMA9 的转强确认才做多；只做多 1 倍杠杆，单笔风险 0.5%、名义仓位上限 35%；"
            "结构止损、移动止损、固定目标和持仓时限控制出场；EMA9 软出场可配置"
        ),
        "params": {
            "htf_adx_filter": 22.0,
            "pullback_min_bars": 8,
            "pullback_confirm_bars": 8,
            "target_r": 10.0,
            "partial_tp_r": 0.0,  # 2.0 默认经 A/B 验证无 edge，回退 fixed；见 strong-pullback-strategy.md
            "partial_tp_frac": 0.5,
            "max_hold_hours": 48.0,
            "max_leverage": 1,
            "max_position_pct": 0.35,
            "risk_per_trade_pct": 0.005,
        },
    },
]


def seed_strategies(db: Session) -> None:
    existing = {row.code for row in db.query(StrategyConfig.code).all()}
    for item in DEFAULT_STRATEGIES:
        if item["code"] not in existing:
            db.add(
                StrategyConfig(
                    code=item["code"],
                    name=item["name"],
                    description=item["description"],
                    enabled=1,
                    params=json.dumps(item["params"], ensure_ascii=False),
                )
            )
    db.commit()
