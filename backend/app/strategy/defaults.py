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


def seed_strategies(db: Session) -> None:
    existing = {row.code for row in db.query(StrategyConfig.code).all()}
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
