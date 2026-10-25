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
