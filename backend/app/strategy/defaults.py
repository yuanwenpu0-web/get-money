import json

from sqlalchemy.orm import Session

from ..models import StrategyConfig

DEFAULT_STRATEGIES = [
    {
        "code": "ml_ensemble",
        "name": "ML 趋势共振",
        "description": "LightGBM/XGBoost 方向概率为主，叠加 SuperTrend、EMA、MACD 顺势确认",
        "params": {"ml_weight": 0.6, "min_score": 0.55, "require_model": False},
    },
    {
        "code": "trend_follow",
        "name": "SuperTrend 趋势跟随",
        "description": "SuperTrend 方向 + EMA 排列 + MACD 动量，适合单边行情",
        "params": {"min_score": 0.55, "adx_filter": 18},
    },
    {
        "code": "mean_reversion",
        "name": "RSI 均值回归",
        "description": "RSI 超买超卖反向入场，适合震荡行情",
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
