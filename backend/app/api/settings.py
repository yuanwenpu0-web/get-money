from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..backtest.runner import HAS_VBT
from ..config import SUPPORTED_INTERVALS
from ..db import get_db
from ..features.indicators import HAS_PTA, HAS_TALIB
from ..market.binance import BinanceClient
from ..market.data_service import MarketDataService
from ..models import Credential, JobLog, ModelRecord, StrategyConfig, utcnow
from ..schemas import CapitalAdjust, CredentialsPayload, SettingsUpdate
from ..security import decrypt, encrypt, mask
from ..state import all_state, set_state
from ..strategy.rules import resolve_params
from ..trading.portfolio import adjust_capital
from ..trading.scheduler import scheduler, update_schedule

router = APIRouter(prefix="/settings", tags=["settings"])

EDITABLE_KEYS = {
    "data_source",
    "symbols",
    "interval",
    "testnet",
    "fee_rate",
    "slippage",
    "max_leverage",
    "max_position_pct",
    "base_position_pct",
    "heavy_score",
    "min_score",
    "min_hold_bars",
    "sl_atr_mult",
    "tp_atr_mult",
    "min_sl_pct",
    "min_tp_pct",
    "backtest_trailing",
    "trailing_atr_mult",
    "max_hold_bars",
    'risk_per_trade_pct',
    'cooldown_bars',
    'trailing_enabled',
    'trail_activation_atr',
    'max_signal_age_bars',
    "schedule_enabled",
    "schedule_seconds",
    "train_horizon",
    "auto_train_days",
}


def _credentials_state(db: Session) -> dict:
    row = db.query(Credential).order_by(Credential.id.desc()).first()
    if row is None:
        return {"configured": False, "api_key": "", "testnet": bool(all_state(db).get("testnet", True))}
    key = decrypt(row.api_key_enc)
    return {
        "configured": bool(key),
        "api_key": mask(key),
        "testnet": bool(all_state(db).get("testnet", True)),
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }


def _validate(payload: dict) -> None:
    if "data_source" in payload and payload["data_source"] not in ("nofx", "binance"):
        raise HTTPException(400, "data_source 仅支持 nofx / binance")
    if "interval" in payload and payload["interval"] not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "K线周期不支持")
    if "symbols" in payload:
        symbols = [str(item).strip().upper() for item in payload["symbols"] if str(item).strip()]
        if not symbols:
            raise HTTPException(400, "至少保留一个交易对")
        payload["symbols"] = symbols
    ranges = {
        "fee_rate": (0, 0.01),
        "slippage": (0, 0.01),
        "max_leverage": (1, 125),
        "max_position_pct": (0.01, 1),
        "base_position_pct": (0.01, 1),
        "heavy_score": (0.5, 1),
        "min_score": (0.5, 1),
        "min_hold_bars": (0, 10000),
        "sl_atr_mult": (0.1, 10),
        "tp_atr_mult": (0.2, 20),
        "min_sl_pct": (0.0001, 0.2),
        "min_tp_pct": (0.0002, 0.5),
        "trailing_atr_mult": (0.1, 10),
        "max_hold_bars": (0, 100000),
        "schedule_seconds": (10, 3600),
        "train_horizon": (1, 60),
        "auto_train_days": (7, 365),
    }
    for key, (low, high) in ranges.items():
        if key in payload and not (low <= float(payload[key]) <= high):
            raise HTTPException(400, f"{key} 超出允许范围 [{low}, {high}]")
    if "base_position_pct" in payload and "max_position_pct" in payload:
        if float(payload["base_position_pct"]) > float(payload["max_position_pct"]):
            raise HTTPException(400, "基础仓位不能大于单仓上限")
    if "min_score" in payload and "heavy_score" in payload:
        if float(payload["min_score"]) > float(payload["heavy_score"]):
            raise HTTPException(400, "重仓评分必须不低于开仓阈值")


@router.get("")
def read_settings(db: Session = Depends(get_db)):
    from ..market import nofx_source

    state = all_state(db)
    stats = MarketDataService(db).stats()
    return {
        **state,
        "credentials": _credentials_state(db),
        "scheduler": scheduler.status(),
        "data_stats": stats,
        "nofx": nofx_source.source_status(),
        "features": {"talib": HAS_TALIB, "pandas_ta": HAS_PTA, "vectorbt": HAS_VBT},
        "model_count": db.query(ModelRecord).filter(ModelRecord.status == "ready").count(),
        "intervals": SUPPORTED_INTERVALS,
    }


@router.put("")
def update_settings(payload: SettingsUpdate, db: Session = Depends(get_db)):
    data = {key: value for key, value in payload.model_dump(exclude_none=True).items() if key in EDITABLE_KEYS}
    _validate(data)
    if 'backtest_trailing' in data and 'trailing_enabled' not in data:
        data['trailing_enabled'] = data['backtest_trailing']
    merged = {**all_state(db), **data}
    _validate({key: value for key, value in merged.items() if key in EDITABLE_KEYS})
    try:
        for code in ('ml_ensemble', 'trend_follow', 'mean_reversion'):
            resolve_params(code, merged)
        for strategy in db.query(StrategyConfig).all():
            resolve_params(strategy.code, merged, json.loads(strategy.params or '{}'))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    for key, value in data.items():
        set_state(db, key, value)
    if "schedule_enabled" in data or "schedule_seconds" in data:
        update_schedule(
            db,
            enabled=bool(data.get("schedule_enabled", all_state(db)["schedule_enabled"])),
            seconds=int(data.get("schedule_seconds", all_state(db)["schedule_seconds"])),
        )
    db.add(JobLog(job="settings", level="info", message=f"更新配置: {', '.join(data.keys())}"))
    db.commit()
    return read_settings(db)


@router.post("/capital")
def capital_adjust(payload: CapitalAdjust, db: Session = Depends(get_db)):
    if payload.mode not in ("add", "withdraw", "set"):
        raise HTTPException(400, "mode 仅支持 add / withdraw / set")
    try:
        snapshot = adjust_capital(db, payload.amount, payload.mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    db.add(JobLog(job="capital", level="info", message=f"{payload.mode} {payload.amount} {payload.note}"))
    db.commit()
    return snapshot


@router.get("/credentials")
def read_credentials(db: Session = Depends(get_db)):
    return _credentials_state(db)


@router.put("/credentials")
def save_credentials(payload: CredentialsPayload, db: Session = Depends(get_db)):
    if not payload.api_key.strip() or not payload.api_secret.strip():
        raise HTTPException(400, "API key 与 secret 不能为空")
    row = db.query(Credential).order_by(Credential.id.desc()).first()
    if row is None:
        row = Credential(api_key_enc="", api_secret_enc="")
        db.add(row)
    row.api_key_enc = encrypt(payload.api_key.strip())
    row.api_secret_enc = encrypt(payload.api_secret.strip())
    row.updated_at = utcnow()
    set_state(db, "testnet", payload.testnet)
    db.commit()
    return _credentials_state(db)


@router.post("/credentials/test")
def test_credentials(db: Session = Depends(get_db)):
    row = db.query(Credential).order_by(Credential.id.desc()).first()
    state = all_state(db)
    key = decrypt(row.api_key_enc) if row else ""
    secret = decrypt(row.api_secret_enc) if row else ""
    client = BinanceClient(key, secret, bool(state.get("testnet", True)))
    try:
        server_time = client.server_time()
        account = client.account() if key and secret else None
        balances = []
        if account:
            balances = [
                {"asset": item["asset"], "free": float(item["free"]), "locked": float(item["locked"])}
                for item in account.get("balances", [])
                if float(item["free"]) > 0 or float(item["locked"]) > 0
            ][:20]
        return {"ok": True, "server_time": server_time, "signed_ok": bool(account), "balances": balances}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}
