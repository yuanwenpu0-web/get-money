from __future__ import annotations

import json
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..backtest.runner import HAS_VBT
from ..config import SUPPORTED_INTERVALS
from ..db import get_db
from ..features.indicators import HAS_PTA, HAS_TALIB
from ..market.data_service import MarketDataService
from ..market.futures import BinanceFuturesClient
from ..models import Credential, JobLog, ModelRecord, StrategyConfig, utcnow
from ..schemas import CapitalAdjust, CredentialsPayload, SettingsUpdate
from ..security import decrypt, encrypt, mask
from ..state import all_state, set_state
from ..strategy.rules import resolve_params
from ..trading.portfolio import adjust_capital
from ..trading.scheduler import scheduler, update_schedule

router = APIRouter(prefix="/settings", tags=["settings"])

EDITABLE_KEYS = {
    "data_source", "symbols", "interval", "symbol_scope", "execution_mode",
    "proxy_url", "testnet", "fee_rate", "slippage", "max_leverage",
    "max_position_pct", "base_position_pct", "heavy_score", "min_score",
    "min_hold_bars", "sl_atr_mult", "tp_atr_mult", "min_sl_pct", "min_tp_pct",
    "backtest_trailing", "trailing_atr_mult", "max_hold_bars", "risk_per_trade_pct",
    "cooldown_bars", "trailing_enabled", "trail_activation_atr", "max_signal_age_bars",
    "min_quote_volume_usd", "symbol_blacklist", "risk_halt", "risk_halt_reason",
    "max_drawdown_halt_pct", "consecutive_loss_halt", "order_type",
    "schedule_enabled", "schedule_seconds", "train_horizon", "auto_train_days",
    "metrics_assist_enabled",
    # 迭代9：持续训练与晋升门禁
    "auto_train_enabled", "auto_train_interval_hours", "auto_train_top_n", "auto_train_workers",
    "ml_min_auc", "ml_promote_rounds", "ml_auto_enable",
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


PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")


def _validate_proxy(value: str) -> None:
    if not value:
        return  # 留空 = 直连
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in PROXY_SCHEMES or not parsed.hostname:
        raise HTTPException(400, "proxy_url 仅支持 http:// https:// socks5:// socks5h:// 且主机不能为空")
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(400, "proxy_url 端口不合法") from exc
    if port is None or not (1 <= port <= 65535):
        raise HTTPException(400, "proxy_url 端口必须在 1-65535")


def _normalize_blacklist(value) -> list[str]:
    """黑名单归一：必须为字符串数组，元素去空白、统一大写、去重（保序）。"""
    if not isinstance(value, list):
        raise HTTPException(400, "symbol_blacklist 必须为字符串数组")
    seen: set[str] = set()
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise HTTPException(400, "symbol_blacklist 元素必须为非空字符串")
        symbol = item.strip().upper()
        if symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    return normalized


def _validate(payload: dict, db: Session | None = None) -> None:
    if "data_source" in payload and payload["data_source"] != "binance":
        raise HTTPException(400, "data_source 仅支持 binance")
    if "execution_mode" in payload and payload["execution_mode"] not in ("paper", "live"):
        raise HTTPException(400, "execution_mode 仅支持 paper / live")
    if "proxy_url" in payload:
        _validate_proxy(str(payload["proxy_url"]))
    if payload.get("execution_mode") == "live" and db is not None:
        row = db.query(Credential).order_by(Credential.id.desc()).first()
        if row is None or not decrypt(row.api_key_enc):
            raise HTTPException(400, "切换 live 前请先配置币安 API 凭证")
    if "interval" in payload and payload["interval"] not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "K线周期不支持")
    if "order_type" in payload and payload["order_type"] not in ("market", "limit_maker"):
        raise HTTPException(400, "order_type 仅支持 market / limit_maker")
    if "risk_halt" in payload and not isinstance(payload["risk_halt"], bool):
        raise HTTPException(400, "risk_halt 必须为布尔值")
    if "metrics_assist_enabled" in payload and not isinstance(payload["metrics_assist_enabled"], bool):
        raise HTTPException(400, "metrics_assist_enabled 必须为布尔值")
    for key in ("auto_train_enabled", "ml_auto_enable"):
        if key in payload and not isinstance(payload[key], bool):
            raise HTTPException(400, f"{key} 必须为布尔值")
    for key, (low, high) in (("auto_train_interval_hours", (1, 720)),
                             ("auto_train_top_n", (1, 50)),
                             ("auto_train_workers", (1, 8)),
                             ("ml_promote_rounds", (1, 10))):
        if key in payload:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise HTTPException(400, f"{key} 必须为整数")
            if not low <= value <= high:
                raise HTTPException(400, f"{key} 超出允许范围 [{low}, {high}]")
    if "min_quote_volume_usd" in payload:
        volume = payload["min_quote_volume_usd"]
        if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
            raise HTTPException(400, "min_quote_volume_usd 必须为不小于 0 的整数（0 = 不过滤）")
    if "symbol_blacklist" in payload:
        payload["symbol_blacklist"] = _normalize_blacklist(payload["symbol_blacklist"])
    if "symbols" in payload and payload.get("symbol_scope") != "all":
        symbols = [str(item).strip().upper() for item in payload["symbols"] if str(item).strip()]
        if not symbols:
            raise HTTPException(400, "至少保留一个交易对")
        payload["symbols"] = symbols
    if payload.get("symbol_scope") == "custom" and "symbols" in payload:
        if not [str(item).strip() for item in payload["symbols"] if str(item).strip()]:
            raise HTTPException(400, "symbol_scope=custom 时 symbols 不能为空")
    if "symbol_scope" in payload and payload["symbol_scope"] not in ("all", "custom"):
        raise HTTPException(400, "symbol_scope 仅支持 all / custom")
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
        "max_drawdown_halt_pct": (0, 1),
        "consecutive_loss_halt": (0, 1000),
        "ml_min_auc": (0.5, 0.9),
    }
    for key, (low, high) in ranges.items():
        if key in payload:
            try:
                value = float(payload[key])
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"{key} 必须为数值") from exc
            if not (low <= value <= high):
                raise HTTPException(400, f"{key} 超出允许范围 [{low}, {high}]")
    if "base_position_pct" in payload and "max_position_pct" in payload:
        if float(payload["base_position_pct"]) > float(payload["max_position_pct"]):
            raise HTTPException(400, "基础仓位不能大于单仓上限")
    if "min_score" in payload and "heavy_score" in payload:
        if float(payload["min_score"]) > float(payload["heavy_score"]):
            raise HTTPException(400, "重仓评分必须不低于开仓阈值")


@router.get("")
def read_settings(db: Session = Depends(get_db), include_stats: bool = True):
    state = all_state(db)
    service = MarketDataService(db)
    stats = service.stats() if include_stats else []
    return {
        **state,
        "credentials": _credentials_state(db),
        "scheduler": scheduler.status(),
        "data_stats": stats,
        "data_stats_count": len(stats) if include_stats else service.stats_count(),
        "features": {"talib": HAS_TALIB, "pandas_ta": HAS_PTA, "vectorbt": HAS_VBT},
        "model_count": db.query(ModelRecord).filter(ModelRecord.status == "ready").count(),
        "intervals": SUPPORTED_INTERVALS,
    }


@router.put("")
def update_settings(payload: SettingsUpdate, db: Session = Depends(get_db)):
    data = {key: value for key, value in payload.model_dump(exclude_none=True).items() if key in EDITABLE_KEYS}
    _validate(data, db)
    if 'backtest_trailing' in data and 'trailing_enabled' not in data:
        data['trailing_enabled'] = data['backtest_trailing']
    merged = {**all_state(db), **data}
    _validate({key: value for key, value in merged.items() if key in EDITABLE_KEYS}, db)
    try:
        for code in ('ml_ensemble', 'trend_follow', 'mean_reversion', 'mtf_resonance', 'strong_pullback', 'absorption_reversal'):
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
    testnet = bool(state.get("testnet", True))
    client = BinanceFuturesClient(key, secret, testnet=testnet)
    try:
        server_time = client.server_time()
        balances: list[dict] = []
        signed_ok = False
        if key and secret:
            signed_ok = True
            if testnet:
                # 合约测试网: /fapi/v2/balance 返回 asset/balance/availableBalance
                balances = [
                    {
                        "asset": item.get("asset"),
                        "free": float(item.get("availableBalance", 0) or 0),
                        "locked": float(item.get("balance", 0) or 0) - float(item.get("availableBalance", 0) or 0),
                    }
                    for item in client.balance()
                    if float(item.get("balance", 0) or 0) > 0
                ][:20]
            else:
                # 合约主网: /fapi/v2/account 的 assets 列表
                account = client.account()
                balances = [
                    {
                        "asset": item.get("asset"),
                        "free": float(item.get("availableBalance", 0) or 0),
                        "locked": float(item.get("walletBalance", 0) or 0) - float(item.get("availableBalance", 0) or 0),
                    }
                    for item in account.get("assets", [])
                    if float(item.get("walletBalance", 0) or 0) > 0
                ][:20]
        return {"ok": True, "server_time": server_time, "signed_ok": signed_ok, "balances": balances}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}
