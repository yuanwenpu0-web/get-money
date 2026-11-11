from __future__ import annotations

import json
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..backtest.runner import HAS_VBT
from ..config import DEFAULT_PUBLIC_INTERVAL_MS, MARKET_SYNC_WORKERS, SUPPORTED_INTERVALS
from ..db import get_db
from ..features.indicators import HAS_PTA, HAS_TALIB
from ..market.data_service import MarketDataService
from ..market.futures import BinanceFuturesClient
from ..models import Credential, JobLog, StrategyConfig, utcnow
from ..schemas import CapitalAdjust, CredentialsPayload, SettingsUpdate
from ..security import decrypt, encrypt, mask
from ..state import all_state, set_state
from ..strategy.rules import known_codes, resolve_params
from ..trading import exchange_gateway as gw
from ..trading import exchange_truth
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
    "min_quote_volume_usd", "symbol_blacklist", "order_type",
    "schedule_enabled", "schedule_seconds", "train_horizon", "exchange_truth_enabled",
    # 巡检并发：改完下一轮巡检即生效，无需重启服务
    "market_sync_workers", "market_public_interval_ms", "signal_workers", "execution_workers",
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
        "market_sync_workers": (1, 32),
        "market_public_interval_ms": (20, 2000),
        "signal_workers": (1, 16),
        "execution_workers": (1, 16),
    }
    # 线程数/间隔必须是整数：ranges 用 float() 校验会放过 2.5 这类值
    for key in ("market_sync_workers", "market_public_interval_ms",
                "signal_workers", "execution_workers"):
        if key in payload and (isinstance(payload[key], bool) or not isinstance(payload[key], int)):
            raise HTTPException(400, f"{key} 必须为整数")
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
    # 并发配置的默认值只在这里补齐（不写库），前端首次打开也能拿到完整表单值
    state.setdefault("market_sync_workers", MARKET_SYNC_WORKERS)
    state.setdefault("market_public_interval_ms", DEFAULT_PUBLIC_INTERVAL_MS)
    state.setdefault("signal_workers", 1)
    state.setdefault("execution_workers", 1)
    for key in ("risk_halt", "risk_halt_reason", "max_drawdown_halt_pct", "consecutive_loss_halt"):
        state.pop(key, None)
    service = MarketDataService(db)
    stats = service.stats() if include_stats else []
    return {
        **state,
        "credentials": _credentials_state(db),
        "scheduler": scheduler.status(),
        "data_stats": stats,
        "data_stats_count": len(stats) if include_stats else service.stats_count(),
        "features": {"talib": HAS_TALIB, "pandas_ta": HAS_PTA, "vectorbt": HAS_VBT},
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
        # 校验目标是「已登记的策略」：默认代码 + 库里真实存在的策略配置行
        registered = known_codes()
        for code in registered:
            resolve_params(code, merged)
        for strategy in db.query(StrategyConfig).all():
            if strategy.code in registered:
                resolve_params(strategy.code, merged, json.loads(strategy.params or '{}'))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    previous = all_state(db)
    target_mode = str(data.get("execution_mode", previous.get("execution_mode") or "paper"))
    target_testnet = bool(data.get("testnet", previous.get("testnet", True)))
    # 正式环境（真实资金）防呆：显式确认 + 该环境的凭证必须通过连通性验证
    if target_mode == "live" and not target_testnet:
        if not payload.confirm_mainnet:
            raise HTTPException(400, "切换到币安正式环境涉及真实资金，请显式确认 confirm_mainnet=true")
        if previous.get("credential_verified_env") != "real":
            raise HTTPException(400, "请先在「测试连接」中验证正式环境 API 凭证，再切换主网")

    for key, value in data.items():
        set_state(db, key, value)
    if "schedule_enabled" in data or "schedule_seconds" in data:
        update_schedule(
            db,
            enabled=bool(data.get("schedule_enabled", all_state(db)["schedule_enabled"])),
            seconds=int(data.get("schedule_seconds", all_state(db)["schedule_seconds"])),
        )

    entering_live = target_mode == "live" and previous.get("execution_mode") != "live"
    env_changed = "testnet" in data and bool(data["testnet"]) != bool(previous.get("testnet", True))
    switch_notes = []
    if entering_live:
        archived = exchange_truth.archive_paper_positions(db)
        switch_notes.append(f"归档本地模拟持仓 {archived} 条")
    if entering_live or env_changed:
        # 环境/模式变化后按交易所当前钱包余额重设收益基准，避免 demo 与正式资金串账
        try:
            result = exchange_truth.rebase(db, force=True)
            switch_notes.append(f"收益基准重置 {result}")
        except Exception as exc:  # noqa: BLE001 - 网络失败不阻塞配置保存
            switch_notes.append(f"收益基准重置失败（可稍后手动重试）: {exc}")
        db.add(JobLog(job="exchange_sync", level="info",
                      message=(f"交易环境切换 → {gw.env_label(db)}；" + "；".join(switch_notes))[:4000]))

    db.add(JobLog(job="settings", level="info",
                  message=(f"更新配置: {', '.join(data.keys())}"
                           + (f"；{'；'.join(switch_notes)}" if switch_notes else ""))[:4000]))
    db.commit()
    return read_settings(db)


@router.post("/capital")
def capital_adjust(payload: CapitalAdjust, db: Session = Depends(get_db)):
    if payload.mode not in ("add", "withdraw", "set"):
        raise HTTPException(400, "mode 仅支持 add / withdraw / set")
    if all_state(db).get("execution_mode") == "live":
        raise HTTPException(400, "实盘模式下资金以币安账户为准，请在交易所出入金后用「重置收益基准」同步")
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
    env = gw.DEMO if testnet else gw.REAL
    # 代理必须与实盘下单链路一致，否则「测试通过但实盘失败」或反之
    client = BinanceFuturesClient(key, secret, testnet=testnet,
                                  proxy=state.get("proxy_url") or None)
    try:
        server_time = client.server_time()
        balances: list[dict] = []
        account_view: dict = {}
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
            account_view = gw.fetch_account(client)
            set_state(db, "credential_verified_env", env)
        return {"ok": True, "server_time": server_time, "signed_ok": signed_ok,
                "env": env, "env_label": gw.env_label(db), "account": account_view,
                "balances": balances}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


@router.get("/exchange")
def exchange_status(db: Session = Depends(get_db)):
    """当前交易环境 + 币安账户快照 + 本地账本对账差异。"""
    state = all_state(db)
    acc = exchange_truth.cached(db, state)
    return {
        "env": gw.env_of(db),
        "env_label": gw.env_label(db),
        "execution_mode": state.get("execution_mode"),
        "truth_enabled": bool(state.get("exchange_truth_enabled", True)),
        "baseline_env": state.get("exchange_baseline_env", ""),
        "initial_capital": state.get("initial_capital"),
        "credential_verified_env": state.get("credential_verified_env", ""),
        "account": acc,
        "market_source": "futures-mainnet",
    }


@router.post("/exchange/rebase")
def exchange_rebase(db: Session = Depends(get_db)):
    """手动按币安当前钱包余额重置收益基准（出入金或切换环境后使用）。"""
    try:
        result = exchange_truth.rebase(db, force=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"读取币安账户失败: {exc}") from exc
    exchange_truth.sync_account(db, force=True)
    db.add(JobLog(job="exchange_sync", level="info", message=f"手动重置收益基准: {result}"[:4000]))
    db.commit()
    return result
