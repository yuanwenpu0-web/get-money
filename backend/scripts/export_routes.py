"""导出默认币级路由表为 JSON。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.strategy.coin_router import save_routes

save_routes(ROOT / "docs" / "coin-route-defaults.json")
print("导出成功:", ROOT / "docs" / "coin-route-defaults.json")
