# Pine 本地策略工作台

将可取得源码的 TradingView Pine v6 策略转换为本地 Python 运行，使用相同 K 线逐项对照，并在 Binance U 本位永续行情上回测。当前版本是**有限语法的研究原型**：解析通过不代表已与 TradingView 一致，也不代表可以实盘交易。

## 当前能力

- 粘贴或导入 Pine 源码，检查支持范围并生成可下载的 Python 模块。生成模块调用本项目的 `app.pine` 本地运行时，不依赖 TradingView Webhook；它不是可脱离本项目的单文件程序。
- 支持严格受限的 Pine v6 收盘策略：简单赋值、算术/布尔比较、历史引用、`input.int/float`、`ta.ema/sma/rsi/atr/crossover/crossunder`，以及条件内的 `strategy.entry/close`。只支持 `strategy("名称")` 默认配置、每个方向一个订单 ID。重绘、多周期、`var`、`strategy.exit`、限价/止损、移动止损、K 线内计算及其他不支持的语法会拒绝处理，不能保证任意社区策略都能转换。
- 导入同源 K 线和人工准备的 TradingView 参考 CSV，按时间比较 OHLCV、四个订单信号和可选指标列，输出差异。导出样本的真实性、参数、预热区间、时间周期及收盘确认状态需要使用者核对。项目目前没有可用于证明一致性的真实 TradingView 导出样本。
- BTCUSDT / ETHUSDT 的 Binance U 本位永续公共行情、本地数据库或 CSV 数据回测。收盘决策，下一根开盘模拟成交，固定基础资产数量，支持反手、平仓、手续费和滑点；拒绝缺口和无效 K 线。并不完整复刻 TradingView Broker Emulator，也不模拟资金费、杠杆、保证金、强平、订单簿或交易所精度约束。
- 可保存并恢复本地模拟账户，追加已收盘 K 线后重放结果，重复 K 线幂等。其成交在后续 K 线收盘后回放，使用历史下一根开盘价，**不是实时可成交价格**。支持显式启动 Binance 公共 1 分钟行情轮询，重启服务后默认停止；空账户从最新已收盘 K 线开始逐步预热，数据缺口无法补齐或请求失败时停止。为避免混用行情，已有未绑定 CSV 历史的账户不能启动轮询。仅单进程写入，必须保证本机时钟准确。不会向交易所下单。
- **正式下单尚未接入新策略链路。**旧版自动交易、模型训练及其 API 不属于新服务入口，不会在启动时自动恢复；保留的历史数据不代表已经通过新链路验证。

## 启动

Windows PowerShell，在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pine.txt
cd frontend
npm ci
npm run build
cd ..
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

打开 `http://127.0.0.1:8010`。前端开发可在 `frontend` 下运行 `npm run dev`，Vite 会代理 `/api` 到 8010。公网行情需要网络可访问 Binance，默认使用主网公开行情，不需要 API 密钥；回测行情客户端可读取旧数据库的 `proxy_url`，模拟轮询使用系统代理环境变量。服务仅应绑定可信本机，不提供用户认证。

若要从本地库读取，数据库配置沿用 `DATABASE_URL`；默认 `data/trading.db`。新界面不提供历史数据同步，空库可选 Binance 公共行情或导入 CSV。行情接口只读取，不发送订单。

## 数据与对照

K 线 CSV 列名为 `open_time,open,high,low,close,volume`。`open_time` 是 UTC Unix 毫秒或秒（界面也接受带时区的 ISO 时间）；每一行应是已完成、连续的 1 分钟 K 线。参考 CSV 需要相同 OHLCV 和 `long_entry,short_entry,long_close,short_close` 四个信号列，值为 `true/false` 或 `1/0`；可附加需要比较的数值指标列。TradingView 图表导出须由用户以辅助绘图等方式提供这些列，并将其与提交转换的源码、参数和相同起始历史对齐；辅助绘图语句本身不属于当前支持的转换子集。比较使用精确 OHLCV 和可配置的指标容差；不能由系统推断行情是否真的同源。

回测会过滤当前未收盘 K 线，最后一根的信号保持待处理，不强行平仓。尽量保留足够的共同预热历史，否则指标初始值可能不同。不同的撮合、手续费、滑点及未建模的资金费可使收益不同。

## API 和测试

交互式 API 文档：`http://127.0.0.1:8010/docs`。

- `POST /api/pine/convert`：检查并生成调用本地运行时的 Python。
- `POST /api/pine/compare`：输入源码、K 线、参考数据和可选指标列，返回逐行差异。
- `POST /api/pine/backtest`：输入源码及 K 线，或指定 `data_source=binance/local`、`symbol` 和 `interval`。
- `POST /api/pine/paper/{id}`、`GET /api/pine/paper/{id}`、`POST /api/pine/paper/{id}/candles`：创建、读取、追加已收盘 K 线。账户文件保存在 `data/pine-paper/`。
- `POST /api/pine/paper/{id}/start`、`POST /api/pine/paper/{id}/stop`、`GET /api/pine/paper/{id}/status`：显式控制和检查 Binance 公共行情轮询。

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_pine_api.py backend/tests/test_pine_converter.py backend/tests/test_pine_backtest.py backend/tests/test_pine_paper.py backend/tests/test_pine_paper_runner.py -q
cd frontend
npm run build
```

历史数据库保留；其中的数据不表示已迁移或经过新链路验证。
