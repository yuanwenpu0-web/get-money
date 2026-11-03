# 模型研究：数据基础

这是独立于现有策略、交易和模型的只读研究入口。先设置项目已有的 `DATABASE_URL`，再执行：

```powershell
.\.venv\Scripts\python.exe scripts/check_data.py
.\.venv\Scripts\python.exe scripts/check_data.py --symbol BTCUSDT --output output/btc_quality.json
```

默认同时检查 BTCUSDT 和 ETHUSDT 的全部本地 1m 数据，报告写入 `output/data_quality_report.json`。命令以 0 表示通过、1 表示未通过；数据库读取失败也会在报告中标明错误类型。报告的时间均为 UTC 毫秒。未收盘 K 线单独计数，不进入已收盘数据的质量统计；缺失分钟按数据范围内的分钟间隔计算，范围以外的数据不计入。有效比例只衡量已收盘记录的行级有效性，并不代表数据完整或具备全部字段。

`app.ml.research_data.aggregate_candles` 支持 1、5、15、30、60 分钟，只接受每分钟恰好一条、已经结束的完整 UTC 时间桶。调用方应先检查 `validate_candles` 的报告；报告未通过时不得训练。聚合函数没有前填或未来填充，也不写入数据库。

`klines` 和行情采集现已支持 `taker_buy_volume` 和 `taker_buy_quote_volume`。旧行中为空表示尚未回填，不能用零或估计值冒充真实主动买入量；已运行的旧版服务需要重启后才会持续保存这两个字段。报告只检查已有范围，不代表历史覆盖达到一年，也不证明模型具备预测价值。

批量历史回补由 `scripts/backfill_all_history.py` 执行：只处理项目 `MAJOR_SYMBOLS` 中币安列出的 USDT 永续合约，从最近三年或上市时间开始，按月读取币安公开文件并验证 SHA-256，缺失时尝试日文件和公开 K 线 API。默认三个下载线程，`--workers` 可设为 1～4，数据库仍由单线程写入；全体线程共享限速，API 和公开文件下载分别至少间隔 2 秒和 0.4 秒。HTTP 429 时所有线程共同退避并尊重 `Retry-After`，HTTP 403 或 418 时停止任务。限速无法控制同 IP 的其他应用，应结合其他请求量调整。它不会覆盖已有 K 线价格；如原记录与来源不一致，该条记录不会填入主动买入量，相关月份标记为 `partial`。进度与结果记录在 `output/all_history_backfill.sqlite`，本次后台任务的日志写入 `output/major_history_backfill_repaired.log`。重新运行脚本可继续未完成的月份；首尾月份会重新核查以覆盖范围变化。PostgreSQL 数据目录可用空间低于 40 GiB 时任务停止。若有 `partial` 或 `error` 记录，应检查原因并重新运行；不能将任务启动视为数据已经齐全。

币安源数据中的不合逻辑 K 线会被隔离，记录在进度数据库的 `invalid_times`，对应月份标记 `partial`，其余有效数据继续回补。修复本地冲突记录时，`scripts/reconcile_candles.py` 要求币安历史文件和 REST API 均确认，并在 `output/candle_reconciliation/` 保存修改前记录。对于 2023-11-30 的 XRP、DOT、SUI 三根源数据错误，`scripts/repair_exchange_volume.py` 校验成交文件 SHA-256 和成交 ID 连续性，仅在 OHLC、成交笔数、成交额及主动买入量均一致时，根据逐笔成交重建总成交量；修复依据与原始值记录在 `output/candle_corrections.json`，后续回补只对完全匹配的源记录应用修复。

主流币离线训练使用 `scripts/train_major_models.py`，默认依次处理项目主流币池，逐币训练 15 分钟方向、收益率和实现波动率三个 LightGBM 模型。使用历史特征、时间顺序 70%/15%/15% 切分、预测窗口隔离，方向阈值只拟合训练集；验证集按时间分成早停、校准拟合、校准选择和融合选择四段，各段间隔离预测窗口，最后在测试集评估并与多数类和简单动量基线比较。数据检查未通过时该币种停止训练。输出位于 `data/model_research/<运行时间>/<币种>/`，包含质量报告、模型、特征重要性、数据快照指纹、配置及评估报告，`run_status.json` 记录各币种状态；均为候选研究模型，不参与交易。

本次运行固定截止到 2026-11-02 13:42 UTC（不包含该分钟），以排除那根尚待核查的 BTC K 线和后续未完整回补的记录；各币种使用相同截止时点。校正后的实验版本位于 `data/model_research/verified_20261103/`，进度可查看 `output/verified_lightgbm_training_resume.log` 和 `output/verified_transformer_training_resume.log`。此前 `20261102T153731288932Z` 为初步实验，缺少数据指纹且校准分界有标签重叠，不应作为最终验证结果。可用相同数据与模型参数和 `--resume` 继续未完成的 LightGBM，线程数允许调整；中断的 Transformer 会保留尝试目录，下次重新开始该币种，已完成结果不覆盖。若同时启动 LightGBM 与等待它的 Transformer 批处理，两条命令需要指定同一个新 `--session-id`，以区分旧任务失败和本次失败。

15m 的 CPU Transformer 研究流程为 `scripts/train_transformer_research.py`；需要先安装 `requirements-research-cpu.txt`。本轮使用 120 根序列、每 120 分钟抽样、两轮训练。序列限定在各自训练、验证和测试区间内，输入标准化只拟合训练集，模型融合权重只用验证集选择。两个模型比较前检查数据快照指纹，输入变化时拒绝比较。LightGBM、Transformer 和融合模型在相同的样本外时点比较；融合权重通过验证集选择，且只有融合结果优于两种单模型的样本外 Log Loss 时才允许预测脚本使用，且融合预测时点不能早于最终测试标签可用的时间。候选模型也不会自动投入实盘。

各币种实验文件位于校正后版本的 `<币种>/transformer/`，包括模型、训练曲线、评估和简化回测报告；`comparison_summary.md` 汇总 16 个币种的结果。回测使用下一根 K 线开盘价、固定 15 分钟持有、双边手续费与滑点，回撤只在平仓时计算；按 120 分钟抽样，未包含资金费率、保证金、强平和止损，只用于离线研究，不能当作完整合约回测。

`python scripts/predict_research.py --symbol BTCUSDT --model transformer` 输出已核查历史时点的研究预测；`--model lightgbm` 支持已训练的主流币。命令校验训练快照指纹，不允许预测所需模型选择标签尚不可用的时点，也不连接交易执行。Optuna 调参、完整资金费率回测、滚动验证和模拟盘仍未实现。

`scripts/compare_model_orders.py` 提供逐分钟的独立账户下单模拟，使用项目的风险预算和下一根开盘撮合。2026-11-03 的 16 币种结果位于 `data/model_research/verified_20261103/order_backtest_20261103T004153Z/`，`comparison.md` 为汇总，`*_trades.csv`、`*_equity_daily.csv` 和 `*_funding.json` 分别为成交明细、每日权益和实际使用的资金费快照。相同 10,000 USDT 初始权益按币种和模型分别计算，不是共享资金组合。

本次规则预先固定：分钟收盘后判断，三分类中上涨或下跌概率至少 0.45 且为最大类时产生方向；BTC/ETH 的二分类模型另按 0.5 或其验证集确定的阈值产生信号。使用 1 倍杠杆、8% 基础仓位、0.5% 风险预算，按信号 K 线收盘价设置 0.5% 止损和 1% 止盈，最长持仓 15 分钟、平仓冷却 5 分钟。双边逐次收取 0.04% 手续费和 0.02% 滑点。资金费按 8 小时结算、优先使用标记价格；本地重复窗口向币安公开接口核实，不改数据库。止盈止损同一分钟触发时先按止损处理。

回测不包含盘口深度、排队、订单延迟、实际强平和交易所数量步长等，不能视为真实成交或完整合约仿真。二分类模型以 15 分钟抽样训练，逐分钟推断存在分布差异。融合仅作基于验证集权重的测试期假设性比较，不应利用同一测试期结果择优并宣称独立验证。不同币种账户平均收益并非资金组合收益。
