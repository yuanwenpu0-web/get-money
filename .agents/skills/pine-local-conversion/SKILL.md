---
name: pine-local-conversion
description: 将 TradingView Pine 策略转换为本项目的本地 Python 策略，并使用 TradingView 导出数据核对信号、运行回测或本地模拟交易。适用于用户提供策略源码或要求排查转换差异时。
---

# Pine 策略本地转换

以用户提供的完整 Pine 源码为输入；没有源码时不要猜测闭源策略的实现。先检查当前项目实现和用户提供的数据，再决定哪些工作能完成。源码、交易所、交易品种、周期、时区、K 线来源及策略参数应在结果中写清楚。

## 转换

- 使用 `backend/app/pine/converter.py` 的 `compile_pine(source)` 和 `program.to_python()`。生成的 Python 依赖项目内的 `app.pine` 运行时；不要称其为无依赖的独立脚本。
- 转换器只支持明确限定的 Pine v6 子集。解析通过只说明源码通过当前语法检查；某些语义和数据问题在 `program.evaluate(candles)` 时才会被发现。遇到不支持或报错时报告具体语句和原因，不能默默删除逻辑、修改策略语义或声称转换成功。
- 尤其检查指标预热与 `na`、历史引用、重绘、多周期数据、订单 ID、入场和出场顺序、手续费与成交时点。不同数据或缺少可观测信号时，不能推断 TradingView 和 Python 信号一致。
- 对超出当前子集的需求，先保留原始源码和错误信息。按用户要求扩展转换器时补充针对语义的测试；不受支持的部分保持明确标记。

## 数据对照

- 使用同交易所、交易品种、周期和收盘确认的同源 K 线，确保时间范围与预热区间一致。TradingView 导出须包含 OHLCV 和要核对的信号；必要时在 TradingView 中显式导出指标及信号。确保获得的文件确实来自用户指定的策略与图表。
- `backend/app/pine/verification.py` 的 `compare_csv(reference, actual, numeric_columns, atol, rtol)` 接受 DataFrame；`open_time` 使用 UTC 毫秒时间戳或带时区的 ISO 时间，并需要 `open`, `high`, `low`, `close`, `volume`, `long_entry`, `short_entry`, `long_close`, `short_close` 列。参考数据由用户提供或明确取得；不能用 Python 自己的结果冒充 TradingView 参考。
- OHLCV 按精确值对照；可选指标按报告中的容差对照。检查缺失 K 线、指标及各个信号差异，记录时间、字段和原因。报告覆盖范围、容差、差异数量及是否存在未验证部分。通过对照只证明该样本和这些列匹配，不证明所有行情或成交一致。

## 回测与模拟

- 使用 `backend/app/pine/backtest.py` 或 `/api/pine/backtest`，记录行情区间、初始资金、仓位、手续费和滑点；确认仅使用已结束 K 线。按实际实现解释下一根开盘价成交、反手及期末持仓，说明资金费率、流动性、保证金或强平等未建模因素。收益差异应拆分信号差异与成交模型差异。
- 使用 `/api/pine/paper/{account_id}` 系列接口或 `backend/app/pine/paper.py` 进行本地模拟。该账户通过已确认 K 线重放并按历史下一根开盘价模拟成交，不是 Binance 测试网订单或真实成交。新行情可由模拟运行器获取；记录运行状态和错误。账户文件锁只在进程内有效，使用单进程管理同一账户。
- 不自动提交实盘订单，也不将解析、回测或模拟结果标记为实盘批准。若用户要求实盘，先核实项目现有能力和其明确授权，报告尚缺的实现与验证，不伪造下单结果。

## 交付

报告源码和数据来源、可运行代码位置、已支持及拒绝的语义、逐 K 线对照结果、回测配置与结果、模拟状态，以及实际未完成的步骤。若缺少 TradingView 参考数据，明确标记“尚未完成一致性验证”。
