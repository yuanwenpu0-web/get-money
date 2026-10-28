# TradingView 本地策略

目前移植两份 Eterna 脚本的信号：

- [SMA Crossover](../data/pine-crypto-collection/eterna/strategies/trend-following/sma-crossover/strategy.pine)：默认 SMA 50 / 200 交叉。
- [Donchian Channel](../data/pine-crypto-collection/eterna/strategies/trend-following/donchian-channel/strategy.pine)：默认 20 根通道，突破参考前一根通道；原策略以中轨交叉退出。

策略页面可以启用本地过滤器，并绑定一个或多个系统策略。默认关闭；只有启用且绑定后才参与入场判断。同一系统策略绑定多个启用的本地过滤器时，必须在同一根已收盘 K 线全部出现同方向信号，才允许系统开仓。系统仍负责仓位、风控和持仓退出；本地过滤器不是独立交易策略，其原始退出规则不会控制系统持仓。信号缺失、历史不足或数据断档时过滤器不放行。

## 原脚本独立回测

`backend/app/backtest/tradingview_native.py` 单独模拟原脚本的市场单：收盘生成订单，下一根开盘成交；同向不加仓，反向信号反手。默认初始资金 10,000、仓位 10%、单边手续费 0.075%，无滑点。末根信号不成交，未平仓按末根收盘价计入权益。此简化模拟不支持保证金强平、Bar Magnifier、盘中成交或 TradingView 的全部撮合细节；结果不能视为与 TradingView 完全一致，也不是上述绑定系统策略后的结果。

在仓库根目录运行（PowerShell）：

```powershell
$env:PYTHONPATH='backend'
.\.venv\Scripts\python.exe backend/scripts/backtest_tradingview_native.py --symbol BTCUSDT --interval 15m --days 30
```

脚本读取配置的本地行情数据库，以最新已收盘数据为终点，要求窗口内数据连续。策略在窗口内预热，因此窗口开头不会立即产生需要历史数据的信号。也可用 `--database-url` 指定数据库。

本次本地历史数据示例：BTCUSDT、ETHUSDT，15 分钟，2026-09-27 00:30 至 2026-10-27 00:30（北京时间，结束为末根收盘），各 2,880 根。收益和回撤按权益计算，含未平仓浮动盈亏；胜率仅按已平仓交易计算。

| 品种 | 策略 | 收益 | 最大回撤 | 已平仓 | 胜率 |
| --- | --- | ---: | ---: | ---: | ---: |
| BTCUSDT | SMA 交叉 | -1.8096% | -2.0697% | 23 | 8.70% |
| BTCUSDT | Donchian | -1.7000% | -1.7112% | 103 | 23.30% |
| ETHUSDT | SMA 交叉 | -0.3675% | -1.1374% | 19 | 15.79% |
| ETHUSDT | Donchian | -1.4125% | -1.4125% | 105 | 26.67% |

BTC 两项和 ETH SMA 在窗口末仍有持仓；BTC Donchian 的最后一根还产生了未成交平仓信号。后续新增的策略及当前迁移范围见 [迁移状态](tradingview-migration-status.md)。

## 绑定系统策略的回测示例

使用同一数据库，将 SMA 交叉作为 `trend_follow`（SuperTrend 趋势跟随）的入场过滤器，BTCUSDT 15 分钟，初始资金 10,000、单边手续费 0.04%、单边滑点 0.02%，关闭优化。系统回测使用 30 天数据切分训练和测试；样本外测试覆盖北京时间 2026-10-19 14:30 至 2026-10-26 23:00（K 线开盘时间），707 根。结果：收益 **+0.0353%**、最大回撤 **-0.5206%**、已平仓 2 笔、胜率 50%，260 根系统信号被本地过滤器阻止入场。成交数量过少，无法据此判断策略有效性。

这次只读数据库缺少 `sentiment` 表，运行示例时将情绪输入视为空；没有修改用户数据库，也没有改变用户的实际启停或绑定设置。系统回测按当前保存的绑定配置过滤入场，记录实际过滤器参数；持仓出场沿用系统规则。
