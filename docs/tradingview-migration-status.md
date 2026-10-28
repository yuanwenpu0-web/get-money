# TradingView 策略迁移状态

系统策略页面的「TradingView 策略源码目录」按 `data` 中的实际源码扫描，并区分源码查看与可执行功能。目录当前包含 68 份策略：66 份 `.pine` 和 2 份无扩展名源码。SMA 交叉和 Donchian 通道支持既有入场过滤及简化研究回测；另外 EMA Ribbon、Keltner Channel、Linear Regression Channel、Ichimoku Cloud、Chandelier Exit、CCI、Awesome Oscillator、Stochastic RSI、Williams %R 和 BB EMA ETH 已移植信号和简化研究回测。其余 56 份尚未迁移。研究回测不代表原策略的全部订单已接入系统，目录不会启用交易。

在源码详情可填写币种、周期和天数并运行研究回测，使用本地最新已收盘的连续行情窗口和源策略默认参数。源策略的默认资金、仓位及费用可能不同，结果页面会显示具体设置和限制。CCI、AO 与 Williams %R 在源码中只允许空仓入场，反向信号仅平仓；Williams %R 的入场条件互相矛盾，故忠实保留后不会产生交易。

BB EMA ETH 源码默认使用 2018-01-01 至 2022-12-31 UTC 的训练窗口；用近期行情及该默认配置回测会是零成交。浮点信号比较按文档保留九位小数；恰好处于舍入中点时使用 Python 的舍入规则，该边界尚未与 TradingView 对照。

全量原样执行需要 Pine 语言和交易订单语义，而不能只提取 `strategy.entry` 条件。源文件中有多周期请求、加仓、`strategy.exit` 止损和限价、盘中计算、按收盘成交及风控命令；策略可能组合使用这些功能。当前系统的固定风控和订单生命周期不等同于这些脚本。交易所真实订单的部分成交、精度和延迟也无法简单等同于历史模拟成交。

两份源码使用了外部 Pine 库：`fedehide_bot/trading-bot.pine` 引用 `TradingView/ta/7`，`shidu_gma` 引用 `diegonavarromartinez/base_helpers/4`。后一项目附有同名本地源码，但版本是否与线上第 4 版一致尚未确认。应保留各脚本的原始许可和来源，并在确认版本后处理依赖。

在临时目录评估了 PineTS 0.13.1：一些源码可以运行，但 Eterna SMA 交叉原文在 `input.string` 处无法转换；原始数组输入下的多周期探测也没有得到期望的 60 分钟数据。上游列出了与 TradingView 不同的[模拟器行为](https://github.com/LuxAlgo/PineTS/blob/main/docs/strategy.md#known-divergences)。因此尚不能据此声称策略买卖完全一致，也没有把该运行器加入系统或启用策略。

后续每份策略要完成迁移，需核对其源码及所引用库版本，实现其数据、下单、修改、取消和退出语义，并用相同交易品种、周期、参数及行情逐笔对照 TradingView 结果。实时逐 tick 策略还需要相应的实时数据和事件处理，不能仅凭历史 K 线验证。
