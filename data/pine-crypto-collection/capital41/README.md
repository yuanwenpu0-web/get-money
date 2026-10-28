# CAPITAL41 加密货币永续合约脚本

本目录收录 [damianpitt/capital41-indicators](https://github.com/damianpitt/capital41-indicators) 中明确标为加密货币专用的 1 个脚本：Crypto Perp Stress。上游仓库 README 将其单列在“Crypto-Specific”下；其他可用于 BTC 的通用指标未计入。源码来自固定提交 [`ce64522f7f747abb00b37503c00e2955a6d687d2`](https://github.com/damianpitt/capital41-indicators/tree/ce64522f7f747abb00b37503c00e2955a6d687d2)，[原始许可证](LICENSE)为 MIT，转载须保留版权和许可证声明。

## Crypto Perp Stress：加密货币永续合约压力指标

- [本地完整源码](Capital41_Crypto_Perp_Stress/Capital41_Crypto_Perp_Stress.pine)
- [固定提交的上游源码](https://github.com/damianpitt/capital41-indicators/blob/ce64522f7f747abb00b37503c00e2955a6d687d2/Capital41_Crypto_Perp_Stress/Capital41_Crypto_Perp_Stress.pine)
- [上游 README](https://github.com/damianpitt/capital41-indicators/blob/ce64522f7f747abb00b37503c00e2955a6d687d2/README.md)
- 源码 SHA-256：`262a8db41dfa939ef7a225cf2541d7011d9adb6aef3e5616394f6e07a074f872`

以下原文取自源码开头的作者说明；中文为对应翻译。

**原文：**

> What it does:
> - Detects crypto-specific stress using perp basis, OI impulse, and liquidation wick behavior.
> - Flags long-squeeze risk, short-squeeze risk, cascade starts, and reset completion.
> - Includes weekend sensitivity controls for 24/7 markets.
>
> TradingView setup:
> 1) Open TradingView -> Pine Editor.
> 2) Paste this script, Save, and Add to chart.
> 3) Use intraday or 4h on liquid perp pairs.
> 4) Set Spot/Perp/OI symbols for your exchange.
>
> Usage guide:
> - Long-squeeze risk: crowded longs + OI build + downside liquidation wick.
> - Short-squeeze risk: crowded shorts + OI build + upside liquidation wick.
> - Cascade tags: stress + break of prior bar extreme.
> - Reset complete: basis normalizes, OI flushes, and volatility compresses.

**中文翻译：**

- 使用永续合约基差、未平仓量（OI）变化和类似爆仓的影线行为，检测加密市场特有的压力。
- 标记多头挤压风险、空头挤压风险、连锁走势开始以及重置完成，并提供适用于全天候市场的周末灵敏度设置。
- 在 TradingView 的 Pine 编辑器中粘贴源码、保存并添加到图表；作者建议在流动性较好的永续合约上使用日内或 4 小时周期，并设置对应交易所的现货、永续合约及 OI 品种。
- 多头挤压风险由多头拥挤、OI 增长及下方爆仓影线组合判断；空头挤压风险使用空头拥挤、OI 增长及上方爆仓影线；连锁走势标记还要求价格突破上一根 K 线极值；重置完成要求基差回归正常、OI 快速下降及波动率收缩。

**源码核对备注：**现货和永续合约输入默认均为当前图表品种，使用基差前需要分别配置。OI 品种可选；缺失时默认以成交量变化替代，因此该读数不是实际 OI。所谓爆仓影线是通过影线长度、成交量放大和收盘位置推断，并非直接读取爆仓数据。此脚本没有读取资金费率。信号不等同于已验证的交易策略。
