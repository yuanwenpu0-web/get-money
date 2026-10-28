# Findings & Decisions

## Requirements
-

## Research Findings
-

## Technical Decisions
| Decision | Rationale |
|----------|-----------|

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
-

- Martin Luk 对应 strong_pullback；15m 入场，4h 趋势过滤；默认 EMA 软退出。
- 当前 paper 自动巡检启用；全局周期 1h，已有六笔非该策略的 1h 持仓。策略配置更新会影响该策略已有订单。
- 先按过去选取固定参数，最新区间留出；保留手续费和滑点，比较关闭 EMA 退出的移动止损。
