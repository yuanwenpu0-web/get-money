# Task Plan: Pine 策略工作流改造

## Goal
实现从 Pine 源码分析、对照到本地回测的可验证工作流，并明确后续模拟和实盘接入边界。

## Next Step
交付当前实现，后续使用真实 TradingView 样本对照并验证 Binance 网络。

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] 盘点现有架构、行情和交易模块
- **Status:** complete

### Phase 2: Planning & Structure
- [x] 确立独立 Pine 语义链和禁用旧自动交易入口
- **Status:** complete

### Phase 3: Implementation
- [x] 完成有限转换、对照、回测、模拟及界面
- **Status:** complete

### Phase 4: Testing & Verification
- [x] 验证 API、回测、对照、数据库启动和构建
- **Status:** complete

### Phase 5: Delivery
- [x] 说明实际支持范围与未完成的交易链路
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 新链路独立于旧评分策略 | 保持原信号语义 |
| 阻止旧调度自动启动 | 避免沿用过期交易配置 |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| Binance 公共接口探测超时 | 保留模拟数据测试；真实网络接入尚未验证 |
