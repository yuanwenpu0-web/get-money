# 未决项登记（Open Decisions）

| Date | Source | Open Item | Related Constraints | Current Leaning | Blocked By | Resolves When | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-25 | backend (贝洛奇) QA 验收 advisory | 孤儿仓对账兜底：开仓市价单在交易所成交后、本地 Order 记录提交前若发生异常（如 DB 写入失败），会出现"交易所有仓位、本地无订单"的不一致；当前实现仅在 JobLog 记录回滚日志，无自动对账兜底。 | live.py `_place_live`/`_rollback_open`；P0 规则"live 下单失败不得留脏订单"只能覆盖保护单挂失败路径，无法覆盖本地记账层异常。 | 出现该场景时依赖 JobLog 人工核对交易所仓位；后续可增加启动时/周期性对账：用 positionRisk 与本地 OPEN 订单做差集比对并告警。 | 无（实现成本可控，优先级低）。 | 测试网实测出现该场景，或切换主网真实资金之前必须解决。 | OPEN |
| 2026-10-25 | backend (贝洛奇) QA 验收 advisory | 交易所侧止损止盈触发后，本地按 premiumIndex 标记价结算（exchange_sl/exchange_tp/exchange_closed），而非保护单实际成交均价；get_order 返回的 avgPrice 可提供更准确的结算价。 | live_sync.py `_exchange_close_reason`/`_settle_exchange_closed`；本地记账与交易所对账可能存在小幅偏差。 | 维持标记价简化结算，README 第 10 节已声明该边界；改进方向是判定 reason 时顺带读取保护单 avgPrice 用于结算。 | 无。 | 对记账精度/对账一致性有更高要求时（如主网真实资金）。 | OPEN |
