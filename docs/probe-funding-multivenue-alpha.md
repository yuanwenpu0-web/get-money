# P11-005 · 币安官方 Alpha 地址佐证

判定 `BLOCKED` / `insufficient`，净收益回测 `NOT_RUN`。本轮仅查公开资产元数据与费用证据，不改变冻结窗口、K、回看及持有期。

## 官方公开响应

币安接口 `https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list` 返回 HTTP 200、success=true，共 686 条记录。官方文档：`https://developers.binance.com/docs/zh-CN/alpha/market-data/rest-api/token-list`。

对原 146 个新增现货候选，剥一层最长优先的合约倍数前缀后匹配 Alpha symbol，再比较 MEXC 合约地址。EVM 地址忽略大小写，其他地址精确比较。

| 项目 | 数量 |
|---|---:|
| 原候选 | 146 |
| Alpha 存在同名 symbol | 119 |
| Alpha 与 MEXC 地址一致 | 88 |
| 已明确关联永续合约底层身份 | 0 |

88 个地址匹配是官方 Alpha 资产与 MEXC 资产之间的佐证，不是已验证的永续可对冲币数。此数量不能与前轮 Gate/MEXC 的 107 个匹配直接相加。

样例：KOMA 对应 BSC 地址 `0xd5eaaac47bd1993d661bc087e15dfb079a7f3c19`，SIREN 对应 BSC 地址 `0x997a58129890bbda032231a52ed1ddc845fc18e1`，均与 MEXC 一致。BROCCOLIF3B 在本轮 Alpha 列表未找到同名 symbol；这不证明其资产不存在。

## 费用证据与限制

直接请求 `https://www.gate.com/zh/fee` 返回 HTTP 403，原始响应保存在配对 JSON。搜索引擎能展示当前费率页摘要，但没有覆盖冻结窗口的生效日期和逐币历史费率，不能用摘要替代历史费用数据。未尝试绕过访问限制。

Alpha 列表不包含明确的 USD-M 永续合约关联；同名与地址一致尚不能解决代币迁移、ticker 复用及历史时点身份问题。完整历史盘口和有生效日期的费用证据也仍缺失，因此 E1 未完成验证，E2–E6 不运行。

## 验证与下一步

146 条候选唯一性与数量断言通过。逐币结果、官方完整响应及费用页失败响应已保存。没有生产代码变更、交易操作或付费数据购买。

下一步需要明确关联永续合约的官方资产引用，以及原冻结窗口的完整盘口和历史费用。继续增加同名元数据不能替代这些证据；当前研究仍不能得出净收益或实盘可行性结论。
