# P11-004 · Gate 逐币历史盘口档案核查

判定：`BLOCKED` / `insufficient`。净收益回测：`NOT_RUN`。范围为 Gate 1b 数据可行性，非 edge 判定。

冻结窗口：2025-09-22T01:00:00.002Z 至 2026-11-12T00:00:00.000Z；不改变 K、回看或持有期。

## 实际请求与覆盖

匿名请求 `https://api.tardis.dev/v1/exchanges/gate-io` 返回 HTTP 200。使用 datasets.symbols 的逐币日期和 dataTypes，与前轮 146 个新增现货候选的 Gate 币种代码匹配。

| 项目 | 数量 |
|---|---:|
| 新增现货候选 | 146 |
| 找到档案元数据 | 127 |
| 含 incremental_book_L2 类型 | 127 |
| 元数据日期覆盖完整冻结窗口 | 82 |
| 有盘口但仅覆盖部分窗口 | 45 |
| 未找到对应档案 | 19 |
| 已核实币安底层资产身份 | 0 |

完整窗口要求 availableSince 不晚于窗口起点、availableTo 不早于终点。此处仅验证数据供应商声明的日期范围，不证明期间无缺口、持续可交易或资产身份相同；82 不能作为已验证可对冲新增币数。币池与今日上架快照仍存在幸存者偏差。

## 档案访问条件

来源：`https://docs.tardis.dev/downloadable-csv-files/api.md`。

CSV 下载模板为 `https://datasets.tardis.dev/v1/:exchange/:dataType/:year/:month/:day/:symbol.csv.gz`。文档声明每月第一日文件可以免 API key 访问，其余日期需要 Bearer API key。本轮未下载完整盘口档案、未访问付费凭据、未购买数据；免费月初样本不能替代完整历史执行成本。

## 币安官方映射核查

公开请求 `https://fapi.binance.com/fapi/v1/exchangeInfo` 返回 HTTP 200。完整响应与实际 symbol 字段集合保存在配对 JSON。

字段包含 baseAsset、underlyingType、underlyingSubType、onboardDate 等，但没有链或代币合约地址字段。因此这个端点无法独立证明币安永续对应的是 Gate/MEXC 合约地址所指的同一资产；不能将同 ticker 或两家现货所地址一致直接计入 E1。此结论只限本轮检查的端点，不能推论币安所有官方资料均无此信息。

## 决策与下一步

E1 的身份验证尚未完成；E2–E6 缺完整历史盘口和可核实的历史费用，保持未运行。当前手续费或单次盘口不能代替历史成本。

后续需要取得原窗口的完整盘口访问、独立官方资产映射及历史费用依据；获取之前不计算净收益，不缩短窗口，也不以月初样本插值补齐。原始响应、逐币分类和限制见 `probe-funding-multivenue-archive.json`。

验证：146 条唯一候选的分类计数断言通过；完整与部分窗口之和等于 L2 档案数，匹配与未匹配之和等于候选数。没有生产代码变更或交易操作。
