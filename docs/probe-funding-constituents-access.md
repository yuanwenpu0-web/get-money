# P11-006 · 永续指数成分与历史盘口访问核验

判定 `BLOCKED` / `insufficient`；净收益回测 `NOT_RUN`。范围为 Gate 1b 数据与身份证据，非 edge 判定。

## 工具目标与来源

本轮仅访问 `developers.binance.com`、`fapi.binance.com`、`docs.tardis.dev`、`datasets.tardis.dev`，未使用 routify 或其他代理。币安官方文档页返回 HTML，前 15000 字摘录未提取到可用接口字符串；实际接口另行请求，返回真实 JSON。Tardis CSV 地址按官方文档模板构造，未借用历史调用中的代理地址。

## 实际结果

1. 官方文档地址 `https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Query-Index-Price-Constituents.md` 返回 HTTP 200；原始 HTML 的前 15000 字保存在 JSON。无扩展名地址返回 HTTP 202、空响应，未将空响应当作文档证据。
2. `https://fapi.binance.com/fapi/v1/constituents?symbol=KOMAUSDT` 返回 HTTP 200，包含五个成分：Gate、KuCoin、MEXC、PancakeSwap V2、币安永续自身。该响应直接关联 KOMA 永续与现货/链上价格来源，但不包含代币链地址，也没有历史成分生效日期。
3. Gate 免费月初盘口样本 `https://datasets.tardis.dev/v1/gate-io/incremental_book_L2/2026/11/01/KOMA_USDT.csv.gz` 返回 HTTP 200，2,051,817 字节、240,308 条数据。字段为 exchange、symbol、timestamp、local_timestamp、is_snapshot、side、price、amount，首条含快照。响应哈希 `6cead3472385bcf725bd71616475aff183e71190c6ff1fc408fdfbbd8db4185f`。
4. 相邻非月初日期 2026-11-02 返回 HTTP 401，验证文档所述的 API key 访问限制。未尝试绕过限制、未访问付费凭据。

## 证据边界

KOMA 当前指数成分是一个官方关联样例，不能外推为 146 个候选全部完成身份验证，也不能证明历史窗口内成分不变。链上成分字符串尚未解析并核对合约地址。

已下载的样本只有 2026-11-01 单日，不是完整冻结窗口。原窗口完整盘口下载天数为 0，历史有效费率证据仍缺失。不能用样本插值替代历史成本，也不能将当前指数和单日盘口拼成可执行收益。

## 验证与下一步

程序断言通过：官方 KOMA 响应包含 Gate 与 MEXC；样本行数和哈希一致；完整窗口盘口下载天数为 0。原始访问记录见 `probe-funding-constituents-access.json`，样本头部、尾部与校验信息见 `probe-funding-gate-book-sample.json`，未保存整份盘口文件。

下一步需取得历史成分快照、完整原窗口盘口和有效日期的费用证据。没有生产代码变更、交易操作或净收益计算。
