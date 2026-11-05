# 策略结构、模块与迁移映射：当前代码状态

本文依据当前工作区源码编写，描述代码具备的行为，不代表数据库已执行迁移，也不代表策略已通过收益验证。工作区包含尚未提交的改动；本文不推断部署状态或数据库中的实际数据。

## 当前结构

```text
前端策略页面
  ├─ 获取信号指标选项 → /strategies → StrategyConfig + StrategySpec
  └─ 策略列表 → /bindings → StrategyBinding × SymbolPool
                                     ↓
                        active_bindings：参数合并、币池展开
                                     ↓
                      StrategyEngine：StrategyIndicator 包装单个 StrategySpec
                                     ↓
                            信号 → 交易服务 / 订单

启动路径：lifespan → init_db → migrate_binding_enablement → 备份并迁移绑定启用状态
独立路径：strategy_migration.migrate(engine) → 来源快照 + 版本 payload
          strategy_migration.rollback(engine, version_id) → 撤销标记
          rollback_binding_enablement(engine, offline=True) → 校验备份并恢复受影响的绑定开关
```

引擎仍以单个 `strategy_code` 查找已有 `StrategySpec`，包装成 `StrategyIndicator` 后计算信号；绑定、参数与交易执行仍走现有代码。迁移版本和独立指标组合模块未接入该运行链路。参见 `backend/app/strategy/engine.py:149`、`backend/app/strategy/bindings.py:157`。

工作区事实与“已移除独立策略卡、订单快照与组合框架”的表述并不完全一致：策略页当前以绑定列表呈现，没有独立策略卡模型；但订单参数快照仍存在且已接入模拟盘及实盘下单，独立指标组合模块也仍存在，只是未接入运行链路。以下按实际代码区分。

## 现有实体与映射

| 界面或概念 | 当前代码实体 | 当前含义与边界 |
| --- | --- | --- |
| 信号指标 | `StrategySpec`、`StrategyConfig`，由 `StrategyIndicator` 适配 | 前端“信号指标”对应原有策略实现及其配置；不是新建的可独立拼装指标实体。`StrategyConfig` 保存名称、开关和参数。 |
| 策略列表中的一条策略 | `StrategyBinding` / `strategy_bindings` | 一条绑定对应一个 `strategy_code`、一个 `pool_code`、一个 `interval`，附开关、优先级、参数和出场设置；三者有唯一约束。 |
| 币种池 | `SymbolPool` / `symbol_pools` | 支持静态或规则池，保存成员、规则及人工增删；运行时再解析成员。 |
| 币种参数覆盖 | `SymbolParam` / `symbol_params` | 按策略代码、币种、周期覆盖；不以池为键，可影响同策略同周期的多个绑定。 |
| 全局设置 | `AppState` / `app_state` | 参与参数与交易范围解析；迁移快照只选取与策略及范围有关的键。 |
| 信号 | `Signal` / `signals` | 保存策略、池、方向和 `reason` 等；引擎的 `TradeSignal.params` 仅进程内传递，序列化时移除。 |
| 订单 | `Order` / `orders` | 保存策略、池、周期、`binding_id` 和可空的 `params_snapshot` 等；快照保存经校验的有效参数，不包含策略代码实现或迁移版本。历史订单可无快照。 |
| 研究指标目录 | `catalog_indicators` 与 `api/indicators.py` | 与上述策略指标兼容层是不同的目录和注册体系，不应混为一谈。 |

实体定义见 `backend/app/models.py:81`、`backend/app/models.py:216`；界面语义见 `frontend/src/components/BindingMatrix.vue:441`、`frontend/src/views/Strategies.vue:302`。有效参数一般按全局设置、策略默认及配置、绑定覆盖、币种覆盖逐层解析；手续费与滑点受全局成本约束，不能由策略或币种覆盖。出场覆盖另由绑定字段注入，具体以 `backend/app/strategy/bindings.py:115` 和 `backend/app/strategy/rules.py` 为准。

## 模块清单

| 模块 | 当前职责 |
| --- | --- |
| `backend/app/strategy/registry.py`、`specs/` | `StrategySpec` 定义及注册，具体策略声明参数、特征准备和信号等。 |
| `backend/app/strategy/indicators.py` | 包装已注册的 `StrategySpec`，转发参数解析、lookback、prepare、signals；提供独立的显式指标注册表。 |
| `backend/app/strategy/engine.py` | 读取行情和配置，通过兼容包装计算单策略信号，附加观测信息和出场覆盖。引擎直接构造包装对象，不依赖独立指标注册表自动填充。 |
| `backend/app/strategy/indicator_composition.py` | 独立组合代码：首项提供信号，其余项同方向确认；逐项独立计算并返回详情。目前源码引用仅见其测试，未接入引擎、绑定接口或交易执行。 |
| `backend/app/strategy/rules.py`、`signals.py` | 参数解析与校验、通用信号处理。 |
| `backend/app/strategy/bindings.py`、`pools.py`、`symbol_params.py` | 解析绑定、池成员和币种级参数；`exits.py`、`support_levels.py` 提供出场覆盖及支撑位来源。 |
| `backend/app/strategy/strategy_migration.py` | 启动时调用绑定启停迁移；另提供绑定启停回滚及独立版本快照的生成、撤销函数。版本记录不参与运行时切换。 |
| `backend/app/api/strategies.py`、`bindings.py`、`pools.py` | 提供已有策略配置、绑定和币池接口；策略页面仍使用这些概念对应的后端接口。 |
| `backend/app/trading/service.py`、`backend/app/backtest/` | 分别负责交易巡检与回测执行；某些专用撮合能力仅在特定路径支持。 |
| `frontend/src/views/Strategies.vue`、`components/BindingMatrix.vue`、`router.js` | 将原策略配置称为信号指标、将绑定列表称为策略列表，并区分研究指标目录；提供绑定和出场等配置界面。 |
| `backend/app/research/combo_config.py`、`combo_engine.py` | 研究与寻优用途的组合配置及执行代码，不等于运行时通用指标自由组合。 |

## 启动时的绑定启用状态迁移

应用启动时，`lifespan` 调用 `init_db()`，完成建表、补列等操作后，在默认状态与种子数据写入之前调用 `migrate_binding_enablement(engine)`（`backend/app/main.py:28`、`backend/app/db.py:409`）。这是真正接入启动流程的数据库写操作，并不代表任何特定数据库已经执行过。

首次执行时，迁移在事务和来源表锁下，将全部绑定行、全部策略配置行和受影响绑定 ID 以 `schema=2` 快照及摘要保存：仅将当前 `enabled=1` 且对应 `StrategyConfig` 不存在或 `enabled` 不为 1 的绑定设为 0，再写入 `legacy_strategy_enablement_v1` 标记；原本已停用的绑定不变。已有标记时不再重复更新，因此之后用户可独立启用绑定。迁移后的巡检由绑定开关、策略是否注册和池开关等决定，不再以旧策略总开关作为绑定的额外门槛（`backend/app/strategy/strategy_migration.py:140`、`backend/app/strategy/bindings.py:173`）。

默认种子中的 `absorption_reversal × major × 15m` 和 `absorption_reversal × alt × 5m` 均为关闭，仅当绑定表为空、相应策略代码在允许列表且池存在时写入；绑定表已有记录时不补种。创建绑定的 API 请求省略 `enabled` 时也默认为关闭；这不代表已有数据库中的绑定全都关闭（`backend/app/strategy/defaults.py:67`、`backend/app/strategy/defaults.py:131`、`backend/app/schemas.py:87`）。

`rollback_binding_enablement(engine, offline=True)` 是显式调用的离线回滚函数：调用方须先停止全部服务、巡检及开仓进程；`offline=True` 只是调用方声明，代码不验证进程状态，也不是跨进程运行锁。函数验证备份摘要和迁移标记，再比较全部绑定及全部策略配置是否与迁移后预期一致；任何新增、删除或变更均会拒绝回滚并撤销事务。校验通过后只恢复受影响绑定的原 `enabled` 值，并写入 `app_state.binding_enablement_rollback` 标记；该标记使 `active_bindings` 返回空列表及错误，也使下次新版启动失败。它不等于停止已运行的进程或取消已有订单管理。

重复回滚在备份和迁移标记有效时直接返回。缺失、损坏或旧版仅含部分绑定的备份不能用于该回滚，已有迁移标记也不会自动补建完整备份。源码中该函数的调用仅见测试，不能视为已部署的回滚入口（`backend/app/strategy/strategy_migration.py:181`、`backend/tests/test_strategy_migration.py:181`）。

旧接口 `PUT /api/strategies/{code}` 对已允许的策略代码，只要请求中 `enabled` 非空，无论开或关都返回 HTTP 409；参数更新仍可用。绑定启停应通过绑定接口操作。无有效参数的独立引擎调用仍可能检查旧 `StrategyConfig.enabled`，不能将旧字段描述为从所有路径移除（`backend/app/api/strategies.py:80`、`backend/app/strategy/engine.py:131`）。

## 独立版本快照的实际映射

`backend/app/strategy/strategy_migration.py:48` 把 `strategy_bindings`、`strategy_configs`、`symbol_pools`、`symbol_params` 的完整行以及选定的 `app_state` 行序列化到原始快照。对每一条绑定，版本 payload 的 `strategies` 数组生成一项：

| 来源 | 版本 payload 中的内容 |
| --- | --- |
| 绑定 | `binding_id`、`strategy_code`、`pool_code`、`interval`、`priority`、`binding_enabled` |
| 配置及池开关 | 保存 `strategy_enabled`、`pool_enabled`；有绑定启停迁移标记时，`enabled` 为绑定和池开关的合取，否则还要求旧策略开关开启 |
| 全局状态、策略配置、绑定及出场配置 | 经 `binding_params` 解析后的 `params` |
| 同策略同周期的 `SymbolParam` | `symbol_params` 字典，以币种为键保存合并解析结果；并非只收录当前池成员 |
| `SymbolPool` | `pool` 保存原始池行，包括规则和人工覆盖；不是历史时点的展开成员快照 |
| 未绑定配置及覆盖 | 保留在原始快照；不会为其单独生成绑定计划项。 |

显式调用 `migrate(engine)` 时，代码在目标连接中按需创建 `strategy_migration_snapshots`、`strategy_migration_versions`、`strategy_migration_rollbacks` 三张记录表，写入内容哈希寻址的快照与版本；同输入幂等。支持 SQLite 和 PostgreSQL，使用锁与触发器保护记录。`rollback` 核验版本、快照与来源是否漂移后仅追加撤销标记，并禁止相同已撤销版本再次生成；不恢复来源数据，也不切换当前运行版本。数据库所有者仍可移除触发器。详见 `backend/app/strategy/strategy_migration.py:106`。

上述 `rollback(engine, version_id)` 与启动绑定迁移的 `rollback_binding_enablement(engine, offline=True)` 是不同函数：前者仅追加版本撤销标记，不恢复绑定；后者恢复受影响绑定开关，阻止新版启动并使活动绑定解析返回空列表。独立版本的 `migrate` / `rollback` 调用在源码搜索中仅见测试；绑定启停迁移则已接入启动。尚未完成旧运行数据向新版本运行模型的迁移，也没有将运行时切换到版本表。

测试代码覆盖内存 SQLite 的迁移、幂等、异常和回滚冲突等情形；PostgreSQL 的触发器测试只检查模拟连接收到的 SQL，**未验证真实 PostgreSQL 的建表、锁、事务、触发器和回滚行为**。本次是源码核查，未执行这些测试，也不能推断目标数据库或生产环境的状态。

## 已有能力与缺口

- 已接入：前端策略页使用绑定列表，配置单个信号指标、池、周期、参数和出场方式；后端仍使用原有策略和绑定模型。当前页面没有独立策略卡实体；样式文件中仍有遗留卡片样式。
- 已有但未接入：指标兼容层提供独立注册与查询接口，运行时按策略代码直接包装；`IndicatorComposition` 已实现首项信号与其余项同向确认，并有单元测试，但未接入策略页面、绑定、引擎或交易执行，也不是可任意配置条件与优先级的通用编排系统。研究组合代码属于另一条路径。
- 已接入：模拟盘和实盘下单将校验后的生效参数以 `{"version": 1, "params": ...}` 格式写入 `Order.params_snapshot`；普通订单管理及巡检信号计算使用解码后的快照。订单详情按需返回解码参数与 `valid`、`invalid` 或无快照的 `legacy` 状态，常规列表不包含快照。研究订单管理有单独路径（`backend/app/strategy/bindings.py:292`、`backend/app/serializers.py:18`、`backend/app/trading/order_manager.py:63`、`backend/app/trading/service.py:157`）。
- 旧订单快照兼容：无版本的参数对象只补代码明确登记的历史 `exit` 字段；`supertrend_current` 在闭环未启用时还补登记的闭环历史字段，已有值不覆盖，再按当前参数规则完整校验。版本 1 不补字段；未知版本、缺失或无效字段、重复 JSON 键及非有限数字等被拒绝。已有但无效的快照不会回退当前配置；完全没有快照的旧订单仍按当前配置或现有信号处理。兼容范围不等于所有历史快照都可读取（`backend/app/strategy/bindings.py:280`）。
- 币种覆盖异常：启用绑定校验和活动绑定加载使用严格读取；有活动绑定时，任意覆盖行的 JSON 错误或非对象会使整次活动绑定加载抛错。已加载的非空覆盖若参数无效或包含成本，`params_for` 抛错，不回退绑定参数；巡检逐币评估记录错误并继续处理其他项。范围须限定于这些路径：`get_override` 和默认非严格 `load_map` 等旧路径仍可能忽略损坏的覆盖（`backend/app/strategy/symbol_params.py:51`、`backend/app/strategy/bindings.py:72`、`backend/app/trading/service.py:105`）。
- 未接入：订单参数快照不保存指标代码实现、历史行情或迁移版本，不能单独保证历史执行完全复现；字段可空，不能推断既有数据库已迁移或全部历史订单都有快照。
- 专用撮合门控已接入：启用绑定的创建和更新会校验绑定参数及相关币种覆盖，`opening_candle_breakout` 与开启 `cl_enabled` 的 `supertrend_current` 被拒绝，返回 HTTP 400；停用的绑定仍可保存。巡检展开、逐币参数解析及模拟盘、实盘开仓也调用相应门控，普通引擎评估在 `respect_enabled=True` 时执行检查；预演等显式关闭该选项的路径不受此检查约束。两者在回测中路由到各自专用撮合，且不支持通用验证集选参（`backend/app/api/bindings.py:82`、`backend/app/strategy/bindings.py:38`、`backend/app/backtest/runner.py:437`）。这不是对所有专用撮合策略的统一能力声明。
- 已有参数覆盖、池规则和部分出场配置，但策略实现、撮合和入口限制仍由代码规定，不能称为全配置驱动。

## 风险

1. 名称映射易误解：界面的“策略”是绑定，“信号指标”仍是原策略实现及配置；研究指标目录又是另一套体系。
2. 独立版本快照不参与运行，其撤销标记不能回退运行配置；绑定启停回滚要求调用方确保离线，并依赖完整备份及全量漂移校验，不能混为一谈。
3. 动态池的规则与人工覆盖可被记录，但实际成员随数据和全局交易范围变化，不能由池定义直接还原历史成员。
4. 旧订单可能没有参数快照，回退当前配置时会受后续变更影响；已有参数快照也不包含代码、行情和迁移版本，同一币种覆盖可跨池生效。
5. 快照触发器的保护依赖数据库权限及备份；脚本锁表、DDL 和方言支持均需在目标环境验证，不能由内存测试推断生产效果。
6. 出场覆盖或研究组合的存在不代表具有统计优势，也不代表所有执行通道均支持；启用决策需遵循项目回测及验证流程。

## 后续步骤（建议，尚未实现）

1. 明确目标领域模型与术语，定义信号指标、组合条件、策略、绑定、版本与订单快照的边界和关系，避免仅按界面名称推断数据库结构。
2. 若需要迁移到新的版本化运行模型，先设计目标表、字段映射、动态池与未绑定数据的处理、校验规则、幂等与回退方案；在隔离数据库验证快照与数据一致性，再决定切换方式。
3. 设计并实现运行时版本读取、切换及回退，并覆盖参数、信号、回测和交易执行的一致性；不能直接把当前撤销标记当作运行时回滚。
4. 在现有订单参数快照基础上评估版本、代码及历史数据追溯需求，并验证旧订单兼容性和配置修改前后的行为。
5. 如需将指标组合用于实际执行，先定义输出契约、条件表达与校验，评估现有同向确认模块的适用范围，并在回测和支持的交易通道分别验证；明确专用撮合限制。
6. 逐步将需要配置化的行为转为明确的配置契约，保留必要的代码实现与安全校验；完成端到端测试及目标数据库集成测试后再评估部署。
