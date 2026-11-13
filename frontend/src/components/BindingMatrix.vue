<script setup>
/**
 * 策略 × 币种池 绑定矩阵。
 *
 * 一条绑定 = (策略, 池, 周期) + 一套参数覆盖 + 一套出场口径。同一个策略可以在
 * 「主流 15m，支撑位止损」「山寨 5m，ATR 止损」上各跑一套，互不影响；参数的最终
 * 生效值是 全局设置 < 策略默认 < 本条绑定（后端 resolve_params 负责合并与校验）。
 * 出场方法（exit_method / support_indicator / exit_params）由后端 exits 注册表校验。
 */
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import TablePager from './TablePager.vue'
import { PAGE_SIZE } from '../utils/pagination'

const props = defineProps({
  strategies: { type: Array, default: () => [] },
})

// 绑定变化后通知父级刷新（策略页的「信号预演」要用最新的绑定列表做参数层下拉）
const emit = defineEmits(['change'])

const bindings = ref([])
const pools = ref([])
const loading = ref(false)
const error = ref('')
// 绑定列表服务端分页：每页 15 条，total/enabled 用后端全量口径
const listPage = ref(1)
const listTotal = ref(0)
const enabledTotal = ref(0)
const saving = reactive({})
const intervalOptions = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

const bindingVisible = ref(false)
const bindingSaving = ref(false)
const bindingEditing = ref(false)
// 记录打开编辑弹窗时的身份，用来判断本次是否改了「策略/池」这类重字段
const bindingOriginal = ref({ strategy: '', pool: '' })
const bindingForm = reactive({
  id: null,
  strategy: 'absorption_reversal',
  pool: '',
  interval: '15m',
  priority: 100,
  enabled: false,
  params: '{}',
})

// 出场覆盖：方法与支撑线指标的可选清单由后端注册表提供
const exitOptions = ref({ methods: [], support_sources: [] })
const exitForm = reactive({ method: 'strategy', support: '', params: {} })
const exitMethodOptions = computed(() => exitOptions.value.methods || [])
const exitSourceOptions = computed(() => exitOptions.value.support_sources || [])
const activeExitMethod = computed(
  () => exitMethodOptions.value.find((item) => item.code === exitForm.method) || null,
)
// 当前方法声明的参数（默认值 + 合法区间），界面按它渲染数字输入
const exitParamFields = computed(() => {
  const method = activeExitMethod.value
  if (!method) return []
  return Object.entries(method.defaults || {}).map(([key, fallback]) => ({
    key,
    fallback,
    range: method.ranges?.[key] || [0, 1],
  }))
})

function resetExitForm(methodCode = 'strategy', support = '', params = {}) {
  exitForm.method = methodCode || 'strategy'
  exitForm.support = support || ''
  const method = exitMethodOptions.value.find((item) => item.code === exitForm.method)
  exitForm.params = { ...(method?.defaults || {}), ...(params || {}) }
}

function onExitMethodChange() {
  // 换方法后旧方法的参数不再适用，回到新方法默认值
  const method = activeExitMethod.value
  exitForm.params = { ...(method?.defaults || {}) }
  if (!method?.needs_support) exitForm.support = ''
}

function exitSummary(row) {
  const method = row.exit_method || (row.exit || {}).method || 'strategy'
  if (method === 'strategy') return '指标默认出场'
  return (row.exit || {}).label || method
}

function exitIsUnverified(row) {
  const exit = row.exit || {}
  return Boolean(exit.method) && exit.method !== 'strategy' && exit.verified === false
}

const paramsVisible = ref(false)
const paramsSaving = ref(false)
const paramsTarget = ref(null)
const paramsText = ref('{}')

const poolVisible = ref(false)
const poolSaving = ref(false)
const poolForm = reactive({
  code: '',
  name: '',
  kind: 'rule',
  symbols: '',
  rule: '{"base": "all"}',
  // 人工增删层：规则解析完之后叠加，自动维护不会覆盖
  manualInclude: '',
  manualExclude: '',
})

const poolEditing = ref(false)
const metricOptions = [
  { value: 'volatility_1h', label: '1h 对数收益标准差（小数）' },
  { value: 'quote_volume_24h', label: '24h 成交额（USDT，缺失排除）' },
  { value: 'funding_rate', label: '资金费率（小数，0.0001 = 0.01%）' },
  { value: 'oi_value', label: '未平仓名义价值（USD）' },
  { value: 'long_short_ratio', label: '全市场账户多空比' },
  { value: 'top_long_short', label: '大户多空比' },
  { value: 'taker_buy_sell', label: '主动买卖量比' },
]
const metricField = ref('funding_rate')
const metricMin = ref('')
const metricMax = ref('')
const volatilityWindow = ref(24)
const volatilityMaxAge = ref(2)
function addMetricFilter() {
  try {
    const rule = parseJson(poolForm.rule || '{}', '池规则')
    const bounds = {}
    for (const [key, raw] of [['min', metricMin.value], ['max', metricMax.value]]) {
      if (String(raw).trim() !== '') {
        const value = Number(raw)
        if (!Number.isFinite(value)) throw new Error('筛选边界必须是有限数字')
        bounds[key] = value
      }
    }
    if (!Object.keys(bounds).length) throw new Error('请至少填写一个筛选边界')
    rule.filters ||= {}
    rule.filters[metricField.value] = bounds
    if (metricField.value === 'volatility_1h') {
      rule.volatility_window_hours = volatilityWindow.value
      rule.volatility_max_age_hours = volatilityMaxAge.value
    }
    rule.max_age_hours ||= 24
    poolForm.rule = JSON.stringify(rule, null, 2)
  } catch (err) { ElMessage.error(err.message) }
}

const poolMembersVisible = ref(false)
const poolMembers = ref([])
const poolMembersTitle = ref('')
const poolMembersCode = ref('')
const poolMembersManual = ref(null)

const enabledCount = computed(() => enabledTotal.value)

function parseJson(text, label) {
  try {
    return JSON.parse(text)
  } catch (err) {
    throw new Error(`${label} 不是合法 JSON`)
  }
}

function paramsSummary(row) {
  const keys = Object.keys(row.params || {})
  if (!keys.length) return '继承指标默认'
  return keys.map((key) => `${key}=${row.params[key]}`).join(' · ')
}

function load(page = listPage.value) {
  if (loading.value) return Promise.resolve()
  loading.value = true
  error.value = ''
  return Promise.all([
    api.getBindings({ page, page_size: PAGE_SIZE }),
    api.getPools(),
    api.getBindingExitOptions(),
  ])
    .then(([bindingData, poolData, exitData]) => {
      bindings.value = bindingData.bindings || []
      listTotal.value = Number(bindingData.total || 0)
      enabledTotal.value = Number(bindingData.enabled_total ?? 0)
      listPage.value = Number(bindingData.page || page)
      // 删到空页时退回最后一页，避免停在空白表格
      const last = Math.max(1, Math.ceil(listTotal.value / PAGE_SIZE))
      if (listPage.value > last) return load(last)
      pools.value = poolData.pools || []
      exitOptions.value = { methods: exitData.methods || [], support_sources: exitData.support_sources || [] }
    })
    .catch((err) => {
      error.value = err.message
    })
    .finally(() => {
      loading.value = false
    })
}

function openCreate() {
  bindingEditing.value = false
  bindingForm.id = null
  bindingForm.strategy = props.strategies[0]?.code || 'absorption_reversal'
  bindingForm.pool = pools.value[0]?.code || ''
  bindingForm.interval = '15m'
  bindingForm.priority = 100
  bindingForm.enabled = false
  bindingForm.params = '{}'
  resetExitForm()
  bindingVisible.value = true
}

function openEditBinding(row) {
  bindingEditing.value = true
  bindingOriginal.value = { strategy: row.strategy, pool: row.pool }
  bindingForm.id = row.id
  bindingForm.strategy = row.strategy
  bindingForm.pool = row.pool
  bindingForm.interval = row.interval
  bindingForm.priority = row.priority
  bindingForm.enabled = row.enabled
  bindingForm.params = JSON.stringify(row.params || {}, null, 2)
  // 已存的方法参数优先，缺的键用注册表默认值补齐（后端也会补，这里只为界面完整）
  resetExitForm(row.exit_method || 'strategy', row.support_indicator || '', row.exit_params || {})
  bindingVisible.value = true
}

async function submitBinding() {
  let params
  try {
    params = parseJson(bindingForm.params, '参数')
  } catch (err) {
    ElMessage.error(err.message)
    return
  }
  const payload = {
    strategy: bindingForm.strategy,
    pool: bindingForm.pool,
    interval: bindingForm.interval,
    priority: Number(bindingForm.priority),
    enabled: bindingForm.enabled,
    params,
    exit_method: exitForm.method,
    support_indicator: exitForm.support || '',
    // 只提交当前方法声明过的键，避免把别的方法的参数带进来
    exit_params: Object.fromEntries(
      exitParamFields.value.map((field) => [field.key, Number(exitForm.params[field.key] ?? field.fallback)]),
    ),
  }
  if (activeExitMethod.value?.needs_support && !payload.support_indicator) {
    ElMessage.error('支撑位止损必须先选择支撑线指标')
    return
  }
  // 改身份字段会影响历史订单的参数归属，先让用户确认一次
  const scopeChanged = payload.strategy !== bindingOriginal.value.strategy || payload.pool !== bindingOriginal.value.pool
  if (bindingEditing.value && scopeChanged) {
    try {
      await ElMessageBox.confirm(
        '修改指标或币种池后，历史订单/信号仍记录原来的信息，其参数可能回落到指标默认；已开仓的止损止盈不受影响。确认修改？',
        '变更绑定范围',
        { type: 'warning' },
      )    } catch (err) {
      return
    }
  }
  bindingSaving.value = true
  try {
    if (bindingEditing.value) {
      await api.updateBinding(bindingForm.id, payload)
      ElMessage.success('绑定已更新')
    } else {
      await api.createBinding(payload)
      const strategy = props.strategies.find((item) => item.code === bindingForm.strategy)
      if (!strategy?.enabled) ElMessage.warning('策略已创建；指标总开关未启用，巡检不会执行该策略')
      else ElMessage.success('绑定已创建')
    }
    bindingVisible.value = false
    await load()
    emit('change')
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    bindingSaving.value = false
  }
}

async function toggle(row) {
  saving[row.id] = true
  try {
    await api.updateBinding(row.id, { enabled: row.enabled })
    ElMessage.success(`${row.strategy} × ${row.pool} 已${row.enabled ? '启用' : '停用'}`)
    emit('change')
  } catch (err) {
    row.enabled = !row.enabled
    ElMessage.error(err.message)
  } finally {
    saving[row.id] = false
  }
}

function openParams(row) {
  paramsTarget.value = row
  paramsText.value = JSON.stringify(row.params || {}, null, 2)
  paramsVisible.value = true
}

async function saveParams() {
  let params
  try {
    params = parseJson(paramsText.value, '参数')
  } catch (err) {
    ElMessage.error(err.message)
    return
  }
  paramsSaving.value = true
  try {
    await api.updateBinding(paramsTarget.value.id, { params })
    ElMessage.success('绑定参数已保存')
    paramsVisible.value = false
    await load()
    emit('change')
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    paramsSaving.value = false
  }
}

async function removeBinding(row) {
  try {
    await ElMessageBox.confirm(
      `删除策略「${row.strategy} × ${row.pool} × ${row.interval}」？该策略的参数配置会一并丢失。`,
      '删除策略',
      { type: 'warning' },
    )
  } catch (err) {
    return
  }
  try {
    await api.deleteBinding(row.id)
    ElMessage.success('绑定已删除')
    await load()
    emit('change')
  } catch (err) {
    ElMessage.error(err.message)
  }
}

function openCreatePool() {
  poolEditing.value = false
  poolForm.code = ''
  poolForm.name = ''
  poolForm.kind = 'rule'
  poolForm.symbols = ''
  poolForm.rule = '{"base": "all"}'
  poolForm.manualInclude = ''
  poolForm.manualExclude = ''
  volatilityWindow.value = 24
  volatilityMaxAge.value = 2
  poolVisible.value = true
}

function openEditPool(pool) {
  if (!pool) return
  poolEditing.value = true
  poolForm.code = pool.code
  poolForm.name = pool.name
  poolForm.kind = pool.kind
  poolForm.symbols = (pool.symbols || []).join(', ')
  poolForm.rule = JSON.stringify(pool.rule || {}, null, 2)
  poolForm.manualInclude = (pool.manual_include || []).join(', ')
  poolForm.manualExclude = (pool.manual_exclude || []).join(', ')
  volatilityWindow.value = pool.rule?.volatility_window_hours ?? 24
  volatilityMaxAge.value = pool.rule?.volatility_max_age_hours ?? 2
  poolVisible.value = true
}

async function submitPool() {
  let rule
  try {
    rule = parseJson(poolForm.rule || '{}', '池规则')
  } catch (err) {
    ElMessage.error(err.message)
    return
  }
  const symbols = poolForm.symbols.split(/[\s,;]+/).filter(Boolean)
  const manualInclude = poolForm.manualInclude.split(/[\s,;]+/).filter(Boolean)
  const manualExclude = poolForm.manualExclude.split(/[\s,;]+/).filter(Boolean)
  poolSaving.value = true
  try {
    const payload = {
      code: poolForm.code,
      name: poolForm.name || poolForm.code,
      kind: poolForm.kind,
      symbols,
      rule,
      manual_include: manualInclude,
      manual_exclude: manualExclude,
    }
    if (poolEditing.value) {
      delete payload.code
      await api.updatePool(poolForm.code, payload)
    } else {
      await api.createPool({ ...payload, enabled: true })
    }
    ElMessage.success('币种池已保存')
    poolVisible.value = false
    await load()
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    poolSaving.value = false
  }
}

async function showMembers(pool) {
  try {
    const data = await api.getPoolMembers(pool.code)
    poolMembers.value = data.symbols || []
    poolMembersTitle.value = `${pool.name || pool.code}（${data.count} 个）`
    poolMembersCode.value = pool.code
    poolMembersManual.value = {
      include: data.manual_include || [],
      exclude: data.manual_exclude || [],
      added: data.manual_added || [],
    }
    poolMembersVisible.value = true
  } catch (err) {
    ElMessage.error(err.message)
  }
}

function editMembersPool() {
  const pool = pools.value.find((item) => item.code === poolMembersCode.value)
  poolMembersVisible.value = false
  openEditPool(pool)
}

async function removePool(pool) {
  try {
    await ElMessageBox.confirm(`删除币种池「${pool.name}」？`, '删除池', { type: 'warning' })
  } catch (err) {
    return
  }
  try {
    await api.deletePool(pool.code)
    ElMessage.success('币种池已删除')
    await load()
  } catch (err) {
    ElMessage.error(err.message)
  }
}

onMounted(load)
</script>

<template>
  <div class="panel" style="margin-bottom: 14px">
    <div class="panel-head">
      <div>
        <div class="panel-title">策略列表</div>
        <div class="panel-sub">
          每条策略由一个信号指标、币种池和周期构成，可配置参数与出场方式；暂不支持多个信号指标自由组合。
          同一指标、币种池和周期只能有一条策略。参数依次继承全局设置、指标默认、策略覆盖及币种覆盖。
          已启用 {{ enabledCount }} 条；实际执行还取决于指标开关、币种池状态和交易范围。
        </div>
      </div>
      <div class="row">
        <el-button size="small" :loading="loading" @click="load">刷新</el-button>
        <el-button size="small" @click="openCreatePool">新建币种池</el-button>
        <el-button size="small" type="primary" :disabled="!pools.length" @click="openCreate">新增策略</el-button>
      </div>
    </div>

    <div v-if="error" class="row" role="alert" style="margin-bottom: 12px">
      <span class="down">绑定加载失败：{{ error }}</span>
      <el-button size="small" @click="load">重试</el-button>
    </div>

    <div v-if="!bindings.length && !loading" class="empty-hint">
      暂无策略。请选择一个信号指标、币种池和周期创建策略。
    </div>

    <el-table v-else :data="bindings" class="table-compact section-gap">
      <el-table-column label="信号指标" min-width="150">
        <template #default="{ row }">
          <div>{{ row.strategy_name || row.strategy }}</div>
          <div class="muted mono">{{ row.strategy }}</div>
        </template>
      </el-table-column>
      <el-table-column label="币种池" min-width="150">
        <template #default="{ row }">
          <el-button link size="small" @click="showMembers({ code: row.pool, name: row.pool_name })">
            {{ row.pool_name || row.pool }}
          </el-button>
          <div class="muted">{{ row.member_count ?? 0 }} 个币种</div>
        </template>
      </el-table-column>
      <el-table-column prop="interval" label="周期" width="80" />
      <el-table-column label="出场" min-width="180">
        <template #default="{ row }">
          <div>{{ exitSummary(row) }}</div>
          <el-tag v-if="exitIsUnverified(row)" size="small" type="warning" effect="plain">未经统计支持</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="启用" width="80">
        <template #default="{ row }">
          <el-switch v-model="row.enabled" :loading="saving[row.id]" @change="toggle(row)" />
        </template>
      </el-table-column>
      <el-table-column prop="priority" label="优先级" width="90" />
      <el-table-column label="参数覆盖" min-width="260">
        <template #default="{ row }">
          <span class="muted mono">{{ paramsSummary(row) }}</span>
          <div v-if="row.error" class="down">{{ row.error }}</div>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200">
        <template #default="{ row }">
          <el-button size="small" link type="primary" @click="openEditBinding(row)">编辑</el-button>
          <el-button size="small" link @click="openParams(row)">参数</el-button>
          <el-button size="small" link type="danger" @click="removeBinding(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <TablePager
      :page="listPage"
      :page-size="PAGE_SIZE"
      :total="listTotal"
      :disabled="loading"
      @change="load"
    />

    <div class="row" style="margin-top: 14px">
      <span class="muted">币种池</span>
      <el-tag
        v-for="pool in pools"
        :key="pool.code"
        size="small"
        effect="plain"
        closable
        @close="removePool(pool)"
        @click="showMembers(pool)"
        style="cursor: pointer"
      >
        {{ pool.name }} · {{ pool.member_count }} 币
      </el-tag>
      <el-select placeholder="编辑币种池" style="width: 170px" :model-value="''" @change="code => openEditPool(pools.find(pool => pool.code === code))">
        <el-option v-for="pool in pools" :key="pool.code" :label="pool.name" :value="pool.code" />
      </el-select>
      <span v-if="!pools.length" class="muted">暂无币种池</span>
    </div>
    <div class="form-hint">点池名看当前成员，可在成员弹窗里直接编辑该池；已绑定的池改成员无需重建绑定，下一轮巡检立即生效。</div>

    <el-dialog v-model="bindingVisible" :title="bindingEditing ? '编辑策略' : '新增策略'" width="520px">
      <div class="form-hint" style="margin-bottom: 10px">
        每条策略选择一个信号指标、一个币种池和一个周期；不支持多个信号指标自由组合。同一组合只能创建一次。
        <span v-if="bindingEditing">更换指标时请清理旧指标专属参数，保存时会重新校验。</span>
      </div>
      <div class="form-grid">
        <div>
          <div class="muted" style="margin-bottom: 4px">信号指标</div>
          <el-select v-model="bindingForm.strategy" style="width: 100%">
            <el-option v-for="item in strategies" :key="item.code" :label="item.name" :value="item.code" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">币种池</div>
          <el-select v-model="bindingForm.pool" style="width: 100%">
            <el-option v-for="pool in pools" :key="pool.code" :label="`${pool.name}（${pool.member_count}）`" :value="pool.code" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">K线周期</div>
          <el-select v-model="bindingForm.interval" style="width: 100%">
            <el-option v-for="item in intervalOptions" :key="item" :label="item" :value="item" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">优先级（仅排序）</div>
          <el-input-number v-model="bindingForm.priority" :min="0" :max="10000" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">启用</div>
          <el-switch v-model="bindingForm.enabled" />
        </div>
      </div>
      <div class="muted" style="margin: 12px 0 4px">参数覆盖（JSON，填写 {} 继承指标默认）</div>
      <el-input v-model="bindingForm.params" type="textarea" :rows="7" class="mono" />

      <el-divider style="margin: 16px 0 12px" />
      <div class="muted" style="margin-bottom: 8px">
        出场方法（可选项由后端校验；具体指标及执行方式可能存在限制）
      </div>
      <div class="form-grid">
        <div>
          <div class="muted" style="margin-bottom: 4px">方法</div>
          <el-select v-model="exitForm.method" style="width: 100%" @change="onExitMethodChange">
            <el-option v-for="item in exitMethodOptions" :key="item.code" :label="item.label" :value="item.code" />
          </el-select>
        </div>
        <div v-if="activeExitMethod?.needs_support">
          <div class="muted" style="margin-bottom: 4px">支撑线指标</div>
          <el-select v-model="exitForm.support" filterable style="width: 100%" placeholder="选择支撑位来源">
            <el-option v-for="item in exitSourceOptions" :key="item.code" :label="item.label" :value="item.code" />
          </el-select>
        </div>
      </div>
      <div v-if="exitParamFields.length" class="form-grid" style="margin-top: 10px">
        <div v-for="field in exitParamFields" :key="field.key">
          <div class="muted" style="margin-bottom: 4px">
            {{ field.key }}（{{ field.range[0] }}–{{ field.range[1] }}）
          </div>
          <el-input-number
            v-model="exitForm.params[field.key]"
            :min="field.range[0]"
            :max="field.range[1]"
            :step="0.1"
            style="width: 100%"
          />
        </div>
      </div>
      <div v-if="activeExitMethod" class="form-hint">{{ activeExitMethod.description }}</div>
      <el-alert
        v-if="activeExitMethod && activeExitMethod.needs_support"
        title="支撑位量具的评估 verdict=inconclusive（不优于随机价位）；启用前须按手册出 A/B 报告，报告未过 gate 不得当作已验证配置。"
        type="warning"
        :closable="false"
        style="margin-top: 8px"
      />
      <template #footer>
        <el-button @click="bindingVisible = false">取消</el-button>
        <el-button type="primary" :loading="bindingSaving" @click="submitBinding">{{ bindingEditing ? '保存' : '创建' }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="paramsVisible" :title="`绑定参数 · ${paramsTarget?.strategy} × ${paramsTarget?.pool}`" width="560px">
      <div class="muted" style="margin-bottom: 8px">
        只写需要覆盖的字段；同名键会覆盖策略默认，删除某个键即恢复继承。
        保存时会与当前全局设置合并后完整校验，非法值直接拒绝。
      </div>
      <el-input v-model="paramsText" type="textarea" :rows="10" class="mono" />
      <template #footer>
        <el-button @click="paramsVisible = false">取消</el-button>
        <el-button type="primary" :loading="paramsSaving" @click="saveParams">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="poolVisible" :title="poolEditing ? '编辑币种池' : '新建币种池'" width="620px">
      <div class="form-grid">
        <div>
          <div class="muted" style="margin-bottom: 4px">代码（小写字母/数字/下划线）</div>
          <el-input v-model="poolForm.code" :disabled="poolEditing" placeholder="e.g. alt_top100" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">名称</div>
          <el-input v-model="poolForm.name" placeholder="e.g. 山寨前100" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">类型</div>
          <el-select v-model="poolForm.kind" style="width: 100%">
            <el-option label="规则（动态）" value="rule" />
            <el-option label="静态列表" value="static" />
          </el-select>
        </div>
      </div>
      <div class="muted" style="margin: 12px 0 4px">静态成员（逗号/空格分隔，仅 static 用）</div>
      <el-input v-model="poolForm.symbols" type="textarea" :rows="3" class="mono" placeholder="BTCUSDT, ETHUSDT" />
      <div class="muted" style="margin: 12px 0 4px">
        规则（JSON）：base=all/major/static，可加 min_quote_volume_usd、exclude_major、max_symbols、exclude
      </div>
      <el-input v-model="poolForm.rule" type="textarea" :rows="6" class="mono" />
      <div class="muted" style="margin: 12px 0 4px">手工加入（逗号/空格分隔，对 rule 池同样有效）</div>
      <el-input v-model="poolForm.manualInclude" type="textarea" :rows="2" class="mono" placeholder="SOLUSDT, DOGEUSDT" />
      <div class="muted" style="margin: 12px 0 4px">手工剔除（逗号/空格分隔）</div>
      <el-input v-model="poolForm.manualExclude" type="textarea" :rows="2" class="mono" placeholder="BTCUSDT" />
      <div class="form-hint">手工增删在规则筛选、指标条件与 TopN 截断之后叠加，不会被这些条件刷掉，也不会被研究池的自动维护覆盖；仍受全局黑名单与自定义交易范围约束。清空输入框即可撤销。同币同时写入两侧时以剔除为准。</div>
      <div v-if="poolEditing" class="form-hint">保存后，引用此池的已有绑定会使用更新后的成员范围。</div>
      <div class="form-hint">条件取交集。资金费率与情绪默认使用 24 小时内的数据库记录；严格成交额使用交易所 24h 数据（最多缓存 1 小时），缺失或为零时排除，旧成交额门槛仍保留旧行为。</div>
      <div class="form-hint">1h 波动率为最近 N 个小时对数收益的总体标准差，需 N+1 根连续已闭合 K 线，最新闭合时间距今不得超过指定小时数；不足、过期或缺口均排除。它是当前滚动值，不是训练期长期排名，也不适合直接作为历史时点分类。</div>
      <div v-if="metricField === 'volatility_1h'" class="row" style="margin-top: 10px">
        <span>收益窗口（小时）</span><el-input-number v-model="volatilityWindow" :min="2" :max="720" />
        <span>最大数据年龄（小时）</span><el-input-number v-model="volatilityMaxAge" :min="0.1" :step="0.5" />
      </div>
      <div class="row" style="margin-top: 10px; flex-wrap: wrap">
        <el-select v-model="metricField" style="width: 230px"><el-option v-for="item in metricOptions" :key="item.value" :label="item.label" :value="item.value" /></el-select>
        <el-input v-model="metricMin" placeholder="最小值（可空）" style="width: 140px" />
        <el-input v-model="metricMax" placeholder="最大值（可空）" style="width: 140px" />
        <el-button @click="addMetricFilter">添加或替换条件</el-button>
      </div>
      <template #footer>
        <el-button @click="poolVisible = false">取消</el-button>
        <el-button type="primary" :loading="poolSaving" :disabled="!poolForm.code" @click="submitPool">{{ poolEditing ? '保存' : '创建' }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="poolMembersVisible" :title="`成员 · ${poolMembersTitle}`" width="520px">
      <div v-if="poolMembersManual && (poolMembersManual.include.length || poolMembersManual.exclude.length)" class="form-hint">
        手工加入 {{ poolMembersManual.include.length }} 个（实际生效 {{ poolMembersManual.added.length }} 个，被全局范围或黑名单拦下的不计入）；手工剔除 {{ poolMembersManual.exclude.length }} 个。
      </div>
      <div class="mono" style="line-height: 1.8; word-break: break-all">
        {{ poolMembers.join(', ') || '（空池）' }}
      </div>
      <template #footer>
        <el-button @click="poolMembersVisible = false">关闭</el-button>
        <el-button type="primary" @click="editMembersPool">编辑该池</el-button>
      </template>
    </el-dialog>
  </div>
</template>
