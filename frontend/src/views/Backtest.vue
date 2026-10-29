<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import StatCard from '../components/StatCard.vue'
import { closeReasonLabel, money, number, pct, pnlClass, price, tsLabel } from '../utils/format'
import { signalReason } from '../utils/strategy'

const settings = ref({})
const history = ref([])
const activeRun = ref(null)
const result = ref(null)
const running = ref(false)
const loadingHistory = ref(false)
const settingsLoading = ref(false)
const settingsReady = ref(false)
const settingsError = ref('')
const historyError = ref('')
const loadingRun = ref(false)
const resultError = ref('')
const historyPagination = reactive({ page: 1, pageSize: 20, total: 0 })
const tradeRows = ref([])
const tradePagination = reactive({ page: 1, pageSize: 20, total: 0 })
const loadingTrades = ref(false)
const tradesError = ref('')
let pollTimer = null
let disposed = false
let runRequest = 0
let historyRequest = 0
let tradeRequest = 0
let selectedTradeRunId = null
const touchedFields = new Set()

const bindings = ref([])
const strategyOptions = ref([])
const selectedBinding = ref('')

const form = reactive({
  symbol: 'BTCUSDT',
  interval: '1m',
  strategy: 'absorption_reversal',
  pool: '',
  days: 30,
  initial_cash: 10000,
  fee_pct: 0.04,
  slippage_pct: 0.02,
  threshold: null,
  optimize: true,
})

const strategyChoices = computed(() => (
  strategyOptions.value.length ? strategyOptions.value : [{ code: 'absorption_reversal', name: '吸收反转' }]
))
const bindingOptions = computed(() => bindings.value.filter((row) => row.strategy === form.strategy))
const isZigzag = computed(() => form.strategy === 'simple_zigzag_double')
let zigzagSelected = false
let previousOptimize = true

function onStrategyChange() {
  // 换策略后原来选中的绑定不再适用，清空回落到策略默认参数层
  selectedBinding.value = ''
  form.pool = ''
  if (isZigzag.value && !zigzagSelected) {
    zigzagSelected = true
    previousOptimize = form.optimize
    form.optimize = false
    if (!touchedFields.has('initial_cash')) form.initial_cash = 1000000
    if (!touchedFields.has('fee_pct')) form.fee_pct = 0.1
    if (!touchedFields.has('slippage_pct')) form.slippage_pct = 0
  } else if (!isZigzag.value && zigzagSelected) {
    zigzagSelected = false
    form.optimize = previousOptimize
    if (!touchedFields.has('initial_cash')) form.initial_cash = 10000
    if (!touchedFields.has('fee_pct')) form.fee_pct = Number(((settings.value.fee_rate ?? 0.0004) * 100).toFixed(4))
    if (!touchedFields.has('slippage_pct')) form.slippage_pct = Number(((settings.value.slippage ?? 0.0002) * 100).toFixed(4))
  }
}

function applyBinding(bindingId) {
  const row = bindings.value.find((item) => String(item.id) === String(bindingId))
  if (!row) {
    form.pool = ''
    return
  }
  // 绑定的参数层要求 (策略, 池, 周期) 三元组精确命中，周期必须跟着绑定走
  form.pool = row.pool
  form.interval = row.interval
  touchField('interval')
}

function onIntervalChange() {
  touchField('interval')
  // 绑定的参数层要求 (策略, 池, 周期) 三元组精确命中，手动换周期后必须检查，
  // 否则会带着一个不存在的池去回测，参数层被静默忽略。
  if (!form.pool) return
  const hit = bindings.value.some((row) => (
    row.strategy === form.strategy && row.pool === form.pool && row.interval === form.interval
  ))
  if (hit) return
  form.pool = ''
  selectedBinding.value = ''
}

const metrics = computed(() => result.value?.metrics || {})
const equityCurve = computed(() => result.value?.equity_curve || [])
const trades = computed(() => tradeRows.value)
const comparison = computed(() => metrics.value.baseline ? [
  { name: '基础信号（相同风控与成本）', ...metrics.value.baseline },
  { name: '优化策略（样本外）', ...metrics.value },
] : [])
const blockedSignals = computed(() => Object.entries(metrics.value.blocked_signals || {}).map(([reason, count]) => ({ reason: signalReason(reason), count })))
function profitFactor(value) {
  return value.profit_factor_infinite ? '∞' : value.profit_factor == null ? '—' : number(value.profit_factor)
}

const equityOption = computed(() => {
  const points = equityCurve.value
  return {
    backgroundColor: 'transparent',
    grid: { left: 58, right: 18, top: 24, bottom: 46 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15,20,28,0.95)',
      borderColor: 'rgba(148,163,184,0.2)',
      textStyle: { color: '#e6ecf5', fontSize: 11 },
      formatter: (params) => `${params[0].axisValue}<br/>权益 ${money(params[0].value)}`,
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: points.map((item) => shortTime(item.time)),
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.2)' } },
      axisLabel: { color: '#64748b', fontSize: 10, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } },
      axisLabel: { color: '#64748b', fontSize: 10, formatter: (value) => `$${Math.round(value)}` },
    },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 8, borderColor: 'transparent', backgroundColor: 'rgba(148,163,184,0.08)', fillerColor: 'rgba(59,130,246,0.18)', handleStyle: { color: '#60a5fa' }, textStyle: { color: '#64748b' } }],
    series: [
      {
        type: 'line',
        data: points.map((item) => item.equity),
        showSymbol: false,
        lineStyle: { width: 2, color: '#60a5fa' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(96,165,250,0.24)' },
              { offset: 1, color: 'rgba(96,165,250,0)' },
            ],
          },
        },
      },
    ],
  }
})

function shortTime(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value).slice(5, 16)
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false })
}

function touchField(field) {
  touchedFields.add(field)
}

async function loadSettings() {
  if (disposed) return
  settingsLoading.value = true
  settingsError.value = ''
  try {
    const config = await api.getSettings()
    if (disposed) return
    settings.value = config
    if (!touchedFields.has('symbol')) form.symbol = config.symbols?.[0] || 'BTCUSDT'
    if (!touchedFields.has('interval')) form.interval = config.interval || '1m'
    if (!touchedFields.has('fee_pct')) form.fee_pct = isZigzag.value ? 0.1 : Number(((config.fee_rate ?? 0.0004) * 100).toFixed(4))
    if (!touchedFields.has('slippage_pct')) form.slippage_pct = isZigzag.value ? 0 : Number(((config.slippage ?? 0.0002) * 100).toFixed(4))
    settingsReady.value = true
  } catch (error) {
    if (!disposed) settingsError.value = error.message
  } finally {
    if (!disposed) settingsLoading.value = false
  }
}

async function loadBindings() {
  if (disposed) return
  try {
    const [bindingData, strategyData] = await Promise.all([api.getBindings(), api.getStrategies()])
    if (disposed) return
    bindings.value = bindingData.bindings || []
    strategyOptions.value = (strategyData.specs || []).map((spec) => ({ code: spec.code, name: spec.name }))
  } catch {
    // 绑定/策略列表加载失败不阻塞回测，退回「策略默认参数」这一层
    if (!disposed) {
      bindings.value = []
      strategyOptions.value = []
    }
  }
}

function loadBase() {
  return Promise.allSettled([loadSettings(), loadHistory(), loadBindings()])
}

async function loadHistory() {
  if (disposed) return
  const request = ++historyRequest
  loadingHistory.value = true
  historyError.value = ''
  const page = historyPagination.page
  const pageSize = historyPagination.pageSize
  try {
    const data = await api.listBacktests({ page, page_size: pageSize })
    if (disposed || request !== historyRequest) return
    const total = Number(data.total ?? data.backtests?.length ?? 0)
    historyPagination.total = total
    const lastPage = Math.max(1, Math.ceil(total / pageSize))
    if (page > lastPage) {
      historyPagination.page = lastPage
      return await loadHistory()
    }
    history.value = data.backtests || []
  } catch (error) {
    if (!disposed && request === historyRequest) historyError.value = error.message
  } finally {
    if (!disposed && request === historyRequest) loadingHistory.value = false
  }
}

function changeHistoryPage(page) {
  if (page === historyPagination.page) return
  historyPagination.page = page
  return loadHistory()
}

function changeHistorySize(pageSize) {
  if (pageSize === historyPagination.pageSize) return
  historyPagination.pageSize = pageSize
  historyPagination.page = 1
  return loadHistory()
}

function resetTrades(runId = null) {
  tradeRequest += 1
  selectedTradeRunId = runId
  tradeRows.value = []
  tradePagination.page = 1
  tradePagination.total = 0
  loadingTrades.value = false
  tradesError.value = ''
}

async function loadTrades() {
  const runId = selectedTradeRunId
  if (disposed || runId == null || result.value?.id !== runId || result.value?.status !== 'done') return
  const request = ++tradeRequest
  const page = tradePagination.page
  const pageSize = tradePagination.pageSize
  loadingTrades.value = true
  tradesError.value = ''
  try {
    const data = await api.getBacktestTrades(runId, { page, page_size: pageSize })
    if (disposed || request !== tradeRequest || selectedTradeRunId !== runId) return
    const total = Number(data.total ?? data.trades?.length ?? 0)
    tradePagination.total = total
    const lastPage = Math.max(1, Math.ceil(total / pageSize))
    if (page > lastPage) {
      tradePagination.page = lastPage
      return await loadTrades()
    }
    tradeRows.value = data.trades || []
  } catch (error) {
    if (!disposed && request === tradeRequest) tradesError.value = error.message || '成交明细加载失败'
  } finally {
    if (!disposed && request === tradeRequest) loadingTrades.value = false
  }
}

function changeTradePage(page) {
  if (page === tradePagination.page) return
  tradePagination.page = page
  return loadTrades()
}

function changeTradeSize(pageSize) {
  if (pageSize === tradePagination.pageSize) return
  tradePagination.pageSize = pageSize
  tradePagination.page = 1
  return loadTrades()
}

async function poll(runId, request = runRequest) {
  if (disposed || request !== runRequest) return
  try {
    const data = await api.getBacktest(runId, { include_trades: false })
    if (disposed || request !== runRequest) return
    activeRun.value = data
    if (data.status === 'done') {
      result.value = data
      running.value = false
      resetTrades(runId)
      loadTrades()
      ElMessage.success('回测完成')
      loadHistory()
      return
    }
    if (data.status === 'failed') {
      running.value = false
      ElMessage.error(data.error || '回测失败')
      loadHistory()
      return
    }
    pollTimer = setTimeout(() => poll(runId, request), 1800)
  } catch (error) {
    if (disposed || request !== runRequest) return
    running.value = false
    resultError.value = error.message
    ElMessage.error(error.message)
  }
}

async function submit() {
  if (disposed || !settingsReady.value || running.value) return
  const request = ++runRequest
  if (pollTimer) clearTimeout(pollTimer)
  loadingRun.value = false
  running.value = true
  result.value = null
  resultError.value = ''
  resetTrades()
  try {
    const payload = {
      symbol: form.symbol,
      interval: form.interval,
      strategy: form.strategy,
      pool: form.pool || null,
      days: Number(form.days),
      initial_cash: Number(form.initial_cash),
      fee: Number(form.fee_pct) / 100,
      slippage: Number(form.slippage_pct) / 100,
      threshold: isZigzag.value || form.threshold == null ? null : Number(form.threshold),
      optimize: isZigzag.value ? false : form.optimize,
    }
    const created = await api.createBacktest(payload)
    if (disposed || request !== runRequest) return
    activeRun.value = { id: created.id, status: 'running' }
    poll(created.id, request)
  } catch (error) {
    if (disposed || request !== runRequest) return
    running.value = false
    ElMessage.error(error.message)
  }
}

async function loadRun(row) {
  if (disposed) return
  const request = ++runRequest
  if (pollTimer) clearTimeout(pollTimer)
  loadingRun.value = true
  running.value = false
  resultError.value = ''
  resetTrades(row.id)
  try {
    const data = await api.getBacktest(row.id, { include_trades: false })
    if (disposed || request !== runRequest) return
    result.value = data
    activeRun.value = data
    if (data.status === 'done') loadTrades()
    window.scrollTo({ top: 0, behavior: 'smooth' })
  } catch (error) {
    if (!disposed && request === runRequest) resultError.value = error.message
  } finally {
    if (!disposed && request === runRequest) loadingRun.value = false
  }
}

onMounted(loadBase)
onBeforeUnmount(() => {
  disposed = true
  runRequest += 1
  historyRequest += 1
  tradeRequest += 1
  if (pollTimer) clearTimeout(pollTimer)
})
</script>

<template>
  <div>
    <div class="panel" style="margin-bottom: 14px">
      <div class="panel-head">
        <div>
          <div class="panel-title">策略验证与收益对比</div>
          <div class="panel-sub">{{ isZigzag ? '收盘更新止损挂单，按后续 K 线价格路径模拟触发；使用所选完整数据窗口。' : '收盘信号于下一根开盘撮合，计入双边成本、杠杆与风险限额；独立测试集评估' }}</div>
        </div>
        <el-tag size="small" effect="plain" type="info">{{ isZigzag ? 'Pine 止损挂单撮合' : '统一顺序撮合' }}</el-tag>
      </div>

      <div v-if="settingsLoading" class="muted" role="status">正在加载回测设置，回测历史可独立查看…</div>
      <div v-else-if="settingsError" class="form-hint" role="alert">
        回测设置加载失败：{{ settingsError }}
        <el-button size="small" text @click="loadSettings">重试</el-button>
      </div>
      <div class="form-grid">
        <div>
          <div class="muted" style="margin-bottom: 4px">币种</div>
          <el-select v-model="form.symbol" style="width: 100%" @change="touchField('symbol')">
            <el-option v-for="symbol in settings.symbols || ['BTCUSDT', 'ETHUSDT']" :key="symbol" :label="symbol" :value="symbol" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">K线周期</div>
          <el-select v-model="form.interval" style="width: 100%" @change="onIntervalChange">
            <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">策略</div>
          <el-select v-model="form.strategy" style="width: 100%" @change="onStrategyChange">
            <el-option v-for="item in strategyChoices" :key="item.code" :label="item.name" :value="item.code" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">参数层（策略 × 池绑定）</div>
          <el-select
            v-model="selectedBinding"
            clearable
            placeholder="策略默认参数"
            style="width: 100%"
            @change="applyBinding"
          >
            <el-option
              v-for="item in bindingOptions"
              :key="item.id"
              :label="`${item.pool_name || item.pool} × ${item.interval}${item.enabled ? '' : '（巡检已停用）'}`"
              :value="item.id"
            />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">回测天数</div>
          <el-input-number v-model="form.days" :min="1" :max="180" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">初始资金</div>
          <el-input-number v-model="form.initial_cash" :min="1000" :step="1000" style="width: 100%" @change="touchField('initial_cash')" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">手续费（单边 %）</div>
          <el-input-number v-model="form.fee_pct" :min="0" :max="1" :step="0.01" :precision="4" style="width: 100%" @change="touchField('fee_pct')" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">滑点（单边 %）</div>
          <el-input-number v-model="form.slippage_pct" :min="0" :max="1" :step="0.01" :precision="4" style="width: 100%" @change="touchField('slippage_pct')" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">开仓评分阈值（留空使用策略配置）</div>
          <el-input-number v-model="form.threshold" :disabled="isZigzag" :min="0.5" :max="1" :step="0.01" :precision="2" placeholder="使用策略配置" style="width: 100%" />
        </div>
      </div>

      <div class="row" style="margin-top: 14px">
        <el-switch v-model="form.optimize" :disabled="isZigzag" active-text="验证集选参" />
        <el-button type="primary" :loading="running" :disabled="!settingsReady" @click="submit">开始回测</el-button>
        <span class="muted" v-if="running">回测在后台运行，通常需要数秒到一分钟</span>
        <span class="muted" v-else>{{ isZigzag ? '使用所选完整数据窗口；不进行参数优化。该策略目前仅支持回测。' : form.optimize ? '前 60% 训练、15% 验证选参、后 25% 测试；边界隔离预测跨度' : '前 75% 训练、后 25% 测试；边界隔离预测跨度' }}</span>
      </div>
    </div>

    <div v-if="loadingRun" class="panel section-gap muted" role="status">正在加载回测结果，参数与历史列表仍可查看…</div>
    <div v-if="resultError" class="panel section-gap" role="alert">回测结果加载失败：{{ resultError }}，可点击历史记录重试。</div>
    <template v-if="result">
      <div class="grid stats-grid backtest-stats">
        <StatCard label="总收益" :value="pct(metrics.total_return_pct)" :tone="pnlClass(metrics.total_return_pct)" :sub="`${money(metrics.initial_cash)} → ${money(metrics.final_equity)}`" />
        <StatCard label="最大回撤" :value="pct(metrics.max_drawdown_pct)" :tone="pnlClass(metrics.max_drawdown_pct)" :sub="'权益峰值回落'" />
        <StatCard label="夏普" :value="number(metrics.sharpe)" sub="按日收益年化" />
        <StatCard label="胜率" :value="`${metrics.win_rate ?? 0}%`" :sub="`${metrics.total_trades || 0} 笔交易`" />
        <StatCard label="盈利因子" :value="profitFactor(metrics)" sub="总盈利 / 总亏损" />
        <StatCard label="手续费" :value="money(metrics.total_fees)" :tone="pnlClass(-Number(metrics.total_fees || 0))" :sub="`滑点 ${(metrics.slippage * 100).toFixed(3)}% · 费率 ${(metrics.fee_rate * 100).toFixed(3)}%`" />
        <StatCard label="买入持有" :value="pct(metrics.buy_hold_return_pct)" :tone="pnlClass(metrics.buy_hold_return_pct)" sub="测试区间基准" />
        <StatCard label="回测引擎" :value="metrics.engine === 'sequential' ? '顺序撮合' : metrics.engine === 'pine_stop_orders' ? 'Pine 止损挂单' : metrics.engine" :sub="metrics.engine === 'pine_stop_orders' ? `${metrics.signal_bars} 根挂单请求 K 线` : `${metrics.signal_bars} 个信号K线`" />
      </div>

      <div v-if="metrics.warnings?.length" class="panel section-gap">
        <div v-for="warning in metrics.warnings" :key="warning" class="form-hint">{{ warning }}</div>
      </div>

      <div v-if="comparison.length" class="panel section-gap">
        <div class="panel-head">
          <div>
            <div class="panel-title">样本外对比</div>
            <div class="panel-sub">两组使用相同数据、撮合与风险限制；基础信号关闭行情和成本过滤。收益差 {{ number(metrics.return_improvement_pct) }} 个百分点</div>
          </div>
        </div>
        <el-table :data="comparison" class="table-compact">
          <el-table-column prop="name" label="方案" min-width="240" />
          <el-table-column label="净收益"><template #default="{ row }"><span :class="pnlClass(row.total_return_pct)">{{ pct(row.total_return_pct) }}</span></template></el-table-column>
          <el-table-column label="最大回撤"><template #default="{ row }">{{ pct(row.max_drawdown_pct) }}</template></el-table-column>
          <el-table-column prop="total_trades" label="成交数" />
          <el-table-column label="手续费"><template #default="{ row }">{{ money(row.total_fees) }}</template></el-table-column>
          <el-table-column label="盈利因子"><template #default="{ row }">{{ profitFactor(row) }}</template></el-table-column>
        </el-table>
        <div class="form-hint">测试区间：{{ metrics.split?.test_start }} → {{ metrics.split?.test_end }}；边界清除 {{ metrics.split?.purge_bars }} 根标签。</div>
        <el-collapse style="margin-top: 12px">
          <el-collapse-item title="参数选择与信号过滤" name="diagnostics">
            <template v-if="metrics.validation?.enabled">
              <p>验证区间：{{ metrics.validation.start }} → {{ metrics.validation.end }}。选择 {{ metrics.validation.selected }}；{{ metrics.validation.status === 'insufficient_trades' ? '成交不足，保留当前参数' : '按净收益与回撤评分选出' }}。结果仅用于本次回测。</p>
              <el-table :data="metrics.validation.candidates || []" size="small">
                <el-table-column prop="name" label="候选" />
                <el-table-column prop="total_return_pct" label="验证净收益 %" />
                <el-table-column prop="max_drawdown_pct" label="回撤 %" />
                <el-table-column prop="total_trades" label="成交数" />
                <el-table-column prop="objective" label="评分" />
                <el-table-column label="样本量"><template #default="{ row }">{{ row.eligible ? '足够' : '不足 10 笔' }}</template></el-table-column>
              </el-table>
            </template>
            <el-table :data="blockedSignals" size="small" style="margin-top: 12px">
              <el-table-column prop="reason" label="过滤原因" />
              <el-table-column prop="count" label="K线数" width="100" />
            </el-table>
            <pre class="mono" style="white-space: pre-wrap">{{ JSON.stringify(metrics.parameters, null, 2) }}</pre>
          </el-collapse-item>
        </el-collapse>
      </div>

      <div class="panel section-gap">
        <div class="panel-head">
          <div>
            <div class="panel-title">权益曲线</div>
            <div class="panel-sub">{{ metrics.start }} → {{ metrics.end }} · {{ metrics.bars }} 根K线 · 阈值 {{ metrics.threshold }}</div>
          </div>
          <el-tag size="small" effect="plain">{{ result.symbol }} · {{ result.interval }} · {{ result.strategy }}</el-tag>
        </div>
        <EChart v-if="equityCurve.length > 1" :option="equityOption" height="330px" />
        <div v-else class="empty-hint">无权益曲线</div>
      </div>

      <div class="panel section-gap">
        <div class="panel-head">
          <div>
            <div class="panel-title">成交明细</div>
            <div class="panel-sub">最多保留最近 500 条平仓记录；分批止盈会显示多条，交易统计按完整持仓计算。</div>
          </div>
          <el-button size="small" :loading="loadingTrades" @click="loadTrades">刷新</el-button>
        </div>
        <div v-if="tradesError" class="form-hint" role="alert">
          成交明细加载失败：{{ tradesError }}
          <el-button size="small" text @click="loadTrades">重试</el-button>
        </div>
        <el-table v-loading="loadingTrades" :data="trades" class="table-compact" max-height="420" :empty-text="loadingTrades ? '正在加载成交明细…' : tradesError ? '成交明细加载失败' : '无成交'">
          <el-table-column label="开仓" min-width="140"><template #default="{ row }">{{ tsLabel(row.entry_time) }}</template></el-table-column>
          <el-table-column label="平仓" min-width="140"><template #default="{ row }">{{ tsLabel(row.exit_time) }}</template></el-table-column>
          <el-table-column prop="side" label="方向" width="80" />
          <el-table-column label="开 / 平" min-width="130"><template #default="{ row }"><span class="mono">{{ price(row.entry_price) }} / {{ price(row.exit_price) }}</span></template></el-table-column>
          <el-table-column label="数量" width="110"><template #default="{ row }"><span class="mono">{{ number(row.size, 6) }}</span></template></el-table-column>
          <el-table-column label="盈亏" width="110"><template #default="{ row }"><span class="mono" :class="pnlClass(row.pnl)">{{ money(row.pnl) }}</span></template></el-table-column>
          <el-table-column label="收益率" width="90"><template #default="{ row }"><span class="mono" :class="pnlClass(row.return_pct)">{{ pct(row.return_pct) }}</span></template></el-table-column>
          <el-table-column label="手续费" width="90"><template #default="{ row }"><span class="mono">{{ money(row.fees) }}</span></template></el-table-column>
          <el-table-column label="退出" min-width="100"><template #default="{ row }">{{ closeReasonLabel(row.status) }}</template></el-table-column>
        </el-table>
        <el-pagination
          class="table-pagination"
          :current-page="tradePagination.page"
          :page-size="tradePagination.pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="tradePagination.total"
          layout="total, sizes, prev, pager, next, jumper"
          @current-change="changeTradePage"
          @size-change="changeTradeSize"
        />
      </div>
    </template>

    <div class="panel section-gap">
      <div class="panel-head">
        <div>
          <div class="panel-title">回测历史</div>
          <div class="panel-sub">点击查看详细曲线与成交</div>
        </div>
        <el-button size="small" :loading="loadingHistory" @click="loadHistory">刷新</el-button>
      </div>
      <div v-if="historyError" class="form-hint" role="alert">
        回测历史加载失败：{{ historyError }}
        <el-button size="small" text @click="loadHistory">重试</el-button>
      </div>
      <el-table :data="history" class="table-compact" :empty-text="loadingHistory ? '正在加载回测历史…' : '暂无回测记录'" @row-click="loadRun">
        <el-table-column prop="id" label="#" width="56" />
        <el-table-column prop="symbol" label="币种" width="96" />
        <el-table-column prop="interval" label="周期" width="70" />
        <el-table-column prop="strategy" label="策略" min-width="120" />
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <el-tag size="small" effect="plain" :type="row.status === 'done' ? 'success' : row.status === 'failed' ? 'danger' : 'warning'">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="收益 / 回撤" min-width="150">
          <template #default="{ row }">
            <span v-if="row.metrics?.total_return_pct !== undefined" class="mono" :class="pnlClass(row.metrics.total_return_pct)">
              {{ pct(row.metrics.total_return_pct) }} / {{ pct(row.metrics.max_drawdown_pct) }}
            </span>
            <span v-else class="muted">{{ row.error ? row.error.slice(0, 40) : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="交易数" width="80"><template #default="{ row }">{{ row.metrics?.total_trades ?? '—' }}</template></el-table-column>
        <el-table-column label="创建时间" min-width="150"><template #default="{ row }">{{ tsLabel(row.created_at) }}</template></el-table-column>
      </el-table>
      <el-pagination
        class="table-pagination"
        :current-page="historyPagination.page"
        :page-size="historyPagination.pageSize"
        :page-sizes="[10, 20, 50, 100]"
        :total="historyPagination.total"
        layout="total, sizes, prev, pager, next, jumper"
        @current-change="changeHistoryPage"
        @size-change="changeHistorySize"
      />
    </div>
  </div>
</template>

<style scoped>
.backtest-stats { grid-template-columns: repeat(4, minmax(0, 1fr)); }
@media (max-width: 900px) {
  .backtest-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
