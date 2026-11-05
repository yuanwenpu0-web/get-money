<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import { closeReasonLabel, money, pct, pnlClass, price, sideLabel, statusLabel, strategyLabel, tsLabel } from '../utils/format'

const orders = ref([])
const settings = ref({})
const loading = ref(false)
const ordersLoaded = ref(false)
const ordersError = ref('')
const settingsLoading = ref(false)
const settingsError = ref('')
const orderStrategies = ref([])
const strategiesLoading = ref(false)
const strategiesError = ref('')
const strategiesUnavailable = ref(false)
const filters = reactive({ status: 'OPEN', symbol: '', strategy: '' })
const pagination = reactive({ page: 1, pageSize: 20, total: 0 })
const orderSummary = ref(null)
let timer = null
let loadPromise = null
let settingsPromise = null
let strategiesPromise = null
let reloadRequested = false
let disposed = false

const chartVisible = ref(false)
const chartSymbol = ref('')
const chartInterval = ref('1h')
const chartBars = ref([])
const chartOrder = ref(null)
const chartLoading = ref(false)
const chartError = ref('')
let chartRequest = 0

// 订单弹窗里的均线：EA20 / EA60（指数移动平均，基于收盘价）。
// 前 period-1 根不给值（不画线），第 period 根用 SMA 播种，之后按 2/(n+1) 递推。
const EMA_LINES = [
  { name: 'EA20', period: 20, color: '#fbbf24' },
  { name: 'EA60', period: 60, color: '#c084fc' },
]

function emaSeries(values, period) {
  const empty = () => new Array(values.length).fill(null)
  if (period <= 0 || values.length < period) return empty()
  const alpha = 2 / (period + 1)
  const result = new Array(values.length).fill(null)
  let sum = 0
  let previous = null
  for (let index = 0; index < values.length; index += 1) {
    const raw = values[index]
    // 收盘价有缺口就整体不画，避免给出一条错误的均线
    const value = raw === null || raw === undefined || raw === '' ? NaN : Number(raw)
    if (!Number.isFinite(value)) return empty()
    if (index < period) sum += value
    if (index === period - 1) previous = sum / period
    else if (index >= period) previous = value * alpha + previous * (1 - alpha)
    if (previous !== null) result[index] = +previous.toFixed(8)
  }
  return result
}

const emaSeriesList = computed(() => {
  const closes = chartBars.value.map((bar) => bar.close)
  return EMA_LINES.map((line) => ({ ...line, data: emaSeries(closes, line.period) }))
})

const entryPoint = computed(() => {
  const order = chartOrder.value
  const match = /^(\d+)(m|h|d|w)$/.exec(chartInterval.value)
  const time = order?.created_at ? Date.parse(order.created_at) : NaN
  const entryPrice = Number(order?.entry_price)
  if (!match || !Number.isFinite(time) || !Number.isFinite(entryPrice) || entryPrice <= 0) return null
  const duration = Number(match[1]) * { m: 60000, h: 3600000, d: 86400000, w: 604800000 }[match[2]]
  const index = chartBars.value.findIndex((bar) => time >= Number(bar.time) && time < Number(bar.time) + duration)
  return index < 0 ? null : { index, price: entryPrice }
})

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  animation: false,
  tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
  legend: {
    top: 2,
    left: 65,
    itemWidth: 16,
    itemHeight: 8,
    itemGap: 16,
    icon: 'roundRect',
    textStyle: { color: '#cbd5e1', fontSize: 11, fontWeight: 600 },
    data: EMA_LINES.map((line) => line.name),
  },
  grid: [{ left: 65, right: 56, top: 32, height: '60%' }, { left: 65, right: 56, top: '76%', height: '12%' }],
  xAxis: [
    { type: 'category', data: chartBars.value.map((bar) => tsLabel(bar.time)), axisLabel: { show: false } },
    { type: 'category', gridIndex: 1, data: chartBars.value.map((bar) => tsLabel(bar.time)), axisLabel: { hideOverlap: true } },
  ],
  yAxis: [{ scale: true }, { scale: true, gridIndex: 1, splitNumber: 2 }],
  dataZoom: [{ type: 'inside', xAxisIndex: [0, 1], start: entryPoint.value && entryPoint.value.index < chartBars.value.length * 0.4 ? 0 : 40, end: 100 }, { type: 'slider', xAxisIndex: [0, 1], bottom: 0, height: 16 }],
  series: [
    { name: 'K线', type: 'candlestick', data: chartBars.value.map((bar) => [bar.open, bar.close, bar.low, bar.high]), itemStyle: { color: '#34d399', color0: '#fb7185', borderColor: '#34d399', borderColor0: '#fb7185' },
      markPoint: { symbol: 'pin', symbolSize: 48, itemStyle: { color: '#e6a23c' }, label: { formatter: chartOrder.value?.side === 'SHORT' ? '做空' : '买入', color: '#fff' }, data: entryPoint.value ? [{ name: '开仓', coord: [entryPoint.value.index, entryPoint.value.price], value: entryPoint.value.price }] : [] },
    },
    // 线尾直接标注名字，避免只靠顶部图例认线；两条线贴近时上下错开
    ...emaSeriesList.value.map((line) => ({
      name: line.name,
      type: 'line',
      data: line.data,
      showSymbol: false,
      smooth: false,
      connectNulls: false,
      lineStyle: { width: 1.2, color: line.color },
      itemStyle: { color: line.color },
      emphasis: { focus: 'series' },
      tooltip: { valueFormatter: (value) => (typeof value === 'number' ? price(value) : '—') },
      labelLayout: { moveOverlap: 'shiftY' },
      endLabel: {
        show: true,
        formatter: line.name,
        distance: 4,
        fontSize: 10,
        fontWeight: 'bold',
        color: '#0b1220',
        backgroundColor: line.color,
        padding: [2, 4],
        borderRadius: 3,
      },
    })),
    { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: chartBars.value.map((bar) => bar.volume), itemStyle: { color: 'rgba(96,165,250,0.45)' } },
  ],
}))

async function loadChartBars() {
  const request = ++chartRequest
  chartBars.value = []
  chartLoading.value = true
  chartError.value = ''
  try {
    const data = await api.getKlines(chartSymbol.value, chartInterval.value, 600)
    if (!disposed && request === chartRequest) chartBars.value = data.bars || []
  } catch (error) {
    if (!disposed && request === chartRequest) chartError.value = error.message || '加载失败'
  } finally {
    if (!disposed && request === chartRequest) chartLoading.value = false
  }
}

function openChart(row) {
  chartOrder.value = { created_at: row.created_at, entry_price: row.entry_price, side: row.side }
  chartSymbol.value = row.symbol
  chartInterval.value = row.interval || '1h'
  chartVisible.value = true
  loadChartBars()
}

function closeChart() {
  chartRequest += 1
  chartLoading.value = false
}

const editVisible = ref(false)
const editForm = reactive({ id: null, side: '', mark: 0, sl: 0, tp: 0 })
const editSaving = ref(false)

const summary = computed(() => {
  if (orderSummary.value) {
    return {
      open: Number(orderSummary.value.open || 0),
      closed: Number(orderSummary.value.closed || 0),
      realized: Number(orderSummary.value.realized_pnl || 0),
      unrealized: Number(orderSummary.value.unrealized_pnl || 0),
    }
  }
  const closed = orders.value.filter((row) => row.status === 'CLOSED')
  return {
    open: orders.value.filter((row) => row.status === 'OPEN').length,
    closed: closed.length,
    realized: closed.reduce((total, row) => total + Number(row.realized_pnl || 0), 0),
    unrealized: orders.value.filter((row) => row.status === 'OPEN').reduce((total, row) => total + Number(row.unrealized_pnl || 0), 0),
  }
})

function orderParams() {
  const params = { page: pagination.page, page_size: pagination.pageSize }
  if (filters.status) params.status = filters.status
  if (filters.symbol) params.symbol = filters.symbol
  if (filters.strategy) params.strategy = filters.strategy
  return params
}

function isCurrentQuery(params) {
  return JSON.stringify(params) === JSON.stringify(orderParams())
}

const symbols = computed(() => {
  const list = new Set(settings.value.symbols || [])
  orders.value.forEach((row) => list.add(row.symbol))
  return [...list]
})

const strategyOptions = computed(() => {
  const options = new Map(orderStrategies.value.map((item) => [item.code, { code: item.code, label: item.name || item.code }]))
  orders.value.forEach((row) => {
    if (row.strategy && !options.has(row.strategy)) options.set(row.strategy, { code: row.strategy, label: row.strategy })
  })
  if (filters.strategy && !options.has(filters.strategy)) options.set(filters.strategy, { code: filters.strategy, label: filters.strategy })
  return [...options.values()]
})

function load() {
  if (disposed) return Promise.resolve()
  reloadRequested = true
  if (loadPromise) return loadPromise
  loading.value = true
  loadPromise = (async () => {
    do {
      reloadRequested = false
      const params = orderParams()
      try {
        const data = await api.getOrders(params)
        if (!disposed && isCurrentQuery(params)) {
          const rows = data.orders || []
          pagination.total = Number(data.total ?? rows.length)
          const lastPage = Math.max(1, Math.ceil(pagination.total / pagination.pageSize))
          if (pagination.page > lastPage) {
            pagination.page = lastPage
            reloadRequested = true
            continue
          }
          orders.value = rows
          orderSummary.value = data.summary || null
          ordersLoaded.value = true
          ordersError.value = ''
        }
      } catch (error) {
        if (!disposed && isCurrentQuery(params)) ordersError.value = error.message || '加载失败'
      }
    } while (reloadRequested && !disposed)
  })().finally(() => {
    loadPromise = null
    loading.value = false
  })
  return loadPromise
}

function changePage(page) {
  if (page === pagination.page) return Promise.resolve()
  pagination.page = page
  return load()
}

function changePageSize(size) {
  if (size === pagination.pageSize) return Promise.resolve()
  pagination.pageSize = size
  pagination.page = 1
  return load()
}

function changeFilters() {
  pagination.page = 1
  ordersLoaded.value = false
  return load()
}

function loadSettings() {
  if (disposed) return Promise.resolve()
  if (settingsPromise) return settingsPromise
  settingsLoading.value = true
  settingsPromise = (async () => {
    try {
      const data = await api.getSettings()
      if (!disposed) {
        settings.value = data
        settingsError.value = ''
      }
    } catch (error) {
      if (!disposed) settingsError.value = error.message || '加载失败'
    }
  })().finally(() => {
    settingsLoading.value = false
    settingsPromise = null
  })
  return settingsPromise
}

function loadOrderStrategies() {
  if (disposed) return Promise.resolve()
  if (strategiesPromise) return strategiesPromise
  strategiesLoading.value = true
  strategiesPromise = (async () => {
    try {
      const data = await api.getOrderStrategies()
      if (!disposed) {
        orderStrategies.value = data.strategies || []
        strategiesError.value = ''
        strategiesUnavailable.value = false
      }
    } catch (error) {
      // 旧版后台没有 /orders/strategies，此时筛选参数会被直接忽略。
      // 明确禁用筛选，避免「选了却没过滤」的静默失败。
      if (!disposed) {
        strategiesError.value = error.message || '加载失败'
        strategiesUnavailable.value = true
        if (filters.strategy) {
          filters.strategy = ''
          changeFilters()
        }
      }
    }
  })().finally(() => {
    strategiesLoading.value = false
    strategiesPromise = null
  })
  return strategiesPromise
}

function retryStrategies() {
  strategiesUnavailable.value = false
  strategiesError.value = ''
  return loadOrderStrategies()
}

async function closeOrder(row) {
  try {
    await ElMessageBox.confirm(
      `按当前标记价 ${price(row.mark_price)} 平仓 ${row.symbol} ${sideLabel(row.side)}？`,
      '确认平仓',
      { type: 'warning', confirmButtonText: '平仓', cancelButtonText: '取消' },
    )
    await api.closeOrder(row.id)
    ElMessage.success('已平仓')
    await load()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message)
  }
}

async function reverseOrder(row) {
  try {
    await ElMessageBox.confirm(
      `先平掉当前 ${sideLabel(row.side)}，再按相反方向开同等保证金仓位？此操作按市价执行。`,
      '确认反手',
      { type: 'warning', confirmButtonText: '反手', cancelButtonText: '取消' },
    )
    await api.reverseOrder(row.id)
    ElMessage.success('已反手')
    await load()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message)
  }
}

function openEdit(row) {
  editForm.id = row.id
  editForm.side = row.side
  editForm.mark = row.mark_price
  editForm.sl = row.sl
  editForm.tp = row.tp
  editVisible.value = true
}

async function saveEdit() {
  editSaving.value = true
  try {
    await api.patchOrder(editForm.id, { sl: Number(editForm.sl), tp: Number(editForm.tp) })
    ElMessage.success('止损止盈已更新')
    editVisible.value = false
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    editSaving.value = false
  }
}

onMounted(() => {
  loadSettings()
  loadOrderStrategies()
  load()
  timer = setInterval(load, 30000)
})
onBeforeUnmount(() => {
  disposed = true
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div>
    <div class="grid stats-grid" style="margin-bottom: 14px">
      <div class="stat-card"><div class="stat-label">当前筛选持仓</div><div class="stat-value">{{ ordersLoaded ? summary.open : '—' }}</div><div class="stat-sub">符合筛选的全部持仓</div></div>
      <div class="stat-card"><div class="stat-label">当前筛选已平仓</div><div class="stat-value">{{ ordersLoaded ? summary.closed : '—' }}</div><div class="stat-sub">含止损 / 止盈 / 信号反转</div></div>
      <div class="stat-card"><div class="stat-label">已实现盈亏</div><div class="stat-value" :class="pnlClass(summary.realized)">{{ ordersLoaded ? money(summary.realized) : '—' }}</div><div class="stat-sub">全部匹配订单，已扣手续费</div></div>
      <div class="stat-card"><div class="stat-label">未实现盈亏</div><div class="stat-value" :class="pnlClass(summary.unrealized)">{{ ordersLoaded ? money(summary.unrealized) : '—' }}</div><div class="stat-sub">全部匹配持仓，按标记价</div></div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">订单列表</div>
          <div class="panel-sub">{{ loading ? '订单加载中…' : '修改止损止盈后由巡检任务继续监控' }}</div>
        </div>
        <div class="row">
          <el-select v-model="filters.status" size="small" style="width: 130px" @change="changeFilters">
            <el-option label="持仓中" value="OPEN" />
            <el-option label="已平仓" value="CLOSED" />
            <el-option label="全部" value="" />
          </el-select>
          <el-select v-model="filters.symbol" :loading="settingsLoading" size="small" clearable placeholder="全部币种" style="width: 130px" @change="changeFilters">
            <el-option v-for="symbol in symbols" :key="symbol" :label="symbol" :value="symbol" />
          </el-select>
          <el-select v-model="filters.strategy" :loading="strategiesLoading" :disabled="strategiesUnavailable" size="small" clearable filterable :placeholder="strategiesUnavailable ? '指标筛选不可用' : '全部信号指标'" style="width: 180px" @change="changeFilters">
            <el-option v-for="item in strategyOptions" :key="item.code" :label="item.label" :value="item.code" />
          </el-select>
          <el-button size="small" @click="load">刷新</el-button>
        </div>
      </div>

      <div v-if="ordersError" class="muted" role="alert" style="margin-bottom: 10px">
        <span class="down">订单加载失败：{{ ordersError }}</span>
        <el-button size="small" text type="primary" @click="load">重试</el-button>
      </div>
      <div v-if="settingsError" class="muted" role="alert" style="margin-bottom: 10px">
        币种配置加载失败，仍可查看订单。
        <el-button size="small" text type="primary" @click="loadSettings">重试</el-button>
      </div>
      <div v-if="strategiesError" class="muted" role="alert" style="margin-bottom: 10px">
        <span v-if="strategiesUnavailable" class="down">后台服务未加载策略筛选，请重启后台服务后点击重试；在此之前筛选不可用。</span>
        <span v-else>策略选项加载失败：{{ strategiesError }}</span>
        <el-button size="small" text type="primary" @click="retryStrategies">重试</el-button>
      </div>
      <el-table v-loading="loading" :data="orders" class="table-compact" :empty-text="loading ? '订单加载中…' : ordersError ? '订单加载失败' : '暂无订单'">
        <el-table-column prop="id" label="#" width="56" />
        <el-table-column prop="symbol" label="币种" width="110">
          <template #default="{ row }"><el-button link type="primary" @click="openChart(row)">{{ row.symbol }}</el-button></template>
        </el-table-column>
        <el-table-column label="方向" width="76">
          <template #default="{ row }">
            <el-tag size="small" :type="row.side === 'LONG' ? 'success' : 'danger'" effect="plain">{{ sideLabel(row.side) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="86">
          <template #default="{ row }">
            <el-tag size="small" effect="plain" :type="row.status === 'OPEN' ? 'warning' : 'info'">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="数量" width="100"><template #default="{ row }"><span class="mono">{{ row.qty }}</span></template></el-table-column>
        <el-table-column label="开仓价" width="96"><template #default="{ row }"><span class="mono">{{ price(row.entry_price) }}</span></template></el-table-column>
        <el-table-column label="标记 / 平仓价" width="120">
          <template #default="{ row }"><span class="mono">{{ price(row.status === 'OPEN' ? row.mark_price : row.close_price) }}</span></template>
        </el-table-column>
        <el-table-column label="杠杆" width="62"><template #default="{ row }">×{{ row.leverage }}</template></el-table-column>
        <el-table-column label="保证金" width="96"><template #default="{ row }"><span class="mono">{{ money(row.margin) }}</span></template></el-table-column>
        <el-table-column label="止损 / 止盈" min-width="145">
          <template #default="{ row }"><span class="mono">{{ price(row.sl) }} / {{ price(row.tp) }}</span></template>
        </el-table-column>
        <el-table-column label="盈亏" width="130">
          <template #default="{ row }">
            <span class="mono" :class="pnlClass(row.status === 'OPEN' ? row.unrealized_pnl : row.realized_pnl)">
              {{ money(row.status === 'OPEN' ? row.unrealized_pnl : row.realized_pnl) }}
            </span>
            <div class="muted" v-if="row.status === 'OPEN'">浮盈 {{ pct(row.unrealized_pnl / Math.max(row.margin, 1) * 100) }}</div>
          </template>
        </el-table-column>
        <el-table-column label="手续费" width="96"><template #default="{ row }"><span class="mono">{{ money(Number(row.entry_fee || 0) + Number(row.exit_fee || 0)) }}</span></template></el-table-column>
        <el-table-column label="信号指标 / 平仓原因" min-width="160">
          <template #default="{ row }">
            <div>{{ strategyLabel(row.strategy) }}</div>
            <div class="muted">{{ row.status === 'CLOSED' ? closeReasonLabel(row.close_reason) : '持仓监控中' }}</div>
          </template>
        </el-table-column>
        <el-table-column label="时间" min-width="150">
          <template #default="{ row }">
            <div class="muted">开 {{ tsLabel(row.created_at) }}</div>
            <div class="muted" v-if="row.closed_at">平 {{ tsLabel(row.closed_at) }}</div>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="190" fixed="right">
          <template #default="{ row }">
            <template v-if="row.status === 'OPEN'">
              <el-button size="small" link type="primary" :disabled="loading" @click="openEdit(row)">改止损止盈</el-button>
              <el-button size="small" link type="warning" :disabled="loading" @click="reverseOrder(row)">反手</el-button>
              <el-button size="small" link type="danger" :disabled="loading" @click="closeOrder(row)">平仓</el-button>
            </template>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
      </el-table>
      <div class="table-pagination">
        <el-pagination
          :current-page="pagination.page"
          :page-size="pagination.pageSize"
          :total="pagination.total"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @current-change="changePage"
          @size-change="changePageSize"
        />
      </div>
    </div>

    <el-dialog v-model="chartVisible" :title="`${chartSymbol} · ${chartInterval} K线`" width="min(960px, 95vw)" destroy-on-close @close="closeChart">
      <div class="row" style="margin-bottom: 12px">
        <span class="muted">最近 {{ chartBars.length }} 根K线 · EA20 / EA60</span>
        <el-select v-model="chartInterval" size="small" style="width: 100px" @change="loadChartBars">
          <el-option v-for="item in [...new Set([chartInterval, '1m', '5m', '15m', '1h', '4h', '1d'])]" :key="item" :label="item" :value="item" />
        </el-select>
        <el-button size="small" :loading="chartLoading" @click="loadChartBars">刷新</el-button>
      </div>
      <div class="muted" style="margin-bottom: 8px">{{ chartOrder?.side === 'SHORT' ? '做空开仓' : '买入开仓' }}：{{ price(chartOrder?.entry_price) }} · {{ tsLabel(chartOrder?.created_at) }}</div>
      <div v-if="chartBars.length && !entryPoint" class="form-hint" role="status">当前K线范围内没有对应的开仓时间，无法显示开仓标记。</div>
      <div v-if="chartError" class="form-hint" role="alert">K线加载失败：{{ chartError }} <el-button size="small" link type="primary" @click="loadChartBars">重试</el-button></div>
      <EChart v-if="chartBars.length" :option="chartOption" height="470px" />
      <div v-else class="empty-hint" style="min-height: 220px" role="status">{{ chartLoading ? '正在加载K线…' : chartError ? 'K线暂不可用，请重试。' : '暂无K线数据。' }}</div>
    </el-dialog>

    <el-dialog v-model="editVisible" title="调整止损止盈" width="420px">
      <div class="form-grid" style="grid-template-columns: 1fr">
        <div>
          <div class="muted">当前标记价 {{ price(editForm.mark) }} · {{ sideLabel(editForm.side) }}</div>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">止损价</div>
          <el-input-number v-model="editForm.sl" :precision="2" :step="10" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">止盈价</div>
          <el-input-number v-model="editForm.tp" :precision="2" :step="10" style="width: 100%" />
        </div>
      </div>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" :loading="editSaving" @click="saveEdit">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>
