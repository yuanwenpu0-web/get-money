<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import TablePager from '../components/TablePager.vue'
import { closeReasonLabel, money, pct, pnlClass, price, sideLabel, statusLabel, strategyLabel, tsLabel } from '../utils/format'
import { PAGE_SIZE } from '../utils/pagination'
import { barsToCover, intervalMs, KLINE_MAX_LIMIT, locateBar, tradeMarker, zoomForIndexes } from '../utils/kline'

const orders = ref([])
const settings = ref({})
const loading = ref(false)
// 批量平仓用「勾选 ID」而不是 el-table 的 selection：才能跨分页保留、按搜索条件全选
const selectedIds = ref([])
const selectingAll = ref(false)
const batchClosing = ref(false)
const ordersLoaded = ref(false)
const ordersError = ref('')
const settingsLoading = ref(false)
const settingsError = ref('')
const orderStrategies = ref([])
const strategiesLoading = ref(false)
const strategiesError = ref('')
const strategiesUnavailable = ref(false)
const filters = reactive({ status: 'OPEN', symbol: '', strategy: '' })
const pagination = reactive({ page: 1, pageSize: PAGE_SIZE, total: 0 })
const orderSummary = ref(null)
const exchangeEvents = ref([])
const exchangeEventsOpen = ref(false)
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
// 图表视窗（百分比）。拖动时只更新 chartLiveZoom，不触发重算；拼更早历史或自动定位时才写 chartZoom 重建。
const chartZoom = ref({ start: 0, end: 100 })
const chartOlderLoading = ref(false)
const chartHasMore = ref(true)
let chartLiveZoom = { start: 0, end: 100 }
let chartRequest = 0
// 周期内最多自动往前翻几页，保证开仓时间落在本地数据里（不卡死在循环里）。
const MAX_AUTO_PAGES = 3

// 周期切换：常用周期做成快捷按钮，其余放进下拉，减少点击层级。
const QUICK_INTERVALS = ['5m', '15m', '1h', '4h']
const ALL_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w']
const OTHER_INTERVALS = ALL_INTERVALS.filter((item) => !QUICK_INTERVALS.includes(item))

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

// 订单时间 → 所属K线落点：复用 kline.js 的 locateBar，只按时间定位，不要求正好对齐开盘。
// 完全在本地数据范围外返回 null，由调用方决定是否继续加载更早数据。
function orderPoint(timeValue, priceValue) {
  const value = Number(priceValue)
  if (!Number.isFinite(value) || value <= 0) return null
  const located = locateBar(chartBars.value, timeValue, chartInterval.value)
  return located ? { index: located.index, price: value } : null
}

const entryPoint = computed(() => orderPoint(chartOrder.value?.created_at, chartOrder.value?.entry_price))

// 只有已平仓订单才有平仓标记；平仓价或平仓时间缺失时不画，改为在弹窗里给提示。
const isClosed = computed(() => chartOrder.value?.status === 'CLOSED')
const exitPoint = computed(() =>
  isClosed.value ? orderPoint(chartOrder.value?.closed_at, chartOrder.value?.close_price) : null,
)

const markPointData = computed(() => {
  const points = []
  if (entryPoint.value) {
    const marker = tradeMarker({ side: chartOrder.value?.side })
    points.push({
      name: '开仓',
      coord: [entryPoint.value.index, entryPoint.value.price],
      value: entryPoint.value.price,
      symbol: marker.symbol,
      symbolRotate: marker.symbolRotate,
      itemStyle: { color: marker.color },
      label: { formatter: `开仓 ${price(entryPoint.value.price)}` },
    })
  }
  if (exitPoint.value) {
    // 开仓/平仓落在同一根K线时（大周期很常见），两个标签上下错开，避免叠在一起。
    const sameBar = entryPoint.value && entryPoint.value.index === exitPoint.value.index
    const marker = tradeMarker({ side: chartOrder.value?.side, action: 'exit' })
    points.push({
      name: '平仓',
      coord: [exitPoint.value.index, exitPoint.value.price],
      value: exitPoint.value.price,
      symbol: marker.symbol,
      symbolRotate: marker.symbolRotate,
      itemStyle: { color: marker.color },
      label: { formatter: `平仓 ${price(exitPoint.value.price)}`, ...(sameBar ? { position: 'bottom' } : {}) },
    })
  }
  return points
})

// 水平标记线：开仓价 / 平仓价 / 止损 / 止盈；止盈止损仅 OPEN 且值>0 时显示。
// 用 markLine 挂在 K 线系列上，不进顶部 EMA 图例，也不进 tooltip（silent）。
const markLineData = computed(() => {
  const order = chartOrder.value
  if (!order) return []
  const lines = []
  if (entryPoint.value) {
    lines.push({
      yAxis: Number(order.entry_price),
      lineStyle: { color: '#e6a23c', width: 1.2, type: 'solid' },
      label: { formatter: `开仓 ${price(order.entry_price)}`, color: '#e6a23c' },
    })
  }
  if (exitPoint.value) {
    lines.push({
      yAxis: Number(order.close_price),
      lineStyle: { color: '#38bdf8', width: 1.2, type: 'solid' },
      label: { formatter: `平仓 ${price(order.close_price)}`, color: '#38bdf8' },
    })
  }
  if (order.status === 'OPEN') {
    if (Number(order.sl) > 0) {
      lines.push({
        yAxis: Number(order.sl),
        lineStyle: { color: '#fb7185', width: 1.2, type: 'dashed' },
        label: { formatter: `止损 ${price(order.sl)}`, color: '#fb7185' },
      })
    }
    if (Number(order.tp) > 0) {
      lines.push({
        yAxis: Number(order.tp),
        lineStyle: { color: '#34d399', width: 1.2, type: 'dashed' },
        label: { formatter: `止盈 ${price(order.tp)}`, color: '#34d399' },
      })
    }
  }
  return lines
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
  grid: [{ left: 65, right: 56, top: 46, height: '60%' }, { left: 65, right: 56, top: '76%', height: '12%' }],
  xAxis: [
    { type: 'category', data: chartBars.value.map((bar) => tsLabel(bar.time)), axisLabel: { show: false } },
    { type: 'category', gridIndex: 1, data: chartBars.value.map((bar) => tsLabel(bar.time)), axisLabel: { hideOverlap: true } },
  ],
  yAxis: [{ scale: true }, { scale: true, gridIndex: 1, splitNumber: 2 }],
  dataZoom: [
    {
      type: 'inside',
      xAxisIndex: [0, 1],
      start: chartZoom.value.start,
      end: chartZoom.value.end,
      throttle: 30,
      zoomOnMouseWheel: true,
      moveOnMouseWheel: false,
      moveOnMouseMove: true,
    },
    { type: 'slider', xAxisIndex: [0, 1], start: chartZoom.value.start, end: chartZoom.value.end, bottom: 2, height: 18 },
  ],
  series: [
    { name: 'K线', type: 'candlestick', data: chartBars.value.map((bar) => [bar.open, bar.close, bar.low, bar.high]), itemStyle: { color: '#34d399', color0: '#fb7185', borderColor: '#34d399', borderColor0: '#fb7185' },
      markPoint: {
        symbol: 'pin',
        symbolSize: 44,
        itemStyle: { color: '#e6a23c' },
        label: { color: '#fff', fontSize: 10, fontWeight: 600, position: 'top', distance: 4, backgroundColor: 'rgba(2,6,23,0.82)', padding: [3, 5], borderRadius: 3 },
        data: markPointData.value,
      },
      markLine: {
        symbol: ['none', 'none'],
        silent: true,
        label: { color: '#cbd5e1', fontSize: 10, position: 'end' },
        data: markLineData.value,
      },
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

// 自动把开仓/平仓放进初始视窗（替换原来"后 60%"的启发式），切周期后同样自动定位。
function autoZoom() {
  const indexes = [entryPoint.value?.index, exitPoint.value?.index].filter((value) => typeof value === 'number')
  const zoom = zoomForIndexes(chartBars.value.length, indexes, { minSpanBars: 80 })
  chartZoom.value = { ...zoom }
  chartLiveZoom = { ...zoom }
}

// 只把更早的一批K线拼到前面；不调整视窗，调用方决定要不要重定位。
// 返回拼上的根数（0 表示已到头或无数据）。拖动加载与自动翻页共用。
async function prependOlder(limit) {
  const rows = chartBars.value
  const firstTime = Number(rows[0]?.time)
  if (!rows.length || !Number.isFinite(firstTime)) return 0
  const data = await api.getKlines(chartSymbol.value, chartInterval.value, limit, firstTime - 1)
  const older = data.bars || []
  if (older.length) {
    chartBars.value = [...older, ...rows]
  } else {
    // 只有真正返回空页才算到头，短页（不足 limit）仍可能有更早数据，继续翻。
    chartHasMore.value = false
  }
  return older.length
}

// 加载完首屏后，若开仓时间仍早于本地最早一根且还能往前取，自动再翻更早分页（最多 MAX_AUTO_PAGES 次），
// 加载完再自动定位，避免"切到 5m 标记消失"。
async function ensureEntryInWindow(request) {
  const order = chartOrder.value
  const entryTime = order?.created_at ? Date.parse(order.created_at) : NaN
  let pages = 0
  while (request === chartRequest && Number.isFinite(entryTime) && pages < MAX_AUTO_PAGES) {
    if (locateBar(chartBars.value, order.created_at, chartInterval.value)) break
    const first = Number(chartBars.value[0]?.time)
    // 开仓时间晚于首根说明缺的是更新的数据（往前翻没用），或已没有更早数据，停下。
    if (!(entryTime < first) || !chartHasMore.value) break
    const before = first
    const added = await prependOlder(600)
    if (disposed || request !== chartRequest) return
    if (!added || Number(chartBars.value[0]?.time) === before) break
    pages += 1
  }
  if (request === chartRequest) autoZoom()
}

async function loadChartBars() {
  const request = ++chartRequest
  chartBars.value = []
  chartLoading.value = true
  chartError.value = ''
  chartHasMore.value = true
  chartZoom.value = { start: 0, end: 100 }
  chartLiveZoom = { start: 0, end: 100 }
  try {
    const order = chartOrder.value
    const anchors = [order?.created_at, order?.closed_at].filter(Boolean)
    // limit 用 barsToCover 计算（上限 1500），保证开平仓都落在窗口内。
    const limit = Math.min(KLINE_MAX_LIMIT, barsToCover(chartInterval.value, anchors, Date.now()))
    const data = await api.getKlines(chartSymbol.value, chartInterval.value, limit)
    if (!disposed && request === chartRequest) {
      chartBars.value = data.bars || []
      await ensureEntryInWindow(request)
    }
  } catch (error) {
    if (!disposed && request === chartRequest) chartError.value = error.message || '加载失败'
  } finally {
    if (!disposed && request === chartRequest) chartLoading.value = false
  }
}

// 拖到最左边时把更早的一批K线拼到前面，并整体右移视窗，视觉上不跳。
async function loadOlderChartBars() {
  const rows = chartBars.value
  if (!rows.length || chartOlderLoading.value || chartLoading.value || !chartHasMore.value) return
  const request = chartRequest
  const before = rows.length
  chartOlderLoading.value = true
  try {
    const added = await prependOlder(600)
    if (disposed || request !== chartRequest) return
    if (!added) {
      ElMessage.info('已经到本地最早的数据了')
      return
    }
    const total = chartBars.value.length
    const start = ((chartLiveZoom.start / 100) * before + added) / total * 100
    const end = ((chartLiveZoom.end / 100) * before + added) / total * 100
    chartZoom.value = { start, end }
    chartLiveZoom = { start, end }
  } catch (error) {
    if (!disposed) ElMessage.error(error.message || '加载更早K线失败')
  } finally {
    if (!disposed) chartOlderLoading.value = false
  }
}

// 把视窗中心移到指定K线，保留上下文，供"定位开仓/定位平仓"按钮使用。
function zoomToIndex(index) {
  const total = chartBars.value.length
  if (!total || !Number.isInteger(index)) return
  const pad = Math.max(6, Math.round(80 * 0.15))
  const from = Math.max(0, index - pad)
  const to = Math.min(total - 1, index + pad)
  const zoom = { start: (from / total) * 100, end: ((to + 1) / total) * 100 }
  chartZoom.value = { ...zoom }
  chartLiveZoom = { ...zoom }
}

function focusEntry() {
  if (entryPoint.value) zoomToIndex(entryPoint.value.index)
}

function focusExit() {
  if (exitPoint.value) zoomToIndex(exitPoint.value.index)
}

function switchInterval(interval) {
  if (interval === chartInterval.value) return Promise.resolve()
  chartInterval.value = interval
  return loadChartBars()
}

function onChartZoom(zoom) {
  chartLiveZoom = zoom
  if (zoom.start <= 3) loadOlderChartBars()
}

function openChart(row) {
  // 已平仓的订单要能画平仓标记，所以把平仓价/平仓时间/平仓原因一起带进弹窗状态。
  chartOrder.value = {
    id: row.id,
    status: row.status,
    side: row.side,
    created_at: row.created_at,
    entry_price: row.entry_price,
    close_price: row.close_price,
    closed_at: row.closed_at,
    close_reason: row.close_reason,
    realized_pnl: row.realized_pnl,
    // 止损/止盈用于K线上的水平参考线（仅持仓单显示）
    sl: row.sl,
    tp: row.tp,
  }
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
      // 不受筛选影响的持仓总数，仅用于兼容旧版后台
      openTotal: Number(orderSummary.value.open_total ?? orderSummary.value.open ?? 0),
      // 当前搜索条件下可勾选的持仓数（跨分页），与按筛选全选接口同口径
      selectableOpen: Number(orderSummary.value.selectable_open ?? orderSummary.value.open ?? 0),
    }
  }
  const closed = orders.value.filter((row) => row.status === 'CLOSED')
  const open = orders.value.filter((row) => row.status === 'OPEN')
  return {
    open: open.length,
    closed: closed.length,
    realized: closed.reduce((total, row) => total + Number(row.realized_pnl || 0), 0),
    unrealized: open.reduce((total, row) => total + Number(row.unrealized_pnl || 0), 0),
    openTotal: open.length,
    selectableOpen: open.length,
  }
})

// ── 批量勾选 ──────────────────────────────────────────────────────────────
// 自己维护 ID 列表：翻页不清空，切换筛选条件才清空；按筛选全选时把服务端展开的
// 跨页 ID 合并进来。提交时只发 ID 列表，服务端不会重新扩大范围。
const selectedSet = computed(() => new Set(selectedIds.value))
const selectedCount = computed(() => selectedIds.value.length)
const selectableOpen = computed(() => summary.value.selectableOpen)
const pageSelectableIds = computed(() =>
  orders.value.filter((row) => row.status === 'OPEN').map((row) => Number(row.id)),
)
const allPageSelected = computed(() =>
  pageSelectableIds.value.length > 0 && pageSelectableIds.value.every((id) => selectedSet.value.has(id)),
)
const somePageSelected = computed(() =>
  pageSelectableIds.value.some((id) => selectedSet.value.has(id)),
)

function setSelectedIds(ids) {
  selectedIds.value = [...new Set(ids.map(Number).filter((id) => Number.isInteger(id) && id > 0))]
}

function clearSelection() {
  selectedIds.value = []
}

function toggleRowSelection(row, checked) {
  const next = new Set(selectedIds.value)
  if (checked) next.add(Number(row.id))
  else next.delete(Number(row.id))
  selectedIds.value = [...next]
}

function togglePageSelection(checked) {
  const next = new Set(selectedIds.value)
  pageSelectableIds.value.forEach((id) => {
    if (checked) next.add(id)
    else next.delete(id)
  })
  selectedIds.value = [...next]
}

// 列表刷新后，当前页里已不再持仓（被止损/止盈平掉）的勾选项自动失效
function pruneVisibleSelection() {
  const stale = new Set(orders.value.filter((row) => row.status !== 'OPEN').map((row) => Number(row.id)))
  if (!stale.size) return
  const next = selectedIds.value.filter((id) => !stale.has(Number(id)))
  if (next.length !== selectedIds.value.length) selectedIds.value = next
}

async function selectAllMatching() {
  if (selectingAll.value) return
  selectingAll.value = true
  try {
    // 只带筛选条件，不带分页参数：服务端展开全部匹配持仓
    const params = {}
    if (filters.status) params.status = filters.status
    if (filters.symbol) params.symbol = filters.symbol
    if (filters.strategy) params.strategy = filters.strategy
    const data = await api.getOpenOrderIds(params)
    const ids = (data && data.order_ids) || []
    if (!ids.length) {
      ElMessage.info('当前筛选下没有可平仓持仓')
      return
    }
    setSelectedIds([...selectedIds.value, ...ids])
    ElMessage.success(`已按当前筛选选中 ${ids.length} 笔持仓（含其他分页），当前共勾选 ${selectedIds.value.length} 笔`)
  } catch (error) {
    ElMessage.error(error.message || '按筛选全选失败')
  } finally {
    selectingAll.value = false
  }
}

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

async function loadExchangeEvents() {
  if (disposed) return
  try {
    const data = await api.getExchangeEvents({ limit: 20 })
    if (!disposed) exchangeEvents.value = data.events || []
  } catch { /* 回执列表加载失败不影响订单主表 */ }
}

function eventTagType(status) {
  if (status === 'filled' || status === 'accepted' || status === 'canceled') return 'success'
  if (status === 'blocked') return 'warning'
  return 'danger'
}

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
          pruneVisibleSelection()
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

function changeFilters() {
  pagination.page = 1
  ordersLoaded.value = false
  // 勾选绑定在一次搜索上：筛选一变就清空，避免平掉不在当前筛选里的持仓
  clearSelection()
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

// 批量平仓：只提交用户勾选的订单 ID（可来自逐行勾选或按筛选全选），服务端按最新标记价逐笔市价成交
async function closeSelected() {
  const ids = [...selectedIds.value]
  if (!ids.length) {
    ElMessage.info('请先勾选要平掉的持仓')
    return
  }
  const visible = pageSelectableIds.value.filter((id) => selectedSet.value.has(id)).length
  const offPage = ids.length - visible
  const scope = offPage > 0 ? `，其中 ${offPage} 笔不在当前页（按筛选全选的跨页持仓）` : ''
  try {
    await ElMessageBox.confirm(
      `将按最新标记价市价平掉勾选的 ${ids.length} 笔持仓${scope}，且无法撤销。期间已被止损/止盈平掉的订单会自动跳过。`,
      '确认批量平仓',
      { type: 'warning', confirmButtonText: `平掉 ${ids.length} 笔`, cancelButtonText: '取消' },
    )
  } catch (error) {
    return
  }
  batchClosing.value = true
  try {
    const result = await api.batchCloseOrders(ids)
    const closedCount = Number(result?.closed_count || 0)
    const failed = result?.failed || []
    const skippedCount = Number(result?.skipped_count || 0)
    // 失败的留在勾选项里方便重试；成功与自动跳过的清掉
    setSelectedIds(failed.map((item) => item.id))
    if (failed.length) {
      ElMessage.warning(
        `已平 ${closedCount} 笔，跳过 ${skippedCount} 笔，${failed.length} 笔失败（${failed.map((item) => item.symbol || item.id).join('、')}）`,
      )
    } else if (skippedCount) {
      ElMessage.warning(`已平 ${closedCount} 笔，${skippedCount} 笔已不在持仓中，自动跳过`)
    } else {
      ElMessage.success(`已平掉 ${closedCount} 笔持仓，已实现 ${money(result?.realized_pnl)}`)
    }
    await load()
  } catch (error) {
    ElMessage.error(error.message || '批量平仓失败')
  } finally {
    batchClosing.value = false
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
  loadExchangeEvents()
  timer = setInterval(() => { load(); loadExchangeEvents() }, 30000)
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

    <div class="panel" style="margin-bottom: 14px">
      <div class="panel-head">
        <div>
          <div class="panel-title">币安下单回执</div>
          <div class="panel-sub">开仓主单、保护单、平仓、撤单的交易所返回结果；失败带错误码，便于核对「下单是否成功」</div>
        </div>
        <div class="row">
          <el-button size="small" text type="primary" @click="loadExchangeEvents">刷新</el-button>
          <el-button size="small" text @click="exchangeEventsOpen = !exchangeEventsOpen">
            {{ exchangeEventsOpen ? '收起' : `展开最近 ${exchangeEvents.length} 条` }}
          </el-button>
        </div>
      </div>
      <div v-if="exchangeEventsOpen">
        <div v-if="!exchangeEvents.length" class="muted">暂无交易所下单记录（仅实盘 / DEMO 模式产生）</div>
        <div v-for="item in exchangeEvents" :key="item.id" class="row mono" style="gap: 10px; padding: 4px 0; border-bottom: 1px solid rgba(148,163,184,0.1); flex-wrap: wrap">
          <span class="muted">{{ tsLabel(item.created_at) }}</span>
          <el-tag size="small" :type="eventTagType(item.status)" effect="plain">{{ item.status }}</el-tag>
          <span>{{ item.symbol }}</span>
          <span>{{ item.action }}</span>
          <span v-if="item.exchange_order_id" class="muted">#{{ item.exchange_order_id }}</span>
          <span v-if="item.avg_price" class="muted">@{{ price(item.avg_price) }}</span>
          <span v-if="item.code" class="down">{{ item.code }}</span>
          <span v-if="item.message" class="muted" style="flex: 1 1 240px">{{ item.message }}</span>
        </div>
      </div>
      <div v-else class="form-hint">
        最近一笔：{{ exchangeEvents[0] ? `${exchangeEvents[0].symbol} ${exchangeEvents[0].action} ${exchangeEvents[0].status}` : '暂无' }}
      </div>
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
          <el-button
            size="small"
            :loading="selectingAll"
            :disabled="loading || batchClosing || !selectableOpen"
            :title="selectableOpen ? '按当前筛选条件勾选全部持仓（含其他分页）' : '当前筛选下没有可平仓持仓'"
            @click="selectAllMatching"
          >
            按筛选全选{{ selectableOpen ? ` (${selectableOpen})` : '' }}
          </el-button>
          <el-button
            size="small"
            type="danger"
            plain
            :loading="batchClosing"
            :disabled="loading || selectingAll || !selectedCount"
            :title="selectedCount ? '按最新标记价平掉勾选的持仓' : '请先勾选持仓'"
            @click="closeSelected"
          >
            批量平仓{{ selectedCount ? ` (${selectedCount})` : '' }}
          </el-button>
        </div>
      </div>

      <div v-if="selectedCount" class="row muted" style="gap: 6px; margin-bottom: 10px">
        <span>
          已勾选 <b style="color: #f87171">{{ selectedCount }}</b> 笔持仓<span v-if="selectableOpen && selectedCount >= selectableOpen"> · 已覆盖当前筛选下全部持仓</span>
          · 勾选跨分页保留，切换筛选条件时自动清空
        </span>
        <el-button size="small" text type="primary" @click="clearSelection">清空选择</el-button>
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
        <el-table-column width="46" align="center">
          <template #header>
            <el-checkbox
              :model-value="allPageSelected"
              :indeterminate="somePageSelected && !allPageSelected"
              :disabled="!pageSelectableIds.length || batchClosing"
              aria-label="全选本页持仓"
              @change="togglePageSelection"
            />
          </template>
          <template #default="{ row }">
            <el-checkbox
              v-if="row.status === 'OPEN'"
              :model-value="selectedSet.has(Number(row.id))"
              :disabled="loading || batchClosing"
              :aria-label="`勾选订单 ${row.id}`"
              @change="(checked) => toggleRowSelection(row, checked)"
            />
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
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
        <el-table-column label="止损 / 止盈" min-width="175">
          <template #default="{ row }">
            <span class="mono">{{ price(row.sl) }} / {{ price(row.tp) }}</span>
            <div v-if="row.source === 'live'" class="muted" style="font-size: 11px">
              交易所 SL {{ row.sl_status || '—' }} / TP {{ row.tp_status || '—' }}
              <span v-if="row.pnl_source === 'binance'" class="up"> · 成交已核对</span>
              <span v-else-if="row.status === 'CLOSED'"> · 估算盈亏</span>
            </div>
          </template>
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
              <el-button size="small" link type="warning" :disabled="loading || batchClosing" @click="reverseOrder(row)">反手</el-button>
              <el-button size="small" link type="danger" :disabled="loading || batchClosing" @click="closeOrder(row)">平仓</el-button>
            </template>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
      </el-table>
      <TablePager
        :page="pagination.page"
        :page-size="pagination.pageSize"
        :total="pagination.total"
        :disabled="loading"
        jumper
        @change="changePage"
      />
    </div>

    <el-dialog v-model="chartVisible" :title="`${chartSymbol} · ${chartInterval} K线`" width="min(960px, 95vw)" destroy-on-close @close="closeChart">
      <div class="row" style="margin-bottom: 12px; flex-wrap: wrap; gap: 8px">
        <span class="muted">最近 {{ chartBars.length }} 根K线 · EA20 / EA60 · 向左拖动加载更早<span v-if="chartOlderLoading">（加载中…）</span><span v-else-if="chartBars.length && !chartHasMore"> · 已到本地最早数据</span></span>
        <span class="row" style="gap: 6px">
          <el-button
            v-for="item in QUICK_INTERVALS"
            :key="item"
            size="small"
            :type="chartInterval === item ? 'primary' : ''"
            @click="switchInterval(item)"
          >{{ item }}</el-button>
          <el-select :model-value="chartInterval" size="small" style="width: 110px" @change="switchInterval">
            <el-option v-for="item in OTHER_INTERVALS" :key="item" :label="item" :value="item" />
          </el-select>
        </span>
        <span class="row" style="gap: 6px">
          <el-button size="small" :disabled="!entryPoint" @click="focusEntry">定位开仓</el-button>
          <el-button size="small" :disabled="!exitPoint" @click="focusExit">定位平仓</el-button>
        </span>
        <el-button size="small" :loading="chartLoading" @click="loadChartBars">刷新</el-button>
      </div>
      <div class="muted" style="margin-bottom: 8px">{{ chartOrder?.side === 'SHORT' ? '做空开仓' : '买入开仓' }}：{{ price(chartOrder?.entry_price) }} · {{ tsLabel(chartOrder?.created_at) }}</div>
      <div v-if="isClosed" class="muted" style="margin-bottom: 8px">
        平仓：<span class="mono" style="color: #38bdf8">{{ price(chartOrder?.close_price) }}</span> · {{ tsLabel(chartOrder?.closed_at) }}
        <template v-if="chartOrder?.close_reason"> · {{ closeReasonLabel(chartOrder.close_reason) }}</template>
        <span class="mono" :class="pnlClass(chartOrder?.realized_pnl)"> · {{ money(chartOrder?.realized_pnl) }}</span>
      </div>
      <div v-if="chartBars.length && !entryPoint" class="form-hint" role="status">当前本地K线的最早一根仍晚于开仓时间，开仓标记暂不在窗口内。可向左拖动加载更早K线，或切换到更大周期（如 1h）以覆盖更长区间。</div>
      <div v-if="chartBars.length && isClosed && !exitPoint" class="form-hint" role="status">平仓时间不在当前本地K线范围内（或缺少平仓价），平仓标记暂不在窗口内。
        <el-button v-if="chartInterval !== '1h'" size="small" link type="primary" @click="switchInterval('1h')">切到 1h 看更长区间</el-button>
      </div>
      <div v-if="chartError" class="form-hint" role="alert">K线加载失败：{{ chartError }} <el-button size="small" link type="primary" @click="loadChartBars">重试</el-button></div>
      <EChart v-if="chartBars.length" :option="chartOption" height="470px" @datazoom="onChartZoom" />
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
