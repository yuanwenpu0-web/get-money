<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import TablePager from '../components/TablePager.vue'
import { number, price, tsLabel } from '../utils/format'
import { PAGE_SIZE } from '../utils/pagination'
import { barsToCover, locateBar, tradeMarker, zoomForIndexes } from '../utils/kline'

const settings = ref({})
const sources = ref(null)
const bars = ref([])
const allSymbols = ref([])
const loading = ref(false)
const syncing = ref(false)
const settingsLoading = ref(false)
const settingsReady = ref(false)
const settingsError = ref('')
const sourcesLoading = ref(false)
const sourcesError = ref('')
const symbolsLoading = ref(false)
const symbolsError = ref('')
const barsError = ref('')
// 成交标记只是叠加层：订单接口失败只写 tradesError，K线照常显示。
const trades = ref([])
const tradesError = ref('')
const tradesLoading = ref(false)
const showTrades = ref(true)
const locating = ref(false)
const quickIntervals = ['1m', '5m', '15m', '1h', '4h']
const statsPage = ref(1)
const statsPageSize = PAGE_SIZE
const statsTotal = ref(0)
let statsRequest = 0
let tradesRequest = 0
let tradesSymbol = ''
let disposed = false
let barsRequest = 0
let displayedQuery = ''
let queryTouched = false
let syncFormTouched = false

const query = reactive({ symbol: 'BTCUSDT', interval: '1m', limit: 600 })
const syncForm = reactive({ source: 'binance', symbols: ['BTCUSDT'], interval: '1m', days: 30 })

// K线视窗（百分比）。拖动过程中只更新普通变量 liveZoom，不触发重算，
// 避免每帧 setOption 造成卡顿；只有拼上更早的历史K线时才写 zoomWindow 重建图表。
const zoomWindow = ref({ start: 40, end: 100 })
const olderLoading = ref(false)
const hasMoreBars = ref(true)
let liveZoom = { start: 40, end: 100 }

const localStats = computed(() => sources.value?.local?.stats || [])
const symbolOptions = computed(() =>
  allSymbols.value.length ? allSymbols.value : settings.value.symbols || ['BTCUSDT', 'ETHUSDT'],
)
const scopeAll = computed(() => settings.value.symbol_scope !== 'custom')
const syncHint = computed(() =>
  scopeAll.value
    ? `全量同步：所有 U 本位永续合约${allSymbols.value.length ? `（约 ${allSymbols.value.length} 个）` : ''}，耗时较长请耐心等待，期间请勿关闭页面。`
    : `将同步自定义列表中的 ${syncForm.symbols.length} 个合约的 1m 原始K线，其他周期自动聚合。`,
)

function timeShort(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false })
}

function movingAverage(values, length) {
  const result = []
  let sum = 0
  for (let index = 0; index < values.length; index += 1) {
    sum += values[index]
    if (index >= length) sum -= values[index - length]
    result.push(index >= length - 1 ? +(sum / length).toFixed(2) : null)
  }
  return result
}

// 成交标记混在同一条 scatter 里，tooltip 仍按 axis 触发，需要自己把各系列拼回去。
function klineTooltip(items) {
  const list = Array.isArray(items) ? items : [items]
  if (!list.length) return ''
  const head = list[0]?.axisValueLabel || ''
  const rows = list.map((item) => {
    if (item.seriesName === '我的成交') return `<div>${item.data?.text || ''}</div>`
    if (item.seriesName === '成交量') return `<div>${item.marker || ''}成交量 ${number(item.value, 0)}</div>`
    if (item.seriesName === 'K线') {
      const value = item.value || []
      // category 轴上个别版本会在数组头部插入索引，按长度兼容
      const offset = value.length >= 5 ? 1 : 0
      return `<div>${item.marker || ''}开 ${price(value[offset])} 收 ${price(value[offset + 1])} 低 ${price(value[offset + 2])} 高 ${price(value[offset + 3])}</div>`
    }
    return `<div>${item.marker || ''}${item.seriesName} ${price(item.value)}</div>`
  })
  return `<div style="color:#94a3b8;font-size:10px;margin-bottom:2px">${head}</div>${rows.join('')}`
}

// 落点全部交给 utils/kline.js：换周期只是把 interval 传进去，标记自然落到新周期的K线上。
function buildTradePoints(rows, interval, orders) {
  if (!rows.length || !orders.length) return []
  const points = []
  for (const order of orders) {
    if (!order) continue
    const entry = locateBar(rows, order.created_at, interval)
    const entryPrice = Number(order.entry_price)
    if (entry && Number.isFinite(entryPrice)) {
      const marker = tradeMarker({ side: order.side })
      points.push({
        id: order.id,
        index: entry.index,
        price: entryPrice,
        marker,
        exit: false,
        text: `${marker.label} ${price(entryPrice)}`,
      })
    }
    if (String(order.status || '').toUpperCase() !== 'CLOSED') continue
    const closePrice = Number(order.close_price)
    if (!Number.isFinite(closePrice) || !closePrice || !order.closed_at) continue
    const exit = locateBar(rows, order.closed_at, interval)
    if (!exit) continue
    const marker = tradeMarker({ side: order.side, action: 'exit' })
    points.push({
      id: order.id,
      index: exit.index,
      price: closePrice,
      marker,
      exit: true,
      text: `${String(order.side || '').toUpperCase() === 'SHORT' ? '平空' : '平多'} ${price(closePrice)}`,
    })
  }
  return points
}

const tradePoints = computed(() =>
  showTrades.value ? buildTradePoints(bars.value, query.interval, trades.value) : [],
)

const tradeHint = computed(() => {
  if (tradesError.value) return `成交标记不可用：${tradesError.value}（K线不受影响）`
  if (tradesLoading.value) return '正在加载成交记录…'
  if (!showTrades.value) return '成交标记已隐藏'
  if (!trades.value.length) return '该币种暂无成交记录'
  return `已叠加 ${trades.value.length} 笔成交`
})

const chartOption = computed(() => {
  const rows = bars.value
  const labels = rows.map((row) => timeShort(row.time))
  const candle = rows.map((row) => [row.open, row.close, row.low, row.high])
  const closes = rows.map((row) => row.close)
  const ma20 = movingAverage(closes, 20)
  const ma60 = movingAverage(closes, 60)
  const points = tradePoints.value
  const series = [
    {
      name: 'K线',
      type: 'candlestick',
      data: candle,
      itemStyle: {
        color: '#34d399',
        color0: '#fb7185',
        borderColor: '#34d399',
        borderColor0: '#fb7185',
      },
    },
    { name: 'MA20', type: 'line', data: ma20, showSymbol: false, lineStyle: { width: 1, color: '#fbbf24' } },
    { name: 'MA60', type: 'line', data: ma60, showSymbol: false, lineStyle: { width: 1, color: '#60a5fa' } },
    {
      name: '成交量',
      type: 'bar',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: rows.map((row) => row.volume),
      itemStyle: { color: 'rgba(96,165,250,0.45)' },
    },
  ]
  // 买卖点单独一条 scatter：不进图例，也不参与 MA 计算。
  if (points.length) {
    series.push({
      name: '我的成交',
      type: 'scatter',
      z: 5,
      data: points.map((point) => ({
        value: [point.index, point.price],
        symbol: point.marker.symbol,
        symbolRotate: point.marker.symbolRotate,
        symbolSize: 9,
        itemStyle: { color: point.marker.color },
        text: point.text,
        label: {
          show: true,
          position: point.exit ? 'bottom' : 'top',
          distance: 4,
          color: point.marker.color,
          fontSize: 9,
          formatter: point.text,
        },
      })),
    })
  }
  return {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: 'rgba(15,20,28,0.96)',
      borderColor: 'rgba(148,163,184,0.2)',
      textStyle: { color: '#e6ecf5', fontSize: 11 },
      formatter: klineTooltip,
    },
    grid: [
      { left: 62, right: 18, top: 26, height: '58%' },
      { left: 62, right: 18, top: '74%', height: '14%' },
    ],
    xAxis: [
      {
        type: 'category',
        data: labels,
        boundaryGap: true,
        axisLine: { lineStyle: { color: 'rgba(148,163,184,0.2)' } },
        axisLabel: { show: false },
        splitLine: { show: false },
      },
      {
        type: 'category',
        gridIndex: 1,
        data: labels,
        boundaryGap: true,
        axisLine: { lineStyle: { color: 'rgba(148,163,184,0.2)' } },
        axisLabel: { color: '#64748b', fontSize: 10, hideOverlap: true },
        splitLine: { show: false },
      },
    ],
    yAxis: [
      {
        scale: true,
        splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } },
        axisLabel: { color: '#64748b', fontSize: 10 },
      },
      {
        scale: true,
        gridIndex: 1,
        splitNumber: 2,
        splitLine: { show: false },
        axisLabel: { color: '#64748b', fontSize: 9 },
      },
    ],
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: [0, 1],
        start: zoomWindow.value.start,
        end: zoomWindow.value.end,
        // 30ms 节流让拖动更跟手；滚轮只缩放不横向平移，避免误触
        throttle: 30,
        zoomOnMouseWheel: true,
        moveOnMouseWheel: false,
        moveOnMouseMove: true,
      },
      {
        type: 'slider',
        xAxisIndex: [0, 1],
        start: zoomWindow.value.start,
        end: zoomWindow.value.end,
        height: 20,
        bottom: 2,
        borderColor: 'transparent',
        backgroundColor: 'rgba(148,163,184,0.08)',
        fillerColor: 'rgba(52,211,153,0.14)',
        handleStyle: { color: '#34d399' },
        textStyle: { color: '#64748b' },
      },
    ],
    series,
  }
})

async function loadSymbols() {
  if (disposed) return
  symbolsLoading.value = true
  symbolsError.value = ''
  try {
    const data = await api.getMarketSymbols()
    if (!disposed) allSymbols.value = data.symbols || []
  } catch (error) {
    if (!disposed) symbolsError.value = error.message
  } finally {
    if (!disposed) symbolsLoading.value = false
  }
}

async function loadSources() {
  if (disposed) return
  const request = ++statsRequest
  const page = statsPage.value
  const pageSize = statsPageSize
  sourcesLoading.value = true
  sourcesError.value = ''
  try {
    const data = await api.getMarketSources({ page, page_size: pageSize })
    if (disposed || request !== statsRequest) return
    const rows = data.local?.stats || []
    const total = data.local?.total ?? rows.length
    const lastPage = Math.max(1, Math.ceil(total / pageSize))
    if (page > lastPage) {
      statsPage.value = lastPage
      return loadSources()
    }
    sources.value = data
    statsTotal.value = total
  } catch (error) {
    if (!disposed && request === statsRequest) sourcesError.value = error.message
  } finally {
    if (!disposed && request === statsRequest) sourcesLoading.value = false
  }
}

function changeStatsPage(page) {
  if (page === statsPage.value) return Promise.resolve()
  statsPage.value = page
  return loadSources()
}

async function loadBars() {
  if (disposed) return
  const request = ++barsRequest
  const symbol = query.symbol
  const interval = query.interval
  const key = `${symbol}:${interval}:${query.limit}`
  if (displayedQuery !== key) bars.value = []
  // 换币种时旧币种的成交点不能留到新图里
  if (tradesSymbol && tradesSymbol !== symbol) trades.value = []
  loading.value = true
  barsError.value = ''
  hasMoreBars.value = true
  zoomWindow.value = { start: 40, end: 100 }
  liveZoom = { start: 40, end: 100 }
  try {
    const data = await api.getKlines(symbol, interval, Number(query.limit))
    if (disposed || request !== barsRequest) return
    bars.value = data.bars || []
    displayedQuery = key
  } catch (error) {
    if (!disposed && request === barsRequest) barsError.value = error.message
  } finally {
    if (!disposed && request === barsRequest) loading.value = false
  }
}

// 只读取当前币种的成交；失败只影响标记，不抛给页面。
async function loadTrades() {
  if (disposed) return []
  const request = ++tradesRequest
  const symbol = query.symbol
  tradesLoading.value = true
  tradesError.value = ''
  try {
    // 只取画标记要用的字段：整条订单（含参数快照）在图上只用到 5 个值
    const data = await api.getOrderMarkers({ symbol, limit: 200 })
    if (disposed || request !== tradesRequest) return []
    const rows = Array.isArray(data?.markers) ? data.markers : []
    const upper = String(symbol).toUpperCase()
    trades.value = rows.filter((order) => !order?.symbol || String(order.symbol).toUpperCase() === upper)
    tradesSymbol = symbol
    return trades.value
  } catch (error) {
    if (!disposed && request === tradesRequest) {
      trades.value = []
      tradesSymbol = symbol
      tradesError.value = error?.message || '成交记录加载失败'
    }
    return []
  } finally {
    if (!disposed && request === tradesRequest) tradesLoading.value = false
  }
}

function tradeAnchors() {
  const anchors = []
  for (const order of trades.value) {
    if (order?.created_at) anchors.push(order.created_at)
    if (order?.closed_at) anchors.push(order.closed_at)
  }
  return anchors
}

// 切换周期后成交跨度会变很多根：按 barsToCover 补更早分页，保证最近成交落在已加载窗口内。
async function ensureTradeCoverage() {
  if (disposed || !showTrades.value || !trades.value.length) return
  const anchors = tradeAnchors()
  if (!anchors.length) return
  const times = anchors.map((value) => Date.parse(value)).filter(Number.isFinite)
  if (!times.length) return
  const oldest = Math.min(...times)
  const target = barsToCover(query.interval, anchors)
  let rounds = 0
  while (!disposed && hasMoreBars.value && !olderLoading.value && rounds < 8) {
    const first = Number(bars.value[0]?.time)
    // 最早一笔成交已经落在窗口内，或者根数够了，就不用再往前翻
    if (!Number.isFinite(first) || oldest >= first || bars.value.length >= target) break
    const before = bars.value.length
    await loadOlderBars({ silent: true })
    if (disposed || bars.value.length === before) break
    rounds += 1
  }
}

function latestOrder() {
  let best = null
  let bestTime = -Infinity
  for (const order of trades.value) {
    const time = Date.parse(order?.created_at || '') || 0
    if (time > bestTime) {
      bestTime = time
      best = order
    }
  }
  return best
}

function latestIndexes(order) {
  if (!order || !bars.value.length) return []
  const found = []
  const entry = locateBar(bars.value, order.created_at, query.interval)
  if (entry) found.push(entry.index)
  if (String(order.status || '').toUpperCase() === 'CLOSED' && order.closed_at) {
    const exit = locateBar(bars.value, order.closed_at, query.interval)
    if (exit) found.push(exit.index)
  }
  return found
}

function olderThanWindow(order) {
  const first = Number(bars.value[0]?.time)
  const time = Date.parse(order?.created_at || '')
  return Number.isFinite(first) && Number.isFinite(time) && time < first
}

// 只有用户点“定位最近成交”才改视窗；日常浏览不打断缩放。
async function locateLatestTrade() {
  if (disposed) return
  locating.value = true
  try {
    if (!trades.value.length && !tradesError.value) await loadTrades()
    if (!trades.value.length) {
      ElMessage.info(tradesError.value ? '成交记录暂不可用，K线不受影响' : '当前币种还没有成交记录')
      return
    }
    const target = latestOrder()
    let rounds = 0
    while (!disposed && olderThanWindow(target) && hasMoreBars.value && !olderLoading.value && rounds < 8) {
      const before = bars.value.length
      await loadOlderBars({ silent: true })
      if (disposed || bars.value.length === before) break
      rounds += 1
    }
    const indexes = latestIndexes(target)
    const zoom = zoomForIndexes(bars.value.length, indexes)
    zoomWindow.value = zoom
    liveZoom = { ...zoom }
    if (!indexes.length) ElMessage.info('最近成交还没落在已加载的K线上，已定位到最新数据')
  } finally {
    if (!disposed) locating.value = false
  }
}

// 拖到最左边时把更早的一批K线拼到前面，并整体右移视窗，视觉上不跳。
async function loadOlderBars(options = {}) {
  const silent = options?.silent === true
  const rows = bars.value
  if (!rows.length || olderLoading.value || loading.value || !hasMoreBars.value) return
  const firstTime = Number(rows[0].time)
  if (!Number.isFinite(firstTime)) return
  const request = barsRequest
  olderLoading.value = true
  try {
    const data = await api.getKlines(query.symbol, query.interval, Number(query.limit), firstTime - 1)
    if (disposed || request !== barsRequest) return
    const older = data.bars || []
    if (!older.length) {
      hasMoreBars.value = false
      if (!silent) ElMessage.info('已经到本地最早的数据了')
      return
    }
    const before = rows.length
    const total = before + older.length
    bars.value = [...older, ...rows]
    const start = ((liveZoom.start / 100) * before + older.length) / total * 100
    const end = ((liveZoom.end / 100) * before + older.length) / total * 100
    zoomWindow.value = { start, end }
    liveZoom = { start, end }
    if (older.length < Number(query.limit)) hasMoreBars.value = false
  } catch (error) {
    if (!disposed && !silent) ElMessage.error(error.message || '加载更早K线失败')
  } finally {
    if (!disposed) olderLoading.value = false
  }
}

function onChartZoom(zoom) {
  liveZoom = zoom
  // 留 3% 余量：inside 拖动很难精确停在 0
  if (zoom.start <= 3) loadOlderBars()
}

async function refreshBars() {
  await loadBars()
  await loadTrades()
  await ensureTradeCoverage()
}

function changeQuery() {
  queryTouched = true
  return refreshBars()
}

function switchInterval(interval) {
  if (query.interval === interval) return Promise.resolve()
  query.interval = interval
  return changeQuery()
}

function onToggleTrades() {
  if (showTrades.value && !trades.value.length && !tradesLoading.value) loadTrades()
}

function touchSyncForm() {
  syncFormTouched = true
}

async function sync() {
  if (!settingsReady.value || syncing.value || disposed) return
  syncing.value = true
  try {
    const payload = {
      source: syncForm.source,
      interval: syncForm.interval,
      days: Number(syncForm.days),
      symbols: scopeAll.value ? [] : syncForm.symbols.map((item) => String(item).trim().toUpperCase()),
    }
    const result = await api.syncMarket(payload)
    if (disposed) return
    const count = result.imported ?? result.inserted ?? 0
    ElMessage.success(`导入完成：${result.symbol ? `${result.symbol} · ` : ''}${count} 根K线`)
    loadSources()
    loadBars()
    loadTrades()
    loadSymbols()
  } catch (error) {
    if (!disposed) ElMessage.error(error.message)
  } finally {
    if (!disposed) syncing.value = false
  }
}

async function loadSettings() {
  if (disposed) return
  settingsLoading.value = true
  settingsError.value = ''
  try {
    const config = await api.getSettings()
    if (disposed) return
    settings.value = config
    if (!queryTouched) {
      const symbol = config.symbols?.[0] || 'BTCUSDT'
      const interval = config.interval || '1m'
      if (query.symbol !== symbol || query.interval !== interval) {
        query.symbol = symbol
        query.interval = interval
        loadBars()
        loadTrades()
      }
    }
    if (!syncFormTouched) {
      syncForm.symbols = config.symbols?.length ? [...config.symbols] : ['BTCUSDT']
      syncForm.interval = config.interval || '1m'
      syncForm.source = config.data_source || 'binance'
    }
    settingsReady.value = true
  } catch (error) {
    if (!disposed) settingsError.value = error.message
  } finally {
    if (!disposed) settingsLoading.value = false
  }
}

function init() {
  return Promise.allSettled([loadSources(), loadSymbols(), loadBars(), loadSettings(), loadTrades()])
}

onMounted(init)
onBeforeUnmount(() => {
  disposed = true
  barsRequest += 1
  statsRequest += 1
  tradesRequest += 1
})
</script>

<template>
  <div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">同步币安K线</div>
            <div class="panel-sub">只获取 1m 原始K线；5m / 15m / 1h / 4h / 1d 在本地聚合</div>
          </div>
        </div>
        <div v-if="settingsLoading" class="muted" role="status">正在加载同步设置，行情与本地数据可独立查看…</div>
        <div v-else-if="settingsError" class="form-hint" role="alert">
          同步设置加载失败：{{ settingsError }}
          <el-button size="small" text @click="loadSettings">重试</el-button>
        </div>
        <div class="form-grid">
          <div>
            <div class="muted" style="margin-bottom: 4px">来源</div>
            <el-select v-model="syncForm.source" style="width: 100%" @change="touchSyncForm">
              <el-option label="币安公共接口" value="binance" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">币种（自定义列表生效时）</div>
            <el-select v-model="syncForm.symbols" multiple filterable :disabled="scopeAll" style="width: 100%" @change="touchSyncForm"
              :placeholder="scopeAll ? '全量范围：忽略此选择' : '选择要同步的合约'">
              <el-option v-for="symbol in symbolOptions" :key="symbol" :label="symbol" :value="symbol" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">K线周期</div>
            <el-select v-model="syncForm.interval" style="width: 100%" @change="touchSyncForm">
              <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">天数</div>
            <el-input-number v-model="syncForm.days" :min="1" :max="365" style="width: 100%" @change="touchSyncForm" />
          </div>
        </div>
        <div class="row" style="margin-top: 12px">
          <el-button type="primary" :loading="syncing" :disabled="!settingsReady" @click="sync">导入 / 同步</el-button>
        </div>
        <div class="form-hint" style="margin-top: 6px">{{ syncHint }}</div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">本地数据</div>
            <div class="panel-sub">1m 原始K线 · 其他周期由原始数据生成</div>
          </div>
          <span class="muted" role="status">{{ sourcesLoading ? '加载中…' : `${statsTotal} 组` }}</span>
        </div>
        <div v-if="sourcesError" class="form-hint" role="alert">
          本地数据加载失败：{{ sourcesError }}
          <el-button size="small" text @click="loadSources">重试</el-button>
        </div>
        <el-table v-loading="sourcesLoading" :data="localStats" class="table-compact" max-height="240" :empty-text="sourcesLoading ? '正在加载本地数据…' : '本地暂无K线'">
          <el-table-column prop="symbol" label="币种" width="96" />
          <el-table-column prop="interval" label="周期" width="66" />
          <el-table-column label="行数" width="90"><template #default="{ row }"><span class="mono">{{ number(row.count, 0) }}</span></template></el-table-column>
          <el-table-column label="起止" min-width="200">
            <template #default="{ row }">
              <div class="muted">{{ tsLabel(row.first_open_time) }}</div>
              <div class="muted">{{ tsLabel(row.last_open_time) }}</div>
            </template>
          </el-table-column>
        </el-table>
        <TablePager :page="statsPage" :page-size="statsPageSize" :total="statsTotal" :disabled="sourcesLoading" @change="changeStatsPage" />
      </div>
    </div>

    <div class="panel section-gap">
      <div class="panel-head">
        <div>
          <div class="panel-title">{{ query.symbol }} · {{ query.interval }}</div>
          <div class="panel-sub" role="status">{{ loading ? 'K线加载中… · ' : '' }}{{ olderLoading ? '加载更早K线… · ' : '' }}{{ bars.length }} 根K线 · MA20 / MA60 · 成交量 · 向左拖动自动加载更早历史<span v-if="bars.length && !hasMoreBars"> · 已到本地最早数据</span></div>
        </div>
        <div class="row">
          <el-button-group>
            <el-button
              v-for="item in quickIntervals" :key="item" size="small"
              :type="query.interval === item ? 'primary' : ''" @click="switchInterval(item)"
            >{{ item }}</el-button>
          </el-button-group>
          <el-select v-model="query.symbol" size="small" filterable :loading="symbolsLoading" style="width: 150px" @change="changeQuery">
            <el-option v-for="symbol in symbolOptions" :key="symbol" :label="symbol" :value="symbol" />
          </el-select>
          <el-select v-model="query.interval" size="small" style="width: 100px" @change="changeQuery">
            <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
          </el-select>
          <el-select v-model="query.limit" size="small" style="width: 110px" @change="changeQuery">
            <el-option label="300 根" :value="300" />
            <el-option label="600 根" :value="600" />
            <el-option label="1200 根" :value="1200" />
          </el-select>
          <el-button size="small" @click="refreshBars">刷新</el-button>
        </div>
      </div>
      <div class="row" style="margin-bottom: 8px">
        <el-switch v-model="showTrades" size="small" active-text="显示我的成交" @change="onToggleTrades" />
        <el-button size="small" :loading="locating" @click="locateLatestTrade">定位最近成交</el-button>
        <span class="muted" style="font-size: 11px" role="status">{{ tradeHint }}</span>
        <el-button v-if="tradesError" size="small" text @click="loadTrades">重试</el-button>
      </div>
      <div v-if="symbolsError" class="form-hint" role="alert">
        币种列表加载失败，暂用已有币种。
        <el-button size="small" text @click="loadSymbols">重试</el-button>
      </div>
      <div v-if="barsError" class="form-hint" role="alert">
        K线加载失败：{{ barsError }}
        <el-button size="small" text @click="loadBars">重试</el-button>
      </div>
      <EChart v-if="bars.length" :option="chartOption" height="470px" @datazoom="onChartZoom" />
      <div v-else class="empty-hint" style="min-height: 220px" role="status">{{ loading ? '正在加载K线…' : barsError ? 'K线暂不可用，请重试。' : '还没有K线。使用上方“导入 / 同步”从币安同步。' }}</div>
    </div>
  </div>
</template>
