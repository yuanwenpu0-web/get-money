<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import { number, tsLabel } from '../utils/format'

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
const statsPage = ref(1)
const statsPageSize = ref(20)
const statsTotal = ref(0)
let statsRequest = 0
let disposed = false
let barsRequest = 0
let displayedQuery = ''
let queryTouched = false
let syncFormTouched = false

const query = reactive({ symbol: 'BTCUSDT', interval: '1m', limit: 600 })
const syncForm = reactive({ source: 'binance', symbols: ['BTCUSDT'], interval: '1m', days: 30 })

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

const chartOption = computed(() => {
  const rows = bars.value
  const labels = rows.map((row) => timeShort(row.time))
  const candle = rows.map((row) => [row.open, row.close, row.low, row.high])
  const closes = rows.map((row) => row.close)
  const ma20 = movingAverage(closes, 20)
  const ma60 = movingAverage(closes, 60)
  return {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: 'rgba(15,20,28,0.96)',
      borderColor: 'rgba(148,163,184,0.2)',
      textStyle: { color: '#e6ecf5', fontSize: 11 },
    },
    grid: [
      { left: 62, right: 18, top: 20, height: '58%' },
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
      { type: 'inside', xAxisIndex: [0, 1], start: 40, end: 100 },
      {
        type: 'slider',
        xAxisIndex: [0, 1],
        height: 14,
        bottom: 4,
        borderColor: 'transparent',
        backgroundColor: 'rgba(148,163,184,0.08)',
        fillerColor: 'rgba(52,211,153,0.14)',
        handleStyle: { color: '#34d399' },
        textStyle: { color: '#64748b' },
      },
    ],
    series: [
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
    ],
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
  const pageSize = statsPageSize.value
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

function changeStatsPageSize(size) {
  if (size === statsPageSize.value) return Promise.resolve()
  statsPageSize.value = size
  statsPage.value = 1
  return loadSources()
}

async function loadBars() {
  if (disposed) return
  const request = ++barsRequest
  const symbol = query.symbol
  const interval = query.interval
  const key = `${symbol}:${interval}:${query.limit}`
  if (displayedQuery !== key) bars.value = []
  loading.value = true
  barsError.value = ''
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

function changeQuery() {
  queryTouched = true
  return loadBars()
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
  return Promise.allSettled([loadSources(), loadSymbols(), loadBars(), loadSettings()])
}

onMounted(init)
onBeforeUnmount(() => {
  disposed = true
  barsRequest += 1
  statsRequest += 1
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
        <div class="table-pagination">
          <el-pagination
            :current-page="statsPage" :page-size="statsPageSize" :total="statsTotal"
            :page-sizes="[10, 20, 50, 100]" layout="total, sizes, prev, pager, next, jumper" size="small"
            @update:current-page="changeStatsPage" @update:page-size="changeStatsPageSize"
          />
        </div>
      </div>
    </div>

    <div class="panel section-gap">
      <div class="panel-head">
        <div>
          <div class="panel-title">{{ query.symbol }} · {{ query.interval }}</div>
          <div class="panel-sub" role="status">{{ loading ? 'K线加载中… · ' : '' }}{{ bars.length }} 根K线 · MA20 / MA60 · 成交量</div>
        </div>
        <div class="row">
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
          <el-button size="small" @click="loadBars">刷新</el-button>
        </div>
      </div>
      <div v-if="symbolsError" class="form-hint" role="alert">
        币种列表加载失败，暂用已有币种。
        <el-button size="small" text @click="loadSymbols">重试</el-button>
      </div>
      <div v-if="barsError" class="form-hint" role="alert">
        K线加载失败：{{ barsError }}
        <el-button size="small" text @click="loadBars">重试</el-button>
      </div>
      <EChart v-if="bars.length" :option="chartOption" height="470px" />
      <div v-else class="empty-hint" style="min-height: 220px" role="status">{{ loading ? '正在加载K线…' : barsError ? 'K线暂不可用，请重试。' : '还没有K线。使用上方“导入 / 同步”从币安同步。' }}</div>
    </div>
  </div>
</template>
