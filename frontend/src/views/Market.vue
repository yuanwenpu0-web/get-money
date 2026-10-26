<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
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
  try {
    const data = await api.getMarketSymbols()
    allSymbols.value = data.symbols || []
  } catch (error) {
    /* 币种列表接口不可用时回退到设置中的交易对 */
  }
}

async function loadSources() {
  try {
    sources.value = await api.getMarketSources()
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function loadBars() {
  loading.value = true
  try {
    const data = await api.getKlines(query.symbol, query.interval, Number(query.limit))
    bars.value = data.bars || []
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

async function sync() {
  syncing.value = true
  try {
    const payload = {
      source: syncForm.source,
      interval: syncForm.interval,
      days: Number(syncForm.days),
      symbols: scopeAll.value ? [] : syncForm.symbols.map((item) => String(item).trim().toUpperCase()),
    }
    const result = await api.syncMarket(payload)
    const count = result.imported ?? result.inserted ?? 0
    ElMessage.success(`导入完成：${result.symbol ? `${result.symbol} · ` : ''}${count} 根K线`)
    await Promise.all([loadSources(), loadBars(), loadSymbols()])
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    syncing.value = false
  }
}

async function init() {
  try {
    settings.value = await api.getSettings()
    query.symbol = settings.value.symbols?.[0] || 'BTCUSDT'
    query.interval = settings.value.interval || '1m'
    syncForm.symbols = settings.value.symbols?.length ? [...settings.value.symbols] : ['BTCUSDT']
    syncForm.interval = query.interval
    syncForm.source = settings.value.data_source || 'binance'
  } catch (error) {
    /* 使用默认 */
  }
  await Promise.all([loadSources(), loadSymbols(), loadBars()])
}

onMounted(init)
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
        <div class="form-grid">
          <div>
            <div class="muted" style="margin-bottom: 4px">来源</div>
            <el-select v-model="syncForm.source" style="width: 100%">
              <el-option label="币安公共接口" value="binance" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">币种（自定义列表生效时）</div>
            <el-select v-model="syncForm.symbols" multiple filterable :disabled="scopeAll" style="width: 100%"
              :placeholder="scopeAll ? '全量范围：忽略此选择' : '选择要同步的合约'">
              <el-option v-for="symbol in symbolOptions" :key="symbol" :label="symbol" :value="symbol" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">K线周期</div>
            <el-select v-model="syncForm.interval" style="width: 100%">
              <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">天数</div>
            <el-input-number v-model="syncForm.days" :min="1" :max="365" style="width: 100%" />
          </div>
        </div>
        <div class="row" style="margin-top: 12px">
          <el-button type="primary" :loading="syncing" @click="sync">导入 / 同步</el-button>
        </div>
        <div class="form-hint" style="margin-top: 6px">{{ syncHint }}</div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">本地数据</div>
            <div class="panel-sub">1m 原始K线 · 其他周期由原始数据生成</div>
          </div>
          <span class="muted">{{ localStats.length }} 组</span>
        </div>
        <el-table :data="localStats" class="table-compact" max-height="240" empty-text="本地暂无K线">
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
      </div>
    </div>

    <div class="panel section-gap" v-loading="loading">
      <div class="panel-head">
        <div>
          <div class="panel-title">{{ query.symbol }} · {{ query.interval }}</div>
          <div class="panel-sub">{{ bars.length }} 根K线 · MA20 / MA60 · 成交量</div>
        </div>
        <div class="row">
          <el-select v-model="query.symbol" size="small" filterable style="width: 150px" @change="loadBars">
            <el-option v-for="symbol in symbolOptions" :key="symbol" :label="symbol" :value="symbol" />
          </el-select>
          <el-select v-model="query.interval" size="small" style="width: 100px" @change="loadBars">
            <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
          </el-select>
          <el-select v-model="query.limit" size="small" style="width: 110px" @change="loadBars">
            <el-option label="300 根" :value="300" />
            <el-option label="600 根" :value="600" />
            <el-option label="1200 根" :value="1200" />
          </el-select>
          <el-button size="small" @click="loadBars">刷新</el-button>
        </div>
      </div>
      <EChart v-if="bars.length" :option="chartOption" height="470px" />
      <div v-else class="empty-hint">还没有K线。使用上方“导入 / 同步”从币安同步。</div>
    </div>
  </div>
</template>
