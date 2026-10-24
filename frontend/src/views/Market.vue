<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import { number, tsLabel } from '../utils/format'

const settings = ref({})
const sources = ref(null)
const bars = ref([])
const loading = ref(false)
const syncing = ref(false)

const query = reactive({ symbol: 'BTCUSDT', interval: '1m', limit: 600 })
const syncForm = reactive({ source: 'nofx', symbol: 'BTCUSDT', interval: '1m', days: 0 })

const localStats = computed(() => sources.value?.local?.stats || [])
const nofx = computed(() => sources.value?.nofx || {})
const nofxItems = computed(() => nofx.value.items || [])

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
    const payload = { ...syncForm, days: Number(syncForm.days) }
    const result = await api.syncMarket(payload)
    const count = result.imported ?? result.inserted ?? 0
    ElMessage.success(`导入完成：${result.symbol} ${result.interval} ${count} 根`)
    await Promise.all([loadSources(), loadBars()])
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
    syncForm.symbol = query.symbol
    syncForm.interval = query.interval
    syncForm.source = settings.value.data_source || 'nofx'
  } catch (error) {
    /* 使用默认 */
  }
  await loadSources()
  await loadBars()
}

onMounted(init)
</script>

<template>
  <div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">从已有项目导入K线</div>
            <div class="panel-sub">nofx-new 的 PostgreSQL 只读导入到本地 SQLite，不再依赖币安网络</div>
          </div>
          <el-tag size="small" effect="plain" :type="nofx.available ? 'success' : 'danger'">
            {{ nofx.available ? 'nofx 在线' : 'nofx 离线' }}
          </el-tag>
        </div>
        <div v-if="nofx.available" class="row" style="margin-bottom: 12px">
          <el-tag v-for="item in nofxItems" :key="`${item.symbol}-${item.interval}`" size="small" effect="plain" type="info">
            {{ item.symbol }} {{ item.interval }} · {{ item.count }} 根
          </el-tag>
          <span class="muted">共 {{ nofx.symbol_count }} 个币种 · {{ (nofx.intervals || []).join(' / ') }}</span>
        </div>
        <div v-else class="form-hint">{{ nofx.error || '无法连接 nofx_lite 数据库' }}</div>
        <div class="form-grid">
          <div>
            <div class="muted" style="margin-bottom: 4px">来源</div>
            <el-select v-model="syncForm.source" style="width: 100%">
              <el-option label="nofx 本地库" value="nofx" />
              <el-option label="币安公共接口" value="binance" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">币种</div>
            <el-select v-model="syncForm.symbol" style="width: 100%">
              <el-option v-for="symbol in settings.symbols || ['BTCUSDT', 'ETHUSDT']" :key="symbol" :label="symbol" :value="symbol" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">K线周期</div>
            <el-select v-model="syncForm.interval" style="width: 100%">
              <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">天数（0 = 全部）</div>
            <el-input-number v-model="syncForm.days" :min="0" :max="365" style="width: 100%" />
          </div>
        </div>
        <div class="row" style="margin-top: 12px">
          <el-button type="primary" :loading="syncing" @click="sync">导入 / 同步</el-button>
          <span class="muted">导入按 (symbol, interval, open_time) 幂等覆盖</span>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">本地数据</div>
            <div class="panel-sub">SQLite · WAL · 唯一索引</div>
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
          <el-select v-model="query.symbol" size="small" style="width: 130px" @change="loadBars">
            <el-option v-for="symbol in settings.symbols || ['BTCUSDT', 'ETHUSDT']" :key="symbol" :label="symbol" :value="symbol" />
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
      <div v-else class="empty-hint">还没有K线。使用上方“导入 / 同步”从 nofx 导入。</div>
    </div>
  </div>
</template>
