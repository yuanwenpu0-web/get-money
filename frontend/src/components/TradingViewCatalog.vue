<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import api from '../api'

const items = ref([])
const loading = ref(false)
const error = ref('')
const search = ref('')
const page = ref(1)
const detail = ref(null)
const detailVisible = ref(false)
const detailLoading = ref(false)
const detailError = ref('')
const backtestForm = ref({ symbol: 'BTCUSDT', interval: '15m', days: 30 })
const backtestRunning = ref(false)
const backtestError = ref('')
const backtestResult = ref(null)
const backtestContext = ref(null)
const pageSize = 20
let request = 0
let backtestRequest = 0
let disposed = false

const filtered = computed(() => {
  const query = search.value.trim().toLocaleLowerCase()
  return items.value.filter((row) => !query || [row.title, row.path].some((v) => String(v || '').toLocaleLowerCase().includes(query)))
})
const displayed = computed(() => filtered.value.slice((page.value - 1) * pageSize, page.value * pageSize))
watch(search, () => { page.value = 1 })

async function load() {
  if (loading.value) return
  loading.value = true
  error.value = ''
  try {
    const data = await api.getTradingViewStrategies()
    if (!disposed) {
      items.value = data.strategies || []
      page.value = Math.min(page.value, Math.max(1, Math.ceil(filtered.value.length / pageSize)))
    }
  } catch (err) {
    if (!disposed) error.value = err.message
  } finally {
    if (!disposed) loading.value = false
  }
}

async function showDetail(row) {
  const current = ++request
  backtestRequest++
  backtestRunning.value = false
  backtestError.value = ''
  backtestResult.value = null
  backtestContext.value = null
  detail.value = row
  detailVisible.value = true
  detailLoading.value = true
  detailError.value = ''
  try {
    const data = await api.getTradingViewStrategy(row.id)
    if (!disposed && current === request) detail.value = data
  } catch (err) {
    if (!disposed && current === request) detailError.value = err.message
  } finally {
    if (!disposed && current === request) detailLoading.value = false
  }
}

function closeDetail() { request++; backtestRequest++; detailLoading.value = false; backtestRunning.value = false }

function supportsResearch(row) {
  return ['entry_filter_and_research', 'research_backtest'].includes(row?.execution_status)
}

async function runBacktest() {
  if (disposed || backtestRunning.value || detailLoading.value || !supportsResearch(detail.value)) return
  const current = ++backtestRequest
  const id = detail.value.id
  backtestRunning.value = true
  backtestError.value = ''
  backtestResult.value = null
  try {
    const payload = { symbol: backtestForm.value.symbol.trim().toUpperCase(), interval: backtestForm.value.interval, days: Number(backtestForm.value.days) }
    const result = await api.runTradingViewBacktest(id, payload)
    if (!disposed && current === backtestRequest) {
      backtestContext.value = payload
      backtestResult.value = result
    }
  } catch (err) {
    if (!disposed && current === backtestRequest) backtestError.value = err.message
  } finally {
    if (!disposed && current === backtestRequest) backtestRunning.value = false
  }
}

function shown(value, suffix = '') {
  return value == null || !Number.isFinite(Number(value)) ? '—' : `${Number(value).toFixed(2)}${suffix}`
}
function shownPrice(value) {
  return value == null || !Number.isFinite(Number(value)) ? '—' : String(Number(Number(value).toPrecision(8)))
}
function shownRate(value) {
  return value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(4)}%`
}
onMounted(load)
onBeforeUnmount(() => { disposed = true; request++; backtestRequest++ })
</script>

<template>
  <div class="panel catalog">
    <div class="panel-head">
      <div>
        <div class="panel-title">TradingView 策略源码目录</div>
        <div class="panel-sub">查看 data 中的 Pine 策略源码。源码收录不表示已迁移或可交易；已支持的入场过滤与研究回测也不等同于完整执行原策略。</div>
      </div>
      <el-button size="small" :loading="loading" @click="load">刷新</el-button>
    </div>
    <div class="row">
      <el-input v-model="search" clearable placeholder="搜索策略名称或源码路径" aria-label="搜索 TradingView 策略" style="max-width: 360px" />
      <span class="muted">{{ filtered.length }} / {{ items.length }} 个策略</span>
    </div>
    <div v-if="error" class="row section-gap" role="alert">
      <span class="down">策略目录加载失败：{{ error }}</span>
      <el-button size="small" @click="load">重试</el-button>
    </div>
    <div v-if="loading && !items.length" class="empty-hint" role="status">正在加载策略目录…</div>
    <div v-else-if="!displayed.length && !error" class="empty-hint">{{ items.length ? '没有符合条件的策略' : '暂无策略源码' }}</div>
    <el-table v-else-if="displayed.length" :data="displayed" class="table-compact section-gap" @row-click="showDetail">
      <el-table-column label="策略" min-width="200">
        <template #default="{ row }"><button class="catalog-title" type="button" @click.stop="showDetail(row)">{{ row.title }}</button></template>
      </el-table-column>
      <el-table-column prop="path" label="源码路径" min-width="270" show-overflow-tooltip />
      <el-table-column label="Pine" width="70"><template #default="{ row }">{{ row.version ? `v${row.version}` : '—' }}</template></el-table-column>
      <el-table-column label="本地状态" min-width="180">
        <template #default="{ row }"><el-tag size="small" effect="plain" :type="row.execution_status === 'not_migrated' ? 'info' : 'warning'">{{ row.execution_status === 'not_migrated' ? '仅源码，尚未迁移' : row.execution_status === 'entry_filter_and_research' ? '支持入场过滤及研究回测' : '支持研究回测' }}</el-tag></template>
      </el-table-column>
      <el-table-column label="操作" width="80"><template #default="{ row }"><el-button link type="primary" @click.stop="showDetail(row)">查看</el-button></template></el-table-column>
    </el-table>
    <div v-if="filtered.length > pageSize" class="table-pagination"><el-pagination v-model:current-page="page" :page-size="pageSize" :total="filtered.length" layout="total, prev, pager, next" /></div>
    <el-dialog v-model="detailVisible" :title="detail?.title || '策略源码'" width="min(900px, 92vw)" @close="closeDetail">
      <template v-if="detail">
        <p class="muted mono">{{ detail.path }}</p>
        <p class="muted">{{ detail.execution_status === 'not_migrated' ? '仅收录源码，尚未迁移为本地可执行策略。' : detail.execution_status === 'entry_filter_and_research' ? '支持入场过滤及研究回测；不支持原脚本完整自动交易。' : '支持研究回测；不支持原脚本完整自动交易。' }}</p>
        <p v-if="detail.source_sha256" class="muted mono">SHA-256: {{ detail.source_sha256 }}</p>
        <div v-if="detailLoading" class="empty-hint" role="status">正在加载源码…</div>
        <div v-else-if="detailError" class="row" role="alert"><span class="down">源码加载失败：{{ detailError }}</span><el-button size="small" @click="showDetail(detail)">重试</el-button></div>
        <template v-else>
          <div v-if="supportsResearch(detail)" class="research-backtest">
            <strong>原策略研究回测</strong>
            <p class="muted">使用本地历史行情与原脚本默认参数模拟；结果不等于绑定系统策略后的交易，也不会自动下单。</p>
            <div class="row">
              <el-input v-model="backtestForm.symbol" aria-label="回测币种" placeholder="BTCUSDT" :disabled="backtestRunning" style="width: 140px" />
              <el-select v-model="backtestForm.interval" aria-label="回测周期" :disabled="backtestRunning" style="width: 100px">
                <el-option v-for="value in ['1m', '5m', '15m', '1h', '4h', '1d']" :key="value" :label="value" :value="value" />
              </el-select>
              <el-input-number v-model="backtestForm.days" aria-label="回测天数" :disabled="backtestRunning" :min="1" :max="180" :step="1" :precision="0" />
              <span class="muted">天</span>
              <el-button type="primary" size="small" :loading="backtestRunning" @click="runBacktest">运行回测</el-button>
            </div>
            <p v-if="backtestError" class="down" role="alert">回测失败：{{ backtestError }}</p>
            <template v-if="backtestResult">
              <p class="muted">{{ backtestContext?.symbol }} · {{ backtestContext?.interval }} · {{ backtestContext?.days }} 天</p>
              <div class="row section-gap" role="status">
                <span>期末权益 {{ shown(backtestResult.metrics?.final_equity) }}</span>
                <span>收益 {{ shown(backtestResult.metrics?.total_return_pct, '%') }}</span>
                <span>最大回撤 {{ shown(backtestResult.metrics?.max_drawdown_pct, '%') }}</span>
                <span>已平仓 {{ backtestResult.metrics?.total_trades ?? '—' }} 笔</span>
                <span>胜率 {{ shown(backtestResult.metrics?.win_rate, '%') }}</span>
              </div>
              <p class="muted">{{ backtestResult.metrics?.start }} — {{ backtestResult.metrics?.end }}</p>
              <p class="muted">初始资金 {{ shown(backtestResult.metrics?.initial_cash) }} · 仓位 {{ shownRate(backtestResult.metrics?.position_pct) }} · 单边手续费 {{ shownRate(backtestResult.metrics?.commission_rate) }}</p>
              <p v-if="backtestResult.open_position" class="muted">期末仍有持仓，权益包含未平仓盈亏。</p>
              <p v-for="assumption in (Array.isArray(backtestResult.assumptions) ? backtestResult.assumptions : [])" :key="assumption" class="muted">{{ assumption }}</p>
              <el-table v-if="backtestResult.trades?.length" :data="backtestResult.trades" height="220" class="table-compact">
                <el-table-column prop="entry_time" label="开仓时间" min-width="170" />
                <el-table-column prop="exit_time" label="平仓时间" min-width="170" />
                <el-table-column label="方向" width="65"><template #default="{ row }">{{ row.direction === 1 ? '多' : row.direction === -1 ? '空' : row.side }}</template></el-table-column>
                <el-table-column label="开仓价" width="105"><template #default="{ row }">{{ shownPrice(row.entry_price) }}</template></el-table-column>
                <el-table-column label="平仓价" width="105"><template #default="{ row }">{{ shownPrice(row.exit_price) }}</template></el-table-column>
                <el-table-column prop="quantity" label="数量" width="110" />
                <el-table-column label="盈亏" width="100"><template #default="{ row }">{{ shown(row.pnl) }}</template></el-table-column>
                <el-table-column label="手续费" width="100"><template #default="{ row }">{{ shown(row.fees) }}</template></el-table-column>
              </el-table>
            </template>
          </div>
          <pre class="catalog-source mono"><code>{{ detail.source }}</code></pre>
        </template>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.catalog { margin-bottom: 14px; }
.catalog-title { border: 0; padding: 0; color: #e6ecf5; background: none; text-align: left; cursor: pointer; font: inherit; font-weight: 600; }
.catalog-title:hover, .catalog-title:focus-visible { color: #34d399; text-decoration: underline; }
.catalog-source { max-height: 55vh; overflow: auto; padding: 14px; border: 1px solid rgba(148, 163, 184, .14); border-radius: 8px; background: #080b11; color: #cbd5e1; font-size: 11px; line-height: 1.6; white-space: pre; }
.research-backtest { margin: 12px 0; padding: 12px; border: 1px solid rgba(148, 163, 184, .14); border-radius: 8px; }
</style>
