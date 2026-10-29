<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { number, pct, pnlClass, tsLabel } from '../utils/format'

const VERDICTS = {
  pass: { label: '通过', type: 'success', hint: '验证段每笔期望 95% CI 下界 > 0' },
  inconclusive: { label: '不可判定', type: 'warning', hint: '验证段 CI 跨 0，样本不足以判断有没有 edge' },
  negative: { label: '负收益', type: 'danger', hint: '验证段每笔期望 95% CI 上界 < 0' },
  insufficient: { label: '样本不足', type: 'info', hint: '验证段成交笔数不达门槛，不下结论' },
  no_trades: { label: '无成交', type: 'info', hint: '训练段没有任何组合达到成交门槛' },
}

const options = ref({ specs: [], pools: [], default_workers: 8, max_workers: 24, busy: false })
const runs = ref([])
const activeId = ref(null)
const run = ref(null)
const results = ref([])
const resultsTotal = ref(0)
const resultsLoading = ref(false)
const resultsError = ref('')
const starting = ref(false)
const cancelling = ref(false)
const applying = ref(false)
const loadError = ref('')
const optionsLoaded = ref(false)
const coverage = reactive({ total: 0, sources: {}, items: [] })
const coverageVisible = ref(false)
const coverageLoading = ref(false)
const universe = ref(null)
const universeLoading = ref(false)

const page = reactive({ page: 1, pageSize: 30 })
const filters = reactive({ verdict: '', q: '' })
const applyMode = ref('symbol')
const applyOnly = ref('positive')
const poolPrefix = ref('')

const form = reactive({
  strategy: 'absorption_reversal',
  interval: '15m',
  scope: 'all',       // all = 该周期全部有历史的币种；pool = 某个池；symbols = 手填
  pool: '',
  symbols: '',
  days: 90,
  combos: 64,
  workers: 0,
  minTrades: 5,
  trainRatio: 0.7,
  objective: 'return_dd',
  deltaSource: 'auto',
})

let pollTimer = null
let disposed = false

const spec = computed(() => options.value.specs.find((item) => item.code === form.strategy) || null)
const gridRows = computed(() => {
  if (!spec.value) return []
  return Object.entries(spec.value.grid || {}).map(([key, values]) => ({ key, values }))
})
const running = computed(() => ['pending', 'running'].includes(run.value?.status))
const busy = computed(() => running.value || options.value.busy)
const progress = computed(() => run.value?.progress || {})
const summary = computed(() => run.value?.summary || {})
const progressPct = computed(() => {
  const total = Number(progress.value.total || 0)
  const done = Number(progress.value.done || 0)
  if (!total) return 0
  return Math.min(100, Math.round((done / total) * 100))
})
const etaText = computed(() => {
  const eta = progress.value.eta
  if (eta == null) return '—'
  if (eta < 60) return `${Math.round(eta)} 秒`
  return `${Math.round(eta / 60)} 分`
})
const verdictCounts = computed(() => {
  const counts = progress.value.verdicts || summary.value.verdicts || {}
  return Object.entries(VERDICTS)
    .map(([key, meta]) => ({ key, label: meta.label, type: meta.type, count: counts[key] || 0 }))
    .filter((item) => item.count > 0)
})
const pooled = computed(() => summary.value.pooled_ci || null)
const canApply = computed(() => !!run.value && run.value.status === 'done')
const selectedSymbols = computed(() => results.value.map((row) => row.symbol))

function verdictMeta(verdict) {
  return VERDICTS[verdict] || { label: verdict || '—', type: 'info', hint: '' }
}

function buildStartPayload() {
  const payload = {
    strategy: form.strategy,
    interval: form.interval,
    days: Number(form.days),
    combos: Number(form.combos),
    workers: Number(form.workers),
    min_trades: Number(form.minTrades),
    train_ratio: Number(form.trainRatio),
    objective: form.objective,
    delta_source: form.deltaSource,
  }
  if (form.scope === 'pool') {
    if (!form.pool) throw new Error('请选择币种池')
    payload.pool = form.pool
  } else if (form.scope === 'symbols') {
    const symbols = form.symbols
      .split(/[,，\s]+/)
      .map((item) => item.trim().toUpperCase())
      .filter(Boolean)
    if (!symbols.length) throw new Error('请填写至少一个币种')
    payload.symbols = symbols
  }
  return payload
}

function buildApplyPayload(symbols = null) {
  const payload = { mode: applyMode.value, only: applyOnly.value }
  if (symbols?.length) payload.symbols = symbols
  if (applyMode.value === 'pool' && poolPrefix.value.trim()) {
    payload.pool_prefix = poolPrefix.value.trim()
  }
  return payload
}

function paramsText(params) {
  const entries = Object.entries(params || {})
  if (!entries.length) return '沿用当前参数'
  return entries.map(([key, value]) => `${key}=${value}`).join(' · ')
}

function applyHint() {
  if (applyOnly.value === 'pass') return '只应用「验证段 CI 下界 > 0」的币种——最严格，通常也最少。'
  if (applyOnly.value === 'positive') return '应用「验证段有成交且净收益为正」的币种：不要求统计显著，但至少不是负的。'
  return '应用全部有参数覆盖的币种，包括不可判定的——只在你自己有额外判断时用。'
}

async function loadOptions() {
  if (disposed) return
  try {
    const data = await api.getOptimizationOptions()
    if (disposed) return
    options.value = data
    optionsLoaded.value = true
    loadError.value = ''
    if (!data.specs.some((item) => item.code === form.strategy) && data.specs.length) {
      form.strategy = data.specs[0].code
    }
  } catch (error) {
    if (!disposed) loadError.value = error.message
  }
}

async function loadRuns() {
  if (disposed) return
  try {
    const data = await api.listOptimizations({ page: 1, page_size: 20 })
    if (disposed) return
    runs.value = data.optimizations || []
  } catch (error) {
    if (!disposed) loadError.value = error.message
  }
}

async function loadCoverage() {
  if (disposed) return
  coverageLoading.value = true
  try {
    const data = await api.getSymbolParams({})
    if (disposed) return
    coverage.total = data.total || 0
    coverage.sources = data.sources || {}
    coverage.items = data.items || []
  } catch (error) {
    if (!disposed) ElMessage.error(`覆盖表读取失败：${error.message}`)
  } finally {
    if (!disposed) coverageLoading.value = false
  }
}

async function previewUniverse() {
  if (disposed) return
  universeLoading.value = true
  try {
    const data = await api.getOptimizationUniverse({ interval: form.interval, days: Number(form.days) })
    if (!disposed) universe.value = data
  } catch {
    // 预览失败不阻塞寻优，只是少一个「会扫多少币」的提示
    if (!disposed) universe.value = null
  } finally {
    if (!disposed) universeLoading.value = false
  }
}

async function startOptimization() {
  if (starting.value) return
  let payload
  try {
    payload = buildStartPayload()
  } catch (error) {
    ElMessage.warning(error.message)
    return
  }
  starting.value = true
  try {
    const data = await api.createOptimization(payload)
    ElMessage.success(
      `已启动：${data.symbol_count} 个币种 × ${data.combos} 组参数 · ${data.workers} 进程`,
    )
    results.value = []
    resultsTotal.value = 0
    page.page = 1
    activeId.value = data.id
    await Promise.all([loadRuns(), refresh()])
    startPolling()
  } catch (error) {
    ElMessage.error(`启动失败：${error.message}`)
  } finally {
    starting.value = false
  }
}

async function refresh() {
  if (disposed || activeId.value == null) return
  try {
    const data = await api.getOptimization(activeId.value)
    if (disposed) return
    run.value = data
    if (!['pending', 'running'].includes(data.status)) {
      stopPolling()
      await Promise.all([loadResults(), loadRuns(), loadCoverage()])
    } else {
      await loadResults()
    }
  } catch (error) {
    stopPolling()
    ElMessage.error(`进度读取失败：${error.message}`)
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(refresh, 2000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function cancelOptimization() {
  if (activeId.value == null || cancelling.value) return
  cancelling.value = true
  try {
    await api.cancelOptimization(activeId.value)
    ElMessage.info('已请求取消，等当前币种跑完即停')
  } catch (error) {
    ElMessage.error(`取消失败：${error.message}`)
  } finally {
    cancelling.value = false
  }
}

async function selectRun(id) {
  if (id === activeId.value && run.value?.status === 'done') return
  activeId.value = id
  page.page = 1
  await refresh()
  if (running.value) startPolling()
}

async function loadResults() {
  if (disposed || activeId.value == null) return
  resultsLoading.value = true
  resultsError.value = ''
  const params = { page: page.page, page_size: page.pageSize, sort: 'score' }
  if (filters.verdict) params.verdict = filters.verdict
  if (filters.q.trim()) params.q = filters.q.trim()
  try {
    const data = await api.getOptimizationResults(activeId.value, params)
    if (disposed) return
    results.value = data.results || []
    resultsTotal.value = Number(data.total || 0)
  } catch (error) {
    if (!disposed) resultsError.value = error.message
  } finally {
    if (!disposed) resultsLoading.value = false
  }
}

function changePage(next) {
  if (next === page.page) return
  page.page = next
  return loadResults()
}

function changeFilter() {
  page.page = 1
  return loadResults()
}

async function applyResults(symbols = null) {
  if (!canApply.value || applying.value) return
  const scope = symbols?.length ? `${symbols.length} 个选中币种` : `任务 #${activeId.value} 的结果`
  const modeText = applyMode.value === 'symbol' ? '写入币种级参数覆盖' : '按参数聚类生成池 + 绑定'
  try {
    await ElMessageBox.confirm(
      `将把 ${scope}（${applyOnly.value === 'pass' ? '仅统计通过的' : applyOnly.value === 'positive' ? '仅验证段为正的' : '全部'}）${modeText}。${
        applyMode.value === 'symbol'
          ? '写入后下一轮巡检即按新参数评估。'
          : '生成的绑定默认停用，需要你到「策略」页手动启用。'
      }`,
      '确认应用',
      { confirmButtonText: '应用', cancelButtonText: '取消', type: 'warning' },
    )
  } catch {
    return
  }
  applying.value = true
  try {
    const data = await api.applyOptimization(activeId.value, buildApplyPayload(symbols))
    ElMessage.success(data.message || '已应用')
    await loadResults()
    await loadCoverage()
  } catch (error) {
    ElMessage.error(`应用失败：${error.message}`)
  } finally {
    applying.value = false
  }
}

async function clearCoverage() {
  if (!coverage.total) return
  try {
    await ElMessageBox.confirm(
      `将删除全部 ${coverage.total} 条币种级参数覆盖，这些币种回到绑定参数。`,
      '清空覆盖',
      { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' },
    )
  } catch {
    return
  }
  try {
    const groups = new Map()
    for (const item of coverage.items) {
      const key = `${item.strategy}::${item.interval}`
      if (!groups.has(key)) groups.set(key, { strategy: item.strategy, interval: item.interval, symbols: [] })
      groups.get(key).symbols.push(item.symbol)
    }
    let removed = 0
    for (const group of groups.values()) {
      const data = await api.deleteSymbolParams(group)
      removed += data.removed || 0
    }
    ElMessage.success(`已删除 ${removed} 条覆盖`)
    await loadCoverage()
  } catch (error) {
    ElMessage.error(`删除失败：${error.message}`)
  }
}

async function removeRun(id) {
  try {
    await ElMessageBox.confirm(`删除任务 #${id} 及其结果？`, '删除任务', {
      confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning',
    })
  } catch {
    return
  }
  try {
    await api.deleteOptimization(id)
    if (activeId.value === id) {
      activeId.value = null
      run.value = null
      results.value = []
      resultsTotal.value = 0
      stopPolling()
    }
    await loadRuns()
  } catch (error) {
    ElMessage.error(`删除失败：${error.message}`)
  }
}

onMounted(async () => {
  await Promise.all([loadOptions(), loadRuns(), loadCoverage(), previewUniverse()])
  const latest = runs.value.find((row) => row.running)
  if (latest) {
    activeId.value = latest.id
    await refresh()
    startPolling()
  }
})

onBeforeUnmount(() => {
  disposed = true
  stopPolling()
})

</script>

<template>
  <div class="panel" style="margin-bottom: 14px">
    <div class="panel-head">
      <div>
        <div class="panel-title">全币种参数寻优</div>
        <div class="panel-sub">
          每个币种独立跑同一张参数网格：训练段选参、验证段只做判定。
          算出结果后可一键写入币种级覆盖，或按参数聚类生成新的池与绑定。
        </div>
      </div>
      <el-tag size="small" effect="plain" type="info">{{ options.default_workers }} 进程 / {{ options.max_workers }} 上限</el-tag>
    </div>

    <el-alert v-if="loadError" type="error" :closable="false" show-icon :title="`加载失败：${loadError}`" />

    <div class="opt-form">
      <div class="opt-field">
        <div class="muted">策略</div>
        <el-select v-model="form.strategy" style="width: 100%">
          <el-option v-for="item in options.specs" :key="item.code" :label="item.name" :value="item.code" />
        </el-select>
      </div>
      <div class="opt-field">
        <div class="muted">周期</div>
        <el-select v-model="form.interval" style="width: 100%" @change="previewUniverse">
          <el-option v-for="item in ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
        </el-select>
      </div>
      <div class="opt-field">
        <div class="muted">范围</div>
        <el-select v-model="form.scope" style="width: 100%">
          <el-option label="全部有历史的币种" value="all" />
          <el-option label="指定币种池" value="pool" />
          <el-option label="手填币种" value="symbols" />
        </el-select>
      </div>
      <div v-if="form.scope === 'pool'" class="opt-field">
        <div class="muted">币种池</div>
        <el-select v-model="form.pool" style="width: 100%">
          <el-option v-for="item in options.pools" :key="item.code" :label="`${item.name}（${item.code}）`" :value="item.code" />
        </el-select>
      </div>
      <div v-else-if="form.scope === 'symbols'" class="opt-field opt-field-wide">
        <div class="muted">币种（逗号分隔）</div>
        <el-input v-model="form.symbols" placeholder="BTCUSDT, ETHUSDT" />
      </div>
      <div class="opt-field">
        <div class="muted">历史天数</div>
        <el-input-number v-model="form.days" :min="30" :max="720" :step="30" style="width: 100%" @change="previewUniverse" />
      </div>
      <div class="opt-field">
        <div class="muted">每组币组合数</div>
        <el-input-number v-model="form.combos" :min="4" :max="512" :step="8" style="width: 100%" />
      </div>
      <div class="opt-field">
        <div class="muted">成交门槛</div>
        <el-input-number v-model="form.minTrades" :min="1" :max="200" style="width: 100%" />
      </div>
      <div class="opt-field">
        <div class="muted">并发进程</div>
        <el-input-number v-model="form.workers" :min="0" :max="options.max_workers || 24" style="width: 100%" />
      </div>
      <div class="opt-field">
        <div class="muted">选参目标</div>
        <el-select v-model="form.objective" style="width: 100%">
          <el-option label="净收益 − 0.5×回撤" value="return_dd" />
          <el-option label="夏普" value="sharpe" />
        </el-select>
      </div>
      <div class="opt-field">
        <div class="muted">delta 源</div>
        <el-select v-model="form.deltaSource" style="width: 100%">
          <el-option label="自动（优先 1m，缺则估计）" value="auto" />
          <el-option label="强制 lower-tf" value="lower" />
          <el-option label="强制估计" value="est" />
        </el-select>
      </div>
      <div class="opt-field">
        <div class="muted">训练占比</div>
        <el-input-number v-model="form.trainRatio" :min="0.5" :max="0.85" :step="0.05" :precision="2" style="width: 100%" />
      </div>
    </div>

    <div v-if="universe" class="opt-universe muted">
      本次将扫描 <strong>{{ universe.count }}</strong> 个币种（{{ universe.interval }} · {{ universe.days }} 天以上历史）
      <span v-if="universe.sample?.length">：{{ universe.sample.join(' / ') }}{{ universe.count > universe.sample.length ? ' …' : '' }}</span>
      <span v-if="universe.count === 0" class="down">　没有币种满足历史要求，请缩短天数或换周期。</span>
    </div>
    <div v-if="gridRows.length" class="opt-grid-preview">
      <span class="muted">搜索空间（{{ spec?.grid_size }} 种组合）：</span>
      <el-tag v-for="row in gridRows" :key="row.key" size="small" effect="plain" style="margin: 2px 4px 2px 0">
        {{ row.key }} = {{ row.values.join(' / ') }}
      </el-tag>
    </div>

    <div class="opt-actions">
      <el-button type="primary" :loading="starting" :disabled="busy" @click="startOptimization">
        {{ busy ? '已有任务在运行' : '开始寻优' }}
      </el-button>
      <el-button v-if="running" :loading="cancelling" @click="cancelOptimization">取消</el-button>
      <span class="muted">
        并发默认留 4 核给巡检；单任务独占进程池，跑完后结果写入数据库供应用。
      </span>
    </div>
  </div>

  <div v-if="run" class="panel" style="margin-bottom: 14px">
    <div class="panel-head">
      <div>
        <div class="panel-title">任务 #{{ run.id }} · {{ run.strategy }} × {{ run.interval }}</div>
        <div class="panel-sub">
          {{ run.days }} 天 · {{ run.workers }} 进程 ·
          {{ run.status === 'done' ? '已完成' : run.status === 'running' ? '运行中' : run.status === 'cancelled' ? '已取消' : run.status === 'failed' ? '失败' : '排队中' }}
          <span v-if="run.finished_at"> · {{ tsLabel(run.finished_at) }}</span>
        </div>
      </div>
      <el-tag size="small" :type="run.status === 'done' ? 'success' : run.status === 'failed' ? 'danger' : 'warning'" effect="plain">
        {{ progressPct }}%
      </el-tag>
    </div>

    <el-progress :percentage="progressPct" :status="run.status === 'failed' ? 'exception' : undefined" :stroke-width="14" />
    <div class="opt-progress-meta muted">
      完成 {{ progress.done || 0 }} / {{ progress.total || 0 }}
      <span v-if="progress.skipped"> · 跳过 {{ progress.skipped }}</span>
      <span v-if="progress.failed"> · 失败 {{ progress.failed }}</span>
      <span v-if="running"> · 剩余约 {{ etaText }}</span>
      <span v-if="progress.current && running"> · 当前 {{ progress.current }}</span>
      <span v-if="summary.elapsed"> · 耗时 {{ Math.round(summary.elapsed) }} 秒</span>
    </div>

    <div v-if="verdictCounts.length" class="opt-verdicts">
      <el-tag v-for="item in verdictCounts" :key="item.key" :type="item.type" size="small" effect="light">
        {{ item.label }} {{ item.count }}
      </el-tag>
    </div>

    <el-alert
      v-if="pooled && pooled.trades"
      :type="pooled.verdict === 'positive' ? 'success' : pooled.verdict === 'negative' ? 'error' : 'warning'"
      :closable="false"
      show-icon
      style="margin-top: 10px"
      :title="`验证段池化：${pooled.trades} 笔，每笔期望 ${number(pooled.mean_return_pct)}%，95% CI [${number(pooled.ci_low_pct)}%, ${number(pooled.ci_high_pct)}%] → ${verdictMeta(pooled.verdict).label}`"
      :description="pooled.verdict === 'positive' ? '池化后仍为正，说明这批参数整体有信息量；单币仍需看各自判定。' : pooled.verdict === 'negative' ? '池化后为负：这批参数整体没有 edge，不建议批量上线。' : '池化后 CI 跨 0，整体仍不可判定。'"
    />
    <el-alert v-else-if="!running && run.status === 'done'" type="warning" :closable="false" show-icon style="margin-top: 10px"
      title="没有任何币种在验证段产生成交" description="这批参数在该区间没有触发信号，或成交门槛设得太高。可以放宽成交门槛、换更长的历史，或换 delta 源再跑一次。" />
    <el-alert v-if="run.error" type="error" :closable="false" show-icon :title="run.error.split('\n')[0]" style="margin-top: 10px" />
  </div>

  <div v-if="canApply" class="panel" style="margin-bottom: 14px">
    <div class="panel-head">
      <div>
        <div class="panel-title">一键应用</div>
        <div class="panel-sub">{{ applyHint() }}</div>
      </div>
      <el-tag size="small" effect="plain" type="warning">应用前请确认判定</el-tag>
    </div>
    <div class="opt-apply">
      <el-radio-group v-model="applyOnly" size="default">
        <el-radio-button value="pass">仅统计通过</el-radio-button>
        <el-radio-button value="positive">验证段为正</el-radio-button>
        <el-radio-button value="all">全部</el-radio-button>
      </el-radio-group>
      <el-radio-group v-model="applyMode" size="default">
        <el-radio-button value="symbol">写币种覆盖</el-radio-button>
        <el-radio-button value="pool">生成池 + 绑定</el-radio-button>
      </el-radio-group>
      <el-input v-if="applyMode === 'pool'" v-model="poolPrefix" placeholder="池名前缀（可选）" style="width: 180px" />
      <el-button type="primary" :loading="applying" @click="applyResults()">按筛选条件应用</el-button>
      <el-button v-if="selectedSymbols.length" :loading="applying" @click="applyResults(selectedSymbols)">
        仅应用本页 {{ selectedSymbols.length }} 个
      </el-button>
    </div>
    <div class="muted" style="margin-top: 8px">
      「写币种覆盖」立即对巡检生效（参数层：全局 &lt; 策略默认 &lt; 绑定 &lt; 币种）；
      「生成池 + 绑定」把参数相同的币种聚成一组，绑定默认停用，需到「策略」页启用。
    </div>
  </div>

  <div class="panel" style="margin-bottom: 14px">
    <div class="panel-head">
      <div>
        <div class="panel-title">逐币结果</div>
        <div class="panel-sub">按评分排序：训练段选出最优，验证段给判定；参数列只显示与当前不同的键</div>
      </div>
      <div class="opt-filters">
        <el-select v-model="filters.verdict" size="small" style="width: 130px" placeholder="全部判定" clearable @change="changeFilter">
          <el-option v-for="(meta, key) in VERDICTS" :key="key" :label="meta.label" :value="key" />
        </el-select>
        <el-input v-model="filters.q" size="small" style="width: 130px" placeholder="币种" clearable @keyup.enter="changeFilter" @clear="changeFilter" />
        <el-button size="small" @click="changeFilter">筛选</el-button>
      </div>
    </div>

    <div v-if="resultsError" class="muted" role="alert">结果加载失败：{{ resultsError }}</div>
    <el-table :data="results" class="table-compact" v-loading="resultsLoading" empty-text="还没有结果（任务未完成或该筛选下为空）">
      <el-table-column prop="symbol" label="币种" width="110" />
      <el-table-column label="判定" width="100">
        <template #default="{ row }">
          <el-tooltip :content="verdictMeta(row.verdict).hint" placement="top">
            <el-tag size="small" :type="verdictMeta(row.verdict).type" effect="light">{{ verdictMeta(row.verdict).label }}</el-tag>
          </el-tooltip>
        </template>
      </el-table-column>
      <el-table-column label="评分" width="80">
        <template #default="{ row }">{{ number(row.score) }}</template>
      </el-table-column>
      <el-table-column label="训练收益" width="100">
        <template #default="{ row }"><span :class="pnlClass(row.train?.total_return_pct)">{{ pct(row.train?.total_return_pct) }}</span></template>
      </el-table-column>
      <el-table-column label="训练笔数" width="90">
        <template #default="{ row }">{{ row.train?.total_trades ?? '—' }}</template>
      </el-table-column>
      <el-table-column label="验证收益" width="100">
        <template #default="{ row }"><span :class="pnlClass(row.valid?.total_return_pct)">{{ pct(row.valid?.total_return_pct) }}</span></template>
      </el-table-column>
      <el-table-column label="验证笔数" width="90">
        <template #default="{ row }">{{ row.valid?.total_trades ?? '—' }}</template>
      </el-table-column>
      <el-table-column label="每笔 CI" width="150">
        <template #default="{ row }">
          <span v-if="row.ci?.trades">[{{ number(row.ci.ci_low_pct) }}%, {{ number(row.ci.ci_high_pct) }}%]</span>
          <span v-else class="muted">—</span>
        </template>
      </el-table-column>
      <el-table-column label="最优参数" min-width="260">
        <template #default="{ row }">
          <span class="opt-params">{{ paramsText(row.params) }}</span>
          <div v-if="row.params_before && Object.keys(row.params_before).length" class="muted opt-before">
            原：{{ paramsText(row.params_before) }}
          </div>
        </template>
      </el-table-column>
      <el-table-column label="delta" width="80">
        <template #default="{ row }"><el-tag size="small" effect="plain" type="info">{{ row.delta_source || '—' }}</el-tag></template>
      </el-table-column>
      <el-table-column label="操作" width="100">
        <template #default="{ row }">
          <el-button size="small" link type="primary" :disabled="!row.params || !Object.keys(row.params).length" @click="applyResults([row.symbol])">应用</el-button>
        </template>
      </el-table-column>
    </el-table>
    <div class="opt-pager">
      <el-pagination
        layout="total, prev, pager, next"
        :total="resultsTotal"
        :page-size="page.pageSize"
        :current-page="page.page"
        @current-change="changePage"
      />
    </div>
  </div>

  <div class="panel" style="margin-bottom: 14px">
    <div class="panel-head">
      <div>
        <div class="panel-title">币种级参数覆盖</div>
        <div class="panel-sub">覆盖优先于绑定参数；删除即回到绑定。当前 {{ coverage.total }} 条</div>
      </div>
      <div>
        <el-button size="small" :loading="coverageLoading" @click="loadCoverage">刷新</el-button>
        <el-button size="small" @click="coverageVisible = true">查看</el-button>
        <el-button size="small" type="danger" plain :disabled="!coverage.total" @click="clearCoverage">清空</el-button>
      </div>
    </div>
    <div class="muted">
      <span v-for="(count, source) in coverage.sources" :key="source" style="margin-right: 12px">
        {{ source }}：{{ count }}
      </span>
      <span v-if="!coverage.total">暂无覆盖，所有币种沿用绑定参数。</span>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <div>
        <div class="panel-title">历史任务</div>
        <div class="panel-sub">点一行查看它的结果</div>
      </div>
      <el-button size="small" @click="loadRuns">刷新</el-button>
    </div>
    <el-table :data="runs" class="table-compact" empty-text="还没有寻优任务">
      <el-table-column prop="id" label="#" width="70" />
      <el-table-column prop="strategy" label="策略" min-width="150" />
      <el-table-column prop="interval" label="周期" width="80" />
      <el-table-column prop="days" label="天数" width="80" />
      <el-table-column label="状态" width="110">
        <template #default="{ row }">
          <el-tag size="small" effect="plain" :type="row.status === 'done' ? 'success' : row.status === 'failed' ? 'danger' : row.running ? 'warning' : 'info'">
            {{ row.status }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="进度" width="120">
        <template #default="{ row }">{{ row.progress?.done || 0 }} / {{ row.progress?.total || 0 }}</template>
      </el-table-column>
      <el-table-column label="创建" width="170">
        <template #default="{ row }">{{ tsLabel(row.created_at) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="150">
        <template #default="{ row }">
          <el-button size="small" link type="primary" @click="selectRun(row.id)">查看</el-button>
          <el-button size="small" link type="danger" :disabled="row.running" @click="removeRun(row.id)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>

  <el-dialog v-model="coverageVisible" title="币种级参数覆盖" width="820px">
    <el-table :data="coverage.items" size="small" max-height="460">
      <el-table-column prop="symbol" label="币种" width="110" />
      <el-table-column prop="strategy" label="策略" width="150" />
      <el-table-column prop="interval" label="周期" width="70" />
      <el-table-column label="参数" min-width="300">
        <template #default="{ row }"><span class="opt-params">{{ paramsText(row.params) }}</span></template>
      </el-table-column>
      <el-table-column prop="source" label="来源" width="110" />
      <el-table-column label="更新" width="160">
        <template #default="{ row }">{{ tsLabel(row.updated_at) }}</template>
      </el-table-column>
    </el-table>
  </el-dialog>
</template>

<style scoped>
.opt-form {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px 12px;
  margin-top: 12px;
}
.opt-field-wide { grid-column: span 2; }
.opt-universe { margin-top: 10px; font-size: 12px; }
.opt-grid-preview {
  margin-top: 10px;
  font-size: 12px;
  line-height: 1.9;
}
.opt-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
  flex-wrap: wrap;
}
.opt-progress-meta { margin-top: 6px; font-size: 12px; }
.opt-verdicts { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.opt-apply { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.opt-filters { display: flex; gap: 8px; align-items: center; }
.opt-params { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.opt-before { font-size: 11px; margin-top: 2px; }
.opt-pager { display: flex; justify-content: flex-end; margin-top: 10px; }
@media (max-width: 1200px) {
  .opt-form { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 760px) {
  .opt-form { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
