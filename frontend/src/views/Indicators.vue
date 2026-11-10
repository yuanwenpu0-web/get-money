<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import api from '../api'
import TablePager from '../components/TablePager.vue'
import { PAGE_SIZE } from '../utils/pagination'

const items = ref([])
const facets = ref({ collections: [], categories: [], indicator_types: [] })
const total = ref(0)
const catalogTotal = ref(0)
const loading = ref(false)
const error = ref('')
const search = ref('')
const collection = ref('')
const category = ref('')
const indicatorType = ref('')
const research = ref(null)
const researchError = ref('')
const page = ref(1)
const pageSize = PAGE_SIZE
const detailVisible = ref(false)
const detail = ref(null)
const detailLoading = ref(false)
const detailError = ref('')
let listRequest = 0
let searchTimer = null
let detailRequest = 0
let disposed = false

const collectionOptions = computed(() => facets.value.collections || [])
const categoryOptions = computed(() => facets.value.categories || [])
const typeOptions = computed(() => facets.value.indicator_types || [])
const rankedSingles = computed(() => [...(research.value?.singles_all || [])].sort((a, b) => b.train.total_pnl - a.train.total_pnl).slice(0, 10))
const rankedCombos = computed(() => [...(research.value?.combos_all || [])].sort((a, b) => b.train.total_pnl - a.train.total_pnl).slice(0, 10))
const filteredCount = computed(() => total.value)

// 目录在服务端筛选 + 分页：一次只回 15 行（原来 156 行 180KB + 0.6s）
async function load() {
  const request = ++listRequest
  loading.value = true
  error.value = ''
  try {
    const data = await api.getIndicators({
      q: search.value.trim(),
      collection: collection.value,
      category: category.value,
      indicator_type: indicatorType.value,
      page: page.value,
      page_size: pageSize,
    })
    if (disposed || request !== listRequest) return
    items.value = data.items || []
    facets.value = data.facets || { collections: [], categories: [], indicator_types: [] }
    total.value = Number(data.total || 0)
    catalogTotal.value = Number(data.catalog_total ?? data.total ?? 0)
    const last = Math.max(1, Math.ceil(total.value / pageSize))
    if (page.value > last) {
      page.value = last
      return load()
    }
  } catch (err) {
    if (!disposed && request === listRequest) error.value = err.message
  } finally {
    if (!disposed && request === listRequest) loading.value = false
  }
}

function changePage(next) {
  if (next === page.value) return
  page.value = next
  return load()
}

function onFilterChange() {
  page.value = 1
  return load()
}

// 搜索输入防抖：每敲一个字都打一次接口会让服务端反复解析目录
function onSearchInput() {
  if (searchTimer) clearTimeout(searchTimer)
  searchTimer = setTimeout(() => { if (!disposed) onFilterChange() }, 300)
}

watch([collection, category, indicatorType], () => { page.value = 1 })
watch(collection, () => { category.value = '' })

async function showDetail(row) {
  const request = ++detailRequest
  detailVisible.value = true
  detail.value = row
  detailLoading.value = true
  detailError.value = ''
  try {
    const data = await api.getIndicator(row.id)
    if (!disposed && request === detailRequest) detail.value = data
  } catch (err) {
    if (!disposed && request === detailRequest) detailError.value = err.message
  } finally {
    if (!disposed && request === detailRequest) detailLoading.value = false
  }
}

function closeDetail() {
  detailRequest++
  detailLoading.value = false
}

function safeSourceUrl(url) {
  try {
    const parsed = new URL(url)
    return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : ''
  } catch {
    return ''
  }
}

onMounted(() => {
  const pending = load()
  if (api.getIndicatorResearch) api.getIndicatorResearch().then((data) => { if (!disposed) research.value = data }).catch((err) => { if (!disposed) researchError.value = err.message })
  return pending
})
onBeforeUnmount(() => {
  disposed = true
  detailRequest++
  if (searchTimer) clearTimeout(searchTimer)
})
</script>

<template>
  <div>
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">指标目录</div>
          <div class="panel-sub">浏览已收录的 TradingView Pine 指标及其源码；目录中的脚本不会自动参与本地计算或交易。</div>
          <div class="panel-sub">指标类型、计算概述和场景依据目录分类或逐项源码归纳；未经核实的条目标为待分类。</div>
        </div>
        <el-button size="small" :loading="loading" @click="load">刷新</el-button>
      </div>
      <div class="row indicator-filters">
        <el-input v-model="search" clearable placeholder="搜索名称、描述或分类" aria-label="搜索指标" style="max-width: 280px" @input="onSearchInput" @clear="onFilterChange" @keyup.enter="onFilterChange" />
        <el-select v-model="collection" clearable placeholder="全部来源" aria-label="指标来源" style="width: 220px" @change="onFilterChange">
          <el-option v-for="item in collectionOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <el-select v-model="category" clearable filterable placeholder="全部分类" aria-label="指标分类" style="width: 200px" @change="onFilterChange">
          <el-option v-for="item in categoryOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <el-select v-model="indicatorType" clearable placeholder="全部指标类型" aria-label="指标类型" style="width: 180px" @change="onFilterChange">
          <el-option v-for="item in typeOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <span class="muted">{{ filteredCount }} / {{ catalogTotal }} 个指标</span>
      </div>
      <div v-if="error" class="row section-gap" role="alert">
        <span class="down">指标加载失败：{{ error }}</span>
        <el-button size="small" :loading="loading" @click="load">重试</el-button>
      </div>
      <div v-if="loading && !items.length" class="empty-hint" role="status">正在加载指标…</div>
      <div v-else-if="!items.length && !error" class="empty-hint">{{ catalogTotal ? '没有符合条件的指标' : '暂无指标' }}</div>
      <el-table v-else-if="items.length" :data="items" class="table-compact section-gap" @row-click="showDetail">
        <el-table-column label="指标" min-width="220">
          <template #default="{ row }">
            <button type="button" class="indicator-title" @click.stop="showDetail(row)">{{ row.title }}</button>
            <div v-if="row.original_title && row.original_title !== row.title" class="muted indicator-original">{{ row.original_title }}</div>
          </template>
        </el-table-column>
        <el-table-column label="说明" min-width="280" show-overflow-tooltip>
          <template #default="{ row }">{{ row.description || '—' }}</template>
        </el-table-column>
        <el-table-column prop="category" label="分类" min-width="130" show-overflow-tooltip />
        <el-table-column prop="indicator_type" label="指标类型" min-width="120" />
        <el-table-column label="Pine" width="80">
          <template #default="{ row }">{{ row.version ? `v${row.version}` : '—' }}</template>
        </el-table-column>
        <el-table-column label="操作" width="90">
          <template #default="{ row }"><el-button link type="primary" @click.stop="showDetail(row)">查看</el-button></template>
        </el-table-column>
      </el-table>
      <TablePager :page="page" :page-size="pageSize" :total="filteredCount" :disabled="loading" @change="changePage" />
    </div>

    <div class="panel section-gap">
      <div class="panel-title">指标参数与组合评估</div>
      <p v-if="researchError" class="muted">评估报告不可用：{{ researchError }}</p>
      <template v-if="research">
        <p class="muted">{{ research.dataset.interval }} · {{ research.dataset.symbols }} 币 · {{ research.dataset.start }} 至 {{ research.dataset.end }} · {{ research.summary.single_tests }} 组单指标参数、{{ research.summary.combo_tests }} 组组合；{{ research.summary.passed }} 组通过检验。</p>
        <p class="muted">多空双向；收盘确认、下一根开盘入场；RSI(14) 穿越 50 出场，最多持有 240 根；手续费和滑点合计往返 0.12%。训练集选参，样本外检验。固定每笔名义本金 10，收益率分母为各段最大并发持仓数 × 10，因此不同候选的收益率分母可能不同。</p>
        <p class="muted">以下按训练期累计盈亏排名；样本外仅用于验证。研究信号包含 15 个家族，不等同于覆盖目录全部指标，也未测试所有参数或组合；资金费、冲击成本和实际组合资金限制未计入。</p>
        <div v-for="group in [{ title: '单指标前十', rows: rankedSingles }, { title: '组合前十', rows: rankedCombos }]" :key="group.title" class="section-gap">
          <div class="panel-title">{{ group.title }}</div>
          <el-table :data="group.rows" class="table-compact">
            <el-table-column type="index" label="排名" width="65" />
            <el-table-column prop="label" label="指标、参数及组合规则" min-width="260" />
            <el-table-column label="训练盈亏" width="105"><template #default="{ row }">{{ row.train.total_pnl.toFixed(3) }}</template></el-table-column>
            <el-table-column label="训练收益率" width="110"><template #default="{ row }">{{ row.train.total_return_pct.toFixed(2) }}%</template></el-table-column>
            <el-table-column label="样本外盈亏" width="115"><template #default="{ row }">{{ row.test.total_pnl.toFixed(3) }}</template></el-table-column>
            <el-table-column label="样本外收益率" width="125"><template #default="{ row }">{{ row.test.total_return_pct.toFixed(2) }}%</template></el-table-column>
            <el-table-column prop="q_value" label="q 值" width="90" />
            <el-table-column prop="verdict" label="判定" width="120" />
          </el-table>
        </div>
        <details class="section-gap"><summary>已测试信号及参数范围</summary>
          <p v-for="(spec, name) in research.grid" :key="name" class="muted">{{ spec.label }}（{{ name }}）：{{ spec.rule }}；参数 {{ JSON.stringify(spec.params) }}；{{ spec.combos }} 组。</p>
        </details>
      </template>
    </div>

    <el-dialog v-model="detailVisible" :title="detail?.title || '指标详情'" width="min(900px, 92vw)" @close="closeDetail">
      <template v-if="detail">
        <div class="row indicator-meta">
          <el-tag size="small" effect="plain">{{ detail.collection }}</el-tag>
          <el-tag v-if="detail.category" size="small" effect="plain" type="info">{{ detail.category }}</el-tag>
          <el-tag v-if="detail.version" size="small" effect="plain" type="info">Pine v{{ detail.version }}</el-tag>
          <el-tag size="small" effect="plain" type="info">{{ detail.license || '未注明' }}</el-tag>
          <a v-if="safeSourceUrl(detail.source_url)" :href="safeSourceUrl(detail.source_url)" target="_blank" rel="noopener noreferrer">查看上游源码 ↗</a>
        </div>
        <p v-if="detail.original_title && detail.original_title !== detail.title" class="muted">{{ detail.original_title }}</p>
        <p v-if="detail.description" class="indicator-description">{{ detail.description }}</p>
        <p v-if="detail.notes" class="indicator-description">{{ detail.notes }}</p>
        <p class="indicator-description">类型：{{ detail.indicator_type }}；计算概述：{{ detail.calculation_summary }}；适用场景：{{ detail.use_case }}。具体公式及参数以源码为准。</p>
        <p class="indicator-description">Python 实现：{{ detail.python_method || '未绑定' }}；目录指标不会自动参与交易或上述信号实验。</p>
        <div v-if="detailLoading" class="empty-hint" role="status">正在加载源码…</div>
        <div v-else-if="detailError" class="row" role="alert">
          <span class="down">详情加载失败：{{ detailError }}</span>
          <el-button size="small" @click="showDetail(detail)">重试</el-button>
        </div>
        <template v-else>
          <div class="panel-title section-gap">Pine 源码</div>
          <pre class="indicator-source mono"><code>{{ detail.source }}</code></pre>
          <details v-if="detail.license_text" class="section-gap">
            <summary>许可文本</summary>
            <pre class="indicator-source mono">{{ detail.license_text }}</pre>
          </details>
        </template>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.indicator-filters { margin-top: 8px; }
.indicator-title { border: 0; padding: 0; color: #e6ecf5; background: none; text-align: left; cursor: pointer; font: inherit; font-weight: 600; }
.indicator-title:hover, .indicator-title:focus-visible { color: #34d399; text-decoration: underline; }
.indicator-original { margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.indicator-meta a { color: #34d399; font-size: 12px; }
.indicator-description { color: #a9b6c8; font-size: 13px; line-height: 1.7; white-space: pre-wrap; }
.indicator-source { max-height: 46vh; overflow: auto; padding: 14px; border: 1px solid rgba(148, 163, 184, .14); border-radius: 8px; background: #080b11; color: #cbd5e1; font-size: 11px; line-height: 1.6; white-space: pre; }
summary { cursor: pointer; color: #a9b6c8; }
</style>

