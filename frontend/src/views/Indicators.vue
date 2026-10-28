<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import api from '../api'

const items = ref([])
const loading = ref(false)
const error = ref('')
const search = ref('')
const collection = ref('')
const category = ref('')
const page = ref(1)
const pageSize = 20
const detailVisible = ref(false)
const detail = ref(null)
const detailLoading = ref(false)
const detailError = ref('')
let listPromise = null
let detailRequest = 0
let disposed = false

const collectionOptions = computed(() => [...new Set(items.value.map((item) => item.collection).filter(Boolean))].sort())
const categoryOptions = computed(() => [...new Set(items.value
  .filter((item) => !collection.value || item.collection === collection.value)
  .map((item) => item.category).filter(Boolean))].sort())
const filtered = computed(() => {
  const query = search.value.trim().toLocaleLowerCase()
  return items.value.filter((item) => {
    if (collection.value && item.collection !== collection.value) return false
    if (category.value && item.category !== category.value) return false
    return !query || [item.title, item.original_title, item.description, item.category, item.collection]
      .some((value) => String(value || '').toLocaleLowerCase().includes(query))
  })
})
const displayed = computed(() => filtered.value.slice((page.value - 1) * pageSize, page.value * pageSize))
watch([search, category, collection], () => { page.value = 1 })
watch(collection, () => { category.value = '' })

function load() {
  if (listPromise) return listPromise
  loading.value = true
  error.value = ''
  listPromise = api.getIndicators().then((data) => {
    if (!disposed) {
      items.value = data.items || []
      page.value = Math.min(page.value, Math.max(1, Math.ceil(filtered.value.length / pageSize)))
    }
  }).catch((err) => {
    if (!disposed) error.value = err.message
  }).finally(() => {
    if (!disposed) loading.value = false
    listPromise = null
  })
  return listPromise
}

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

onMounted(load)
onBeforeUnmount(() => { disposed = true; detailRequest++ })
</script>

<template>
  <div>
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">指标目录</div>
          <div class="panel-sub">浏览已收录的 TradingView Pine 指标及其源码；目录中的脚本不会自动参与本地计算或交易。</div>
        </div>
        <el-button size="small" :loading="loading" @click="load">刷新</el-button>
      </div>
      <div class="row indicator-filters">
        <el-input v-model="search" clearable placeholder="搜索名称、描述或分类" aria-label="搜索指标" style="max-width: 280px" />
        <el-select v-model="collection" clearable placeholder="全部来源" aria-label="指标来源" style="width: 220px">
          <el-option v-for="item in collectionOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <el-select v-model="category" clearable filterable placeholder="全部分类" aria-label="指标分类" style="width: 200px">
          <el-option v-for="item in categoryOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <span class="muted">{{ filtered.length }} / {{ items.length }} 个指标</span>
      </div>
      <div v-if="error" class="row section-gap" role="alert">
        <span class="down">指标加载失败：{{ error }}</span>
        <el-button size="small" :loading="loading" @click="load">重试</el-button>
      </div>
      <div v-if="loading && !items.length" class="empty-hint" role="status">正在加载指标…</div>
      <div v-else-if="!displayed.length && !error" class="empty-hint">{{ items.length ? '没有符合条件的指标' : '暂无指标' }}</div>
      <el-table v-else-if="displayed.length" :data="displayed" class="table-compact section-gap" @row-click="showDetail">
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
        <el-table-column label="Pine" width="80">
          <template #default="{ row }">{{ row.version ? `v${row.version}` : '—' }}</template>
        </el-table-column>
        <el-table-column label="操作" width="90">
          <template #default="{ row }"><el-button link type="primary" @click.stop="showDetail(row)">查看</el-button></template>
        </el-table-column>
      </el-table>
      <div v-if="filtered.length > pageSize" class="table-pagination">
        <el-pagination v-model:current-page="page" :page-size="pageSize" :total="filtered.length" layout="total, prev, pager, next" />
      </div>
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

