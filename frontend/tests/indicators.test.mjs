import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { computed, nextTick, ref, watch } from 'vue'

const plain = (value) => JSON.parse(JSON.stringify(value))

function deferred() {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function indicators(api) {
  const source = readFileSync(new URL('../src/views/Indicators.vue', import.meta.url), 'utf8')
  const script = source.match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const mounts = [], unmounts = []
  const pending = new Map()
  let timer = 0
  const context = vm.createContext({
    PAGE_SIZE: 15,
    api, ref, computed, watch, URL,
    setTimeout: (fn) => { timer += 1; pending.set(timer, fn); return timer },
    clearTimeout: (id) => pending.delete(id),
    onMounted: (fn) => mounts.push(fn), onBeforeUnmount: (fn) => unmounts.push(fn),
  })
  vm.runInContext(`${script}\nglobalThis.subject = {
    items, loading, error, search, collection, category, indicatorType, page, pageSize,
    facets, total, catalogTotal, filteredCount, detail, detailLoading, detailError, detailVisible,
    load, changePage, onFilterChange, onSearchInput, showDetail, closeDetail, safeSourceUrl,
  }`, context)
  return {
    ...context.subject,
    source,
    flushTimers: () => { const fns = [...pending.values()]; pending.clear(); fns.forEach((fn) => fn()) },
    mount: () => Promise.all(mounts.map((fn) => fn())),
    unmount: () => unmounts.forEach((fn) => fn()),
  }
}

const rows = [
  { id: 'a', title: '趋势线', original_title: 'Trendline', description: '识别趋势', category: 'trend', collection: 'first', indicator_type: '趋势方向' },
  { id: 'b', title: 'RSI', description: '动量', category: 'momentum', collection: 'second', indicator_type: '动量' },
]
const facets = {
  collections: ['first', 'second'],
  categories: ['momentum', 'trend'],
  indicator_types: ['动量', '趋势方向'],
}

test('指标目录的筛选与分页都交给服务端：请求带参数，表格只渲染当前页', async () => {
  const requests = []
  const view = indicators({
    getIndicators: async (params) => {
      requests.push(params)
      return { items: rows, facets, total: 2, catalog_total: 156, page: params.page, page_size: 15 }
    },
  })
  const pending = view.mount()
  assert.equal(view.loading.value, true)
  await pending
  assert.equal(view.loading.value, false)
  assert.deepEqual(plain(requests), [{ q: '', collection: '', category: '', indicator_type: '', page: 1, page_size: 15 }])
  assert.equal(view.items.value.length, 2)
  assert.equal(view.total.value, 2)
  assert.equal(view.catalogTotal.value, 156)
  // 下拉选项来自服务端 facets，不再依赖「已拉到全部行」
  assert.deepEqual(view.facets.value.categories, ['momentum', 'trend'])

  view.search.value = '  Trendline '
  view.onFilterChange()
  await nextTick()
  assert.equal(requests[1].q, 'Trendline')
  assert.equal(requests[1].page, 1)

  view.page.value = 1
  view.changePage(3)
  await nextTick()
  assert.equal(requests[2].page, 3)

  view.unmount()
})

test('搜索输入做防抖，不逐字符打接口', async () => {
  const requests = []
  const view = indicators({
    getIndicators: async (params) => {
      requests.push(params.q)
      return { items: [], facets, total: 0, catalog_total: 0, page: 1, page_size: 15 }
    },
  })
  await view.mount()
  const before = requests.length
  view.search.value = 'RS'
  view.onSearchInput()
  view.search.value = 'RSI'
  view.onSearchInput()
  assert.equal(requests.length, before) // 还没到 300ms，一次都没发
  view.flushTimers()
  await nextTick()
  assert.deepEqual(requests.slice(before), ['RSI'])
  view.unmount()
})

test('目录和详情加载失败后可重试，详情返回源码', async () => {
  let listAttempts = 0, detailAttempts = 0
  const view = indicators({
    getIndicators: async () => {
      if (++listAttempts === 1) throw new Error('目录离线')
      return { items: rows, facets, total: 2, catalog_total: 2, page: 1, page_size: 15 }
    },
    getIndicator: async () => {
      if (++detailAttempts === 1) throw new Error('源码离线')
      return { ...rows[0], source: 'indicator("Trend")', license_text: 'MIT' }
    },
  })
  await view.mount()
  assert.equal(view.error.value, '目录离线')
  await view.load()
  assert.equal(view.error.value, '')
  assert.equal(view.items.value.length, 2)
  await view.showDetail(rows[0])
  assert.equal(view.detailError.value, '源码离线')
  assert.equal(view.detailLoading.value, false)
  await view.showDetail(rows[0])
  assert.equal(view.detailError.value, '')
  assert.equal(view.detail.value.source, 'indicator("Trend")')
  assert.equal(view.detail.value.license_text, 'MIT')
  assert.match(view.source, /\{\{ detail\.source \}\}/)
  assert.match(view.source, /不会自动参与本地计算或交易/)
  assert.equal(view.safeSourceUrl('javascript:alert(1)'), '')
  view.unmount()
})

test('切换或关闭详情后忽略过期响应', async () => {
  const first = deferred(), second = deferred()
  const view = indicators({ getIndicator: (id) => id === 'a' ? first.promise : second.promise })
  const old = view.showDetail(rows[0])
  const current = view.showDetail(rows[1])
  second.resolve({ ...rows[1], source: 'second source' })
  await current
  first.resolve({ ...rows[0], source: 'old source' })
  await old
  assert.equal(view.detail.value.id, 'b')
  assert.equal(view.detail.value.source, 'second source')
  const late = deferred()
  const other = indicators({ getIndicator: () => late.promise })
  const pending = other.showDetail(rows[0])
  other.closeDetail()
  late.resolve({ ...rows[0], source: 'late source' })
  await pending
  assert.equal(other.detail.value.source, undefined)
  view.unmount()
  other.unmount()
})
