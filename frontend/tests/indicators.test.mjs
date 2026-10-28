import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { computed, nextTick, ref, watch } from 'vue'

function deferred() {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function indicators(api) {
  const source = readFileSync(new URL('../src/views/Indicators.vue', import.meta.url), 'utf8')
  const script = source.match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const mounts = [], unmounts = []
  const context = vm.createContext({
    api, ref, computed, watch, URL,
    onMounted: (fn) => mounts.push(fn), onBeforeUnmount: (fn) => unmounts.push(fn),
  })
  vm.runInContext(`${script}\nglobalThis.subject = { items, loading, error, search, collection, category, page, filtered, displayed, detail, detailLoading, detailError, detailVisible, load, showDetail, closeDetail, safeSourceUrl }`, context)
  return { ...context.subject, source, mount: () => Promise.all(mounts.map((fn) => fn())), unmount: () => unmounts.forEach((fn) => fn()) }
}

const rows = [
  { id: 'a', title: '趋势线', original_title: 'Trendline', description: '识别趋势', category: 'trend', collection: 'first' },
  { id: 'b', title: 'RSI', description: '动量', category: 'momentum', collection: 'second' },
]

test('指标目录加载后支持名称、说明、来源及分类过滤', async () => {
  const response = deferred()
  const view = indicators({ getIndicators: () => response.promise })
  const pending = view.mount()
  assert.equal(view.loading.value, true)
  response.resolve({ items: rows })
  await pending
  assert.equal(view.loading.value, false)
  assert.equal(view.items.value.length, 2)
  view.search.value = 'TRENDLINE'
  await nextTick()
  assert.equal(view.filtered.value[0].id, 'a')
  view.search.value = '动量'
  await nextTick()
  assert.equal(view.filtered.value[0].id, 'b')
  view.search.value = ''
  view.collection.value = 'first'
  await nextTick()
  view.category.value = 'trend'
  await nextTick()
  assert.equal(view.filtered.value.length, 1)
  assert.equal(view.filtered.value[0].id, 'a')
  view.collection.value = 'second'
  await nextTick()
  assert.equal(view.category.value, '')
  assert.equal(view.filtered.value[0].id, 'b')
  view.unmount()
})

test('目录和详情加载失败后可重试，详情返回源码', async () => {
  let listAttempts = 0, detailAttempts = 0
  const view = indicators({
    getIndicators: async () => { if (++listAttempts === 1) throw new Error('目录离线'); return { items: rows } },
    getIndicator: async () => { if (++detailAttempts === 1) throw new Error('源码离线'); return { ...rows[0], source: 'indicator("Trend")', license_text: 'MIT' } },
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
