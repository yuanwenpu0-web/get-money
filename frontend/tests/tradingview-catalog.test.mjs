import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { computed, nextTick, ref, watch } from 'vue'

function view(api) {
  const source = readFileSync(new URL('../src/components/TradingViewCatalog.vue', import.meta.url), 'utf8')
  const script = source.match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const mounts = [], unmounts = []
  const context = vm.createContext({ api, ref, computed, watch,
    onMounted: (fn) => mounts.push(fn), onBeforeUnmount: (fn) => unmounts.push(fn) })
  vm.runInContext(`${script}\nglobalThis.subject = { items, loading, error, search, page, filtered, displayed, detail, detailVisible, detailLoading, detailError, load, showDetail, closeDetail, backtestForm, backtestRunning, backtestError, backtestResult, runBacktest }`, context)
  return { ...context.subject, source, mount: () => Promise.all(mounts.map((f) => f())), unmount: () => unmounts.forEach((f) => f()) }
}

test('目录显示状态，支持按路径搜索并查看源码', async () => {
  const items = [{ id: 'one', title: 'SMA', path: 'data/sma.pine', execution_status: 'entry_filter_and_research' },
    { id: 'two', title: 'Other', path: 'data/other.pine', execution_status: 'not_migrated' }]
  const page = view({ getTradingViewStrategies: async () => ({ strategies: items }),
    getTradingViewStrategy: async (id) => ({ ...items.find((row) => row.id === id), source: 'strategy("SMA")' }) })
  await page.mount()
  assert.equal(page.items.value.length, 2)
  page.search.value = 'OTHER.PINE'
  await nextTick()
  assert.equal(page.filtered.value[0].id, 'two')
  await page.showDetail(items[0])
  assert.equal(page.detail.value.source, 'strategy("SMA")')
  assert.match(page.source, /尚未迁移/)
  assert.match(page.source, /不支持原脚本完整自动交易/)
  page.unmount()
})

test('目录失败可重试，关闭详情后不会显示迟到的源码', async () => {
  let attempt = 0, resolve
  const page = view({ getTradingViewStrategies: async () => { if (++attempt === 1) throw new Error('离线'); return { strategies: [] } },
    getTradingViewStrategy: () => new Promise((done) => { resolve = done }) })
  await page.mount()
  assert.equal(page.error.value, '离线')
  await page.load()
  assert.equal(page.error.value, '')
  const pending = page.showDetail({ id: 'a', title: 'A' })
  page.closeDetail()
  resolve({ id: 'a', source: 'late' })
  await pending
  assert.equal(page.detail.value.source, undefined)
  page.unmount()
})

test('研究回测发送行情参数并展示结果，未迁移策略不能运行', async () => {
  let calls = 0, received
  const page = view({ getTradingViewStrategies: async () => ({ strategies: [] }),
    getTradingViewStrategy: async (id) => ({ id, execution_status: id === 'ok' ? 'research_backtest' : 'not_migrated', source: 'strategy()' }),
    runTradingViewBacktest: async (id, payload) => { calls++; received = { id, payload }; return { metrics: { final_equity: 10100 }, trades: [], assumptions: [] } } })
  await page.mount()
  await page.showDetail({ id: 'ok' })
  page.backtestForm.value.symbol = ' ethusdt '
  await page.runBacktest()
  assert.equal(calls, 1)
  assert.equal(received.id, 'ok')
  assert.equal(received.payload.symbol, 'ETHUSDT')
  assert.equal(received.payload.interval, '15m')
  assert.equal(page.backtestResult.value.metrics.final_equity, 10100)
  assert.match(page.source, /不支持原脚本完整自动交易/)
  await page.showDetail({ id: 'unknown' })
  await page.runBacktest()
  assert.equal(calls, 1)
  assert.equal(page.backtestResult.value, null)
  page.unmount()
})

test('回测失败可见，切换详情后忽略迟到的回测结果', async () => {
  let resolve
  const page = view({ getTradingViewStrategies: async () => ({ strategies: [] }),
    getTradingViewStrategy: async (id) => ({ id, execution_status: 'entry_filter_and_research' }),
    runTradingViewBacktest: (id) => id === 'fail' ? Promise.reject(new Error('行情不足')) : new Promise((done) => { resolve = done }) })
  await page.mount()
  await page.showDetail({ id: 'fail' })
  await page.runBacktest()
  assert.equal(page.backtestError.value, '行情不足')
  await page.showDetail({ id: 'slow' })
  const pending = page.runBacktest()
  await page.showDetail({ id: 'other' })
  resolve({ metrics: { final_equity: 1 } })
  await pending
  assert.equal(page.backtestResult.value, null)
  assert.equal(page.backtestRunning.value, false)
  page.unmount()
})
