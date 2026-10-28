import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const tick = () => new Promise((resolve) => setImmediate(resolve))
function deferred() {
  let resolve
  const promise = new Promise((done) => { resolve = done })
  return { promise, resolve }
}

function view(api) {
  const source = readFileSync(new URL('../src/views/Market.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const unmounted = []
  const context = vm.createContext({
    api, ref: (value) => ({ value }), reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {}, onBeforeUnmount: (fn) => unmounted.push(fn),
  })
  vm.runInContext(`${source}\nglobalThis.subject = { loadSources, changeStatsPage, changeStatsPageSize, statsPage, statsPageSize, statsTotal, localStats, sourcesLoading }`, context)
  return { ...context.subject, unmount: () => unmounted.forEach((fn) => fn()) }
}

test('Market requests only its selected page and uses the full group total', async () => {
  const requests = []
  const subject = view({ getMarketSources: async (params) => {
    requests.push({ ...params })
    return { local: { stats: [{ symbol: 'S020USDT' }], total: 41 } }
  } })
  await subject.changeStatsPage(2)
  assert.deepEqual(requests, [{ page: 2, page_size: 20 }])
  assert.equal(subject.statsTotal.value, 41)
  assert.equal(subject.localStats.value.length, 1)
  await subject.changeStatsPage(2)
  await subject.changeStatsPageSize(20)
  assert.equal(requests.length, 1)
  await subject.changeStatsPageSize(10)
  assert.deepEqual(requests[1], { page: 1, page_size: 10 })
  assert.equal(subject.statsPage.value, 1)
})

test('Market ignores an obsolete page response and does not clear the new loading state', async () => {
  const requests = []
  const subject = view({ getMarketSources: (params) => {
    const response = deferred()
    requests.push({ ...params, ...response })
    return response.promise
  } })
  const first = subject.loadSources()
  const second = subject.changeStatsPage(2)
  requests[0].resolve({ local: { stats: [{ symbol: 'OLD' }], total: 42 } })
  await first
  assert.equal(subject.sourcesLoading.value, true)
  assert.equal(subject.localStats.value.length, 0)
  requests[1].resolve({ local: { stats: [{ symbol: 'NEW' }], total: 42 } })
  await second
  assert.equal(subject.localStats.value[0].symbol, 'NEW')
  assert.equal(subject.sourcesLoading.value, false)
})

test('Market returns to the last valid page after the group count shrinks', async () => {
  const requests = []
  const subject = view({ getMarketSources: async (params) => {
    requests.push({ ...params })
    return { local: { stats: params.page > 2 ? [] : [{ symbol: 'LAST' }], total: 21 } }
  } })
  await subject.changeStatsPage(3)
  assert.deepEqual(requests.map((request) => request.page), [3, 2])
  assert.equal(subject.statsPage.value, 2)
  assert.equal(subject.localStats.value[0].symbol, 'LAST')
})

test('Market does not apply page data or issue a fallback request after unmount', async () => {
  const response = deferred()
  let requests = 0
  const subject = view({ getMarketSources: () => { requests += 1; return response.promise } })
  const pending = subject.changeStatsPage(5)
  subject.unmount()
  response.resolve({ local: { stats: [], total: 1 } })
  await pending
  await tick()
  assert.equal(requests, 1)
  assert.equal(subject.localStats.value.length, 0)
})
