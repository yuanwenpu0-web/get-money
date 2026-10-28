import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

function deferred() {
  let resolve, reject
  const promise = new Promise((done, fail) => { resolve = done; reject = fail })
  return { promise, resolve, reject }
}

const tick = () => new Promise((resolve) => setImmediate(resolve))
const plain = (value) => JSON.parse(JSON.stringify(value))

function view(overrides = {}) {
  const source = readFileSync(new URL('../src/views/Backtest.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const unmounted = []
  const errors = []
  let timers = 0
  const context = vm.createContext({
    api: {
      getSettings: async () => ({}),
      listBacktests: async () => ({ backtests: [], total: 0, page: 1, page_size: 20 }),
      getBacktest: async (id) => ({ id, status: 'done', metrics: { total_trades: 41 }, equity_curve: [{ time: 1, equity: 100 }] }),
      getBacktestTrades: async () => ({ trades: [], total: 0, page: 1, page_size: 20 }),
      ...overrides,
    },
    ref: (value) => ({ value }), reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {}, onBeforeUnmount: (callback) => unmounted.push(callback),
    ElMessage: { error: (message) => errors.push(message), success: () => {} },
    setTimeout: () => { timers += 1; return timers }, clearTimeout: () => {},
    window: { scrollTo: () => {} },
  })
  vm.runInContext(`${source}\nglobalThis.subject = {
    loadBase, loadHistory, changeHistoryPage, changeHistorySize, history, historyPagination,
    loadRun, result, loadTrades, changeTradePage, changeTradeSize, trades, tradePagination,
    loadingTrades, tradesError, loadingRun, poll,
  }`, context)
  return { ...context.subject, errors, unmount: () => unmounted.forEach((callback) => callback()), timerCount: () => timers }
}

test('Backtest history requests only its page and resets page when page size changes', async () => {
  const requests = []
  const subject = view({
    listBacktests: async (params) => {
      requests.push(plain(params))
      return { backtests: [{ id: params.page }], total: 125, ...params }
    },
  })
  await subject.loadHistory()
  await subject.changeHistoryPage(3)
  assert.equal(subject.history.value[0].id, 3)
  await subject.changeHistorySize(50)
  assert.deepEqual(requests, [
    { page: 1, page_size: 20 }, { page: 3, page_size: 20 }, { page: 1, page_size: 50 },
  ])
  assert.equal(subject.historyPagination.total, 125)
  assert.equal(subject.historyPagination.page, 1)
})

test('An older history response cannot overwrite a newer page', async () => {
  const responses = [deferred(), deferred()]
  let calls = 0
  const subject = view({ listBacktests: () => responses[calls++].promise })
  const first = subject.loadHistory()
  const second = subject.changeHistoryPage(2)
  responses[1].resolve({ backtests: [{ id: 2 }], total: 60 })
  await second
  responses[0].resolve({ backtests: [{ id: 1 }], total: 60 })
  await first
  assert.equal(subject.historyPagination.page, 2)
  assert.equal(subject.history.value[0].id, 2)
})

test('History refresh returns from an empty removed page to the last valid page', async () => {
  const requests = []
  const subject = view({
    listBacktests: async (params) => {
      requests.push(plain(params))
      return { backtests: params.page === 2 ? [{ id: 7 }] : [], total: 25 }
    },
  })
  subject.historyPagination.page = 4
  await subject.loadHistory()
  assert.deepEqual(requests.map((request) => request.page), [4, 2])
  assert.equal(subject.historyPagination.page, 2)
  assert.equal(subject.history.value[0].id, 7)
})

test('Metrics and equity display before the separately paged trades arrive', async () => {
  const response = deferred()
  const requests = []
  const subject = view({
    getBacktest: async (id, params) => {
      requests.push(['details', id, plain(params)])
      return { id, status: 'done', metrics: { total_trades: 41 }, equity_curve: [{ equity: 123 }] }
    },
    getBacktestTrades: (id, params) => {
      requests.push(['trades', id, plain(params)])
      return response.promise
    },
  })
  await subject.loadRun({ id: 12 })
  assert.equal(subject.result.value.metrics.total_trades, 41)
  assert.equal(subject.result.value.equity_curve[0].equity, 123)
  assert.equal(subject.loadingRun.value, false)
  assert.equal(subject.loadingTrades.value, true)
  assert.deepEqual(requests, [
    ['details', 12, { include_trades: false }], ['trades', 12, { page: 1, page_size: 20 }],
  ])
  response.resolve({ trades: [{ entry_time: 'first' }], total: 41 })
  await tick()
  assert.equal(subject.trades.value[0].entry_time, 'first')
  assert.equal(subject.tradePagination.total, 41)
})

test('Trade and history pages are independent, and changing trade size returns to page one', async () => {
  const requests = []
  const subject = view({
    getBacktestTrades: async (id, params) => {
      requests.push([id, plain(params)])
      return { trades: [{ page: params.page }], total: 130 }
    },
  })
  subject.historyPagination.page = 3
  await subject.loadRun({ id: 9 })
  await tick()
  await subject.changeTradePage(2)
  await subject.changeTradeSize(50)
  assert.equal(subject.historyPagination.page, 3)
  assert.equal(subject.tradePagination.page, 1)
  assert.deepEqual(requests, [
    [9, { page: 1, page_size: 20 }], [9, { page: 2, page_size: 20 }], [9, { page: 1, page_size: 50 }],
  ])
})

test('Switching runs resets trades and ignores the prior run response', async () => {
  const responses = [deferred(), deferred()]
  const requests = []
  const subject = view({
    getBacktestTrades: (id, params) => {
      requests.push([id, plain(params)])
      return responses[id - 1].promise
    },
  })
  await subject.loadRun({ id: 1 })
  await subject.loadRun({ id: 2 })
  responses[1].resolve({ trades: [{ run: 2 }], total: 40 })
  await tick()
  responses[0].resolve({ trades: [{ run: 1 }], total: 80 })
  await tick()
  assert.equal(subject.result.value.id, 2)
  assert.equal(subject.trades.value[0].run, 2)
  assert.equal(subject.tradePagination.total, 40)
  assert.deepEqual(requests.map((request) => request[1].page), [1, 1])
})

test('Polling running backtests excludes trades and loads the first trade page once done', async () => {
  let done = false
  const requests = []
  const subject = view({
    getBacktest: async (id, params) => {
      requests.push(['details', plain(params)])
      return { id, status: done ? 'done' : 'running', metrics: {} }
    },
    getBacktestTrades: async (id, params) => {
      requests.push(['trades', plain(params)])
      return { trades: [], total: 0 }
    },
  })
  await subject.poll(3)
  assert.equal(subject.timerCount(), 1)
  assert.equal(requests.length, 1)
  done = true
  await subject.poll(3)
  await tick()
  assert.deepEqual(requests, [
    ['details', { include_trades: false }], ['details', { include_trades: false }],
    ['trades', { page: 1, page_size: 20 }],
  ])
})

test('A trade error is local to the table, retry is possible, and unmount ignores late rows', async () => {
  const response = deferred()
  let calls = 0
  const subject = view({
    getBacktestTrades: async () => {
      if (++calls === 1) throw new Error('trades unavailable')
      return response.promise
    },
  })
  await subject.loadRun({ id: 4 })
  await tick()
  assert.equal(subject.result.value.id, 4)
  assert.equal(subject.tradesError.value, 'trades unavailable')
  const pending = subject.loadTrades()
  subject.unmount()
  response.resolve({ trades: [{ id: 'late' }], total: 1 })
  await pending
  assert.equal(subject.trades.value.length, 0)
  assert.equal(subject.errors.length, 0)
})
