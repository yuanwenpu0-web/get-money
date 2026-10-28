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

function view(name, api, exports) {
  const source = readFileSync(new URL(`../src/views/${name}.vue`, import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1]
    .replace(/^import .*$/gm, '')
  const unmounted = []
  let timers = 0
  const context = vm.createContext({
    api,
    ref: (value) => ({ value }), reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {}, onBeforeUnmount: (callback) => unmounted.push(callback),
    ElMessage: { error: (message) => assert.fail(message), success: () => {} },
    setTimeout: () => { timers += 1; return timers }, clearTimeout: () => {},
    window: { scrollTo: () => {} },
  })
  vm.runInContext(`${source}\nglobalThis.subject = { ${exports.join(', ')} }`, context)
  return { ...context.subject, unmount: () => unmounted.forEach((callback) => callback()), timerCount: () => timers }
}

test('Market displays candles while settings, sources and symbols are still pending', async () => {
  const settings = deferred(), sources = deferred(), symbols = deferred(), candles = deferred()
  const requests = []
  const subject = view('Market', {
    getSettings: () => { requests.push('settings'); return settings.promise },
    getMarketSources: () => { requests.push('sources'); return sources.promise },
    getMarketSymbols: () => { requests.push('symbols'); return symbols.promise },
    getKlines: () => { requests.push('candles'); return candles.promise },
  }, ['init', 'bars', 'loading', 'settingsReady', 'sourcesLoading'])
  const pending = subject.init()
  assert.deepEqual(requests, ['sources', 'symbols', 'candles', 'settings'])
  candles.resolve({ bars: [{ close: 100 }] })
  await tick()
  assert.equal(subject.bars.value[0].close, 100)
  assert.equal(subject.loading.value, false)
  assert.equal(subject.sourcesLoading.value, true)
  assert.equal(subject.settingsReady.value, false)
  settings.resolve({ symbols: ['BTCUSDT'], interval: '1m' })
  sources.resolve({ local: { stats: [] } })
  symbols.resolve({ symbols: [] })
  await pending
})

test('Market switches untouched defaults after settings arrive and ignores the obsolete candle response', async () => {
  const settings = deferred()
  const requests = []
  const subject = view('Market', {
    getSettings: () => settings.promise,
    getMarketSources: async () => ({}), getMarketSymbols: async () => ({ symbols: [] }),
    getKlines(symbol, interval) {
      const response = deferred()
      requests.push({ symbol, interval, ...response })
      return response.promise
    },
  }, ['init', 'bars', 'query', 'loading'])
  const pending = subject.init()
  settings.resolve({ symbols: ['ETHUSDT'], interval: '1h' })
  await tick()
  assert.equal(requests.length, 2)
  assert.equal(requests[1].symbol, 'ETHUSDT')
  assert.equal(requests[1].interval, '1h')
  requests[1].resolve({ bars: [{ close: 200 }] })
  await tick()
  requests[0].resolve({ bars: [{ close: 100 }] })
  await pending
  assert.equal(subject.query.symbol, 'ETHUSDT')
  assert.equal(subject.bars.value[0].close, 200)
  assert.equal(subject.loading.value, false)
})

test('Market preserves a user selection when slow settings arrive', async () => {
  const settings = deferred()
  const requests = []
  const subject = view('Market', {
    getSettings: () => settings.promise,
    getMarketSources: async () => ({}), getMarketSymbols: async () => ({ symbols: [] }),
    getKlines: async (symbol, interval) => { requests.push([symbol, interval]); return { bars: [] } },
  }, ['init', 'query', 'changeQuery'])
  const pending = subject.init()
  subject.query.symbol = 'SOLUSDT'
  subject.query.interval = '4h'
  await subject.changeQuery()
  settings.resolve({ symbols: ['ETHUSDT'], interval: '1h' })
  await pending
  assert.equal(subject.query.symbol, 'SOLUSDT')
  assert.equal(subject.query.interval, '4h')
  assert.equal(requests.length, 2)
})

test('Market source failure does not prevent candles from appearing', async () => {
  const subject = view('Market', {
    getSettings: async () => ({ symbols: ['BTCUSDT'] }),
    getMarketSources: async () => { throw new Error('stats unavailable') },
    getMarketSymbols: async () => ({ symbols: [] }),
    getKlines: async () => ({ bars: [{ close: 123 }] }),
  }, ['init', 'bars', 'sourcesError', 'settingsReady'])
  await subject.init()
  assert.equal(subject.bars.value[0].close, 123)
  assert.equal(subject.sourcesError.value, 'stats unavailable')
  assert.equal(subject.settingsReady.value, true)
})

test('Market does not start a settings-triggered candle request after unmount', async () => {
  const settings = deferred()
  let requests = 0
  const subject = view('Market', {
    getSettings: () => settings.promise,
    getMarketSources: async () => ({}), getMarketSymbols: async () => ({ symbols: [] }),
    getKlines: async () => { requests += 1; return { bars: [] } },
  }, ['init'])
  const pending = subject.init()
  subject.unmount()
  settings.resolve({ symbols: ['ETHUSDT'], interval: '1h' })
  await pending
  assert.equal(requests, 1)
})

test('Backtest history appears before settings and submissions wait for settings', async () => {
  const settings = deferred(), history = deferred()
  let submissions = 0
  const subject = view('Backtest', {
    getSettings: () => settings.promise, listBacktests: () => history.promise,
    createBacktest: () => { submissions += 1; throw new Error('not ready') },
  }, ['loadBase', 'history', 'loadingHistory', 'settingsReady', 'submit'])
  const pending = subject.loadBase()
  history.resolve({ backtests: [{ id: 10 }] })
  await tick()
  assert.equal(subject.history.value[0].id, 10)
  assert.equal(subject.loadingHistory.value, false)
  assert.equal(subject.settingsReady.value, false)
  await subject.submit()
  assert.equal(submissions, 0)
  settings.resolve({ symbols: ['BTCUSDT'] })
  await pending
  assert.equal(subject.settingsReady.value, true)
})

test('Backtest late settings preserve edited fields while applying untouched defaults', async () => {
  const settings = deferred()
  const subject = view('Backtest', {
    getSettings: () => settings.promise, listBacktests: async () => ({ backtests: [] }),
  }, ['loadBase', 'form', 'touchField'])
  const pending = subject.loadBase()
  subject.form.fee_pct = 0.09
  subject.touchField('fee_pct')
  settings.resolve({ symbols: ['ETHUSDT'], interval: '1h', fee_rate: .0002, slippage: .0003 })
  await pending
  assert.equal(subject.form.fee_pct, .09)
  assert.equal(subject.form.symbol, 'ETHUSDT')
  assert.equal(subject.form.interval, '1h')
  assert.equal(subject.form.slippage_pct, .03)
})

test('Backtest history failure leaves loaded settings usable', async () => {
  const subject = view('Backtest', {
    getSettings: async () => ({ symbols: ['BTCUSDT'] }),
    listBacktests: async () => { throw new Error('history unavailable') },
  }, ['loadBase', 'settingsReady', 'historyError'])
  await subject.loadBase()
  assert.equal(subject.settingsReady.value, true)
  assert.equal(subject.historyError.value, 'history unavailable')
})

test('Backtest late poll responses do not restart polling after unmount', async () => {
  const response = deferred()
  const subject = view('Backtest', { getBacktest: () => response.promise }, ['poll', 'activeRun'])
  const pending = subject.poll(1)
  subject.unmount()
  response.resolve({ id: 1, status: 'running' })
  await pending
  assert.equal(subject.timerCount(), 0)
  assert.equal(subject.activeRun.value, null)
})

test('Backtest keeps the latest history selection when result requests finish out of order', async () => {
  const responses = [deferred(), deferred()]
  const subject = view('Backtest', {
    getBacktest: (id) => responses[id - 1].promise,
    getBacktestTrades: async () => ({ trades: [], total: 0 }),
  }, ['loadRun', 'result', 'loadingRun'])
  const first = subject.loadRun({ id: 1 })
  const second = subject.loadRun({ id: 2 })
  responses[1].resolve({ id: 2, status: 'done' })
  await second
  responses[0].resolve({ id: 1, status: 'done' })
  await first
  assert.equal(subject.result.value.id, 2)
  assert.equal(subject.loadingRun.value, false)
})
