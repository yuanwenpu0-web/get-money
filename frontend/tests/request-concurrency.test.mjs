import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

function deferred() {
  let resolve
  const promise = new Promise((done) => { resolve = done })
  return { promise, resolve }
}

const tick = () => new Promise((resolve) => setImmediate(resolve))

function view(name, api, exports) {
  const source = readFileSync(new URL(`../src/views/${name}.vue`, import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1]
    .replace(/^import .*$/gm, '')
  const mounted = []
  const unmounted = []
  let timers = 0
  let clearedTimers = 0
  const context = vm.createContext({
    PAGE_SIZE: 15,
    api,
    ref: (value) => ({ value }),
    reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: (callback) => mounted.push(callback),
    onBeforeUnmount: (callback) => unmounted.push(callback),
    ElMessage: { error: (message) => assert.fail(message) },
    setInterval: () => { timers += 1; return timers },
    clearInterval: () => { clearedTimers += 1 },
  })
  vm.runInContext(`${source}\nglobalThis.subject = { ${exports.join(', ')} }`, context)
  return {
    ...context.subject,
    mount: () => Promise.all(mounted.map((callback) => callback())),
    unmount: () => unmounted.forEach((callback) => callback()),
    timerCount: () => timers,
    clearedTimerCount: () => clearedTimers,
  }
}

test('Strategy preview follows the selected binding and clears it when the indicator changes', async () => {
  let payload
  const subject = view('Strategies', {
    previewSignal: async (request) => { payload = request; return {} },
    // 预演下拉现在按策略维度取绑定（/strategies/{code}/bindings），不再依赖分页的 /bindings
    getStrategyBindings: async () => ({ bindings: [] }),
  }, ['form', 'bindings', 'previewBinding', 'onPreviewBindingChange', 'onPreviewStrategyChange', 'runPreview'])
  subject.form.strategy = 'first'
  subject.bindings.value = [{ id: 12, strategy: 'first', pool: 'major', interval: '1h' }]
  subject.previewBinding.value = 12
  subject.onPreviewBindingChange(12)
  await subject.runPreview()
  assert.equal(payload.pool, 'major')
  assert.equal(payload.interval, '1h')
  subject.form.strategy = 'second'
  subject.onPreviewStrategyChange()
  await subject.runPreview()
  assert.equal(payload.strategy, 'second')
  assert.equal(payload.pool, null)
  assert.equal(subject.previewBinding.value, '')
})

test('Strategy page keeps binding management and removes independent indicator controls', () => {
  const source = readFileSync(new URL('../src/views/Strategies.vue', import.meta.url), 'utf8')
  assert.match(source, /<BindingMatrix :strategies="strategies" @change="loadBindings"\s*\/>/)
  assert.match(source, /@click="runCycle"/)
  assert.match(source, /@click="runPreview\(\)"/)
  assert.match(source, /信号指标加载失败[\s\S]*?@click="loadStrategies"/)
  assert.doesNotMatch(source, /strategy-card|api\.updateStrategy|openEdit\(|editVisible/)
})

test('Orders serializes refreshes and keeps a refresh requested after filters change', async () => {
  const requests = []
  const subject = view('Orders', {
    getOrders(params) {
      const response = deferred()
      requests.push({ params, ...response })
      return response.promise
    },
  }, ['load', 'filters', 'orders'])

  const first = subject.load()
  subject.filters.symbol = 'ETHUSDT'
  const second = subject.load()
  assert.equal(first, second)
  assert.equal(requests.length, 1)
  requests[0].resolve({ orders: [{ id: 1 }] })
  await tick()
  assert.equal(requests.length, 2)
  assert.equal(requests[1].params.symbol, 'ETHUSDT')
  requests[1].resolve({ orders: [{ id: 2 }] })
  await first
  assert.equal(subject.orders.value[0].id, 2)
})

test('Dashboard starts resources independently and does not restart them after unmounting', async () => {
  const response = deferred()
  let dashboardCalls = 0
  let symbolCalls = 0
  const subject = view('Dashboard', {
    getDashboard() { dashboardCalls += 1; return response.promise },
    getEquity: async () => ({ points: [] }),
    getMarketSymbols: async () => { symbolCalls += 1; return { symbols: [] } },
  }, ['load'])

  const pending = subject.load()
  subject.load()
  subject.unmount()
  response.resolve({ portfolio: {} })
  await pending
  await subject.load()
  assert.equal(dashboardCalls, 1)
  assert.equal(symbolCalls, 1)
})

test('Orders loads settings and orders together and clears its timer on unmount', async () => {
  const settings = deferred()
  const orders = deferred()
  const calls = []
  const subject = view('Orders', {
    getSettings() { calls.push('settings'); return settings.promise },
    getOrderStrategies() { calls.push('strategies'); return Promise.resolve({ strategies: [] }) },
    getOrders() { calls.push('orders'); return orders.promise },
  }, [])
  const pending = subject.mount()
  assert.deepEqual(calls, ['settings', 'strategies', 'orders'])
  subject.unmount()
  settings.resolve({ symbols: ['BTCUSDT'] })
  orders.resolve({ orders: [] })
  await pending
  assert.equal(subject.timerCount(), 1)
  assert.equal(subject.clearedTimerCount(), 1)
})

test('Market starts its chart before sources and symbols requests complete', async () => {
  const sources = deferred()
  const symbols = deferred()
  const bars = deferred()
  const settings = deferred()
  const calls = []
  const subject = view('Market', {
    getSettings: () => settings.promise,
    getMarketSources() { calls.push('sources'); return sources.promise },
    getMarketSymbols() { calls.push('symbols'); return symbols.promise },
    getKlines() { calls.push('bars'); return bars.promise },
  }, ['init'])
  const pending = subject.init()
  await tick()
  assert.deepEqual(calls, ['sources', 'symbols', 'bars'])
  settings.resolve({ symbols: ['BTCUSDT'], interval: '1m' })
  sources.resolve({})
  symbols.resolve({ symbols: ['BTCUSDT'] })
  bars.resolve({ bars: [] })
  await pending
})
