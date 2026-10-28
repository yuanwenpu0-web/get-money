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
    getOrders() { calls.push('orders'); return orders.promise },
  }, [])
  const pending = subject.mount()
  assert.deepEqual(calls, ['settings', 'orders'])
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
