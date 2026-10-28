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

function view(name, api, exported) {
  const source = readFileSync(new URL(`../src/views/${name}.vue`, import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1]
    .replace(/^import .*$/gm, '')
  const mounted = [], unmounted = []
  let timers = 0, cleared = 0
  const context = vm.createContext({
    api,
    ref: (value) => ({ value }),
    reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: (callback) => mounted.push(callback),
    onBeforeUnmount: (callback) => unmounted.push(callback),
    ElMessage: { error: (message) => assert.fail(message) },
    setInterval: () => { timers += 1; return timers },
    clearInterval: () => { cleared += 1 },
  })
  vm.runInContext(`${source}\nglobalThis.subject = { ${exported.join(', ')} }`, context)
  return {
    ...context.subject,
    mount: () => Promise.all(mounted.map((callback) => callback())),
    unmount: () => unmounted.forEach((callback) => callback()),
    timers: () => ({ created: timers, cleared }),
  }
}

test('Dashboard shows account data while the equity request is still pending', async () => {
  const account = deferred(), points = deferred()
  const subject = view('Dashboard', {
    getDashboard: () => account.promise,
    getEquity: () => points.promise,
    getMarketSymbols: async () => ({ symbols: ['BTCUSDT'] }),
  }, ['load', 'dashboard', 'equity', 'dashboardLoading', 'equityLoading'])
  const pending = subject.load()
  account.resolve({ portfolio: { equity: 12345 } })
  await tick()
  assert.equal(subject.dashboard.value.portfolio.equity, 12345)
  assert.equal(subject.dashboardLoading.value, false)
  assert.equal(subject.equityLoading.value, true)
  assert.equal(subject.equity.value.length, 0)
  points.resolve({ points: [{ equity: 12345 }] })
  await pending
  assert.equal(subject.equity.value[0].equity, 12345)
})

test('Dashboard shows the chart independently and isolates account failure', async () => {
  const account = deferred()
  const subject = view('Dashboard', {
    getDashboard: () => account.promise,
    getEquity: async () => ({ points: [{ equity: 100 }, { equity: 101 }] }),
    getMarketSymbols: async () => ({ symbols: [] }),
  }, ['load', 'dashboard', 'equity', 'dashboardLoading', 'equityLoading', 'dashboardError', 'equityError'])
  const pending = subject.load()
  await tick()
  assert.equal(subject.equity.value.length, 2)
  assert.equal(subject.equityLoading.value, false)
  assert.equal(subject.dashboardLoading.value, true)
  account.reject(new Error('account offline'))
  await pending
  assert.equal(subject.dashboard.value, null)
  assert.equal(subject.dashboardError.value, 'account offline')
  assert.equal(subject.equityError.value, '')
  assert.equal(subject.equity.value.length, 2)
})

test('Dashboard account refresh can finish again while an older chart is pending', async () => {
  const points = deferred()
  let calls = 0
  const subject = view('Dashboard', {
    getDashboard: async () => ({ portfolio: { equity: ++calls } }),
    getEquity: () => points.promise,
    getMarketSymbols: async () => ({ symbols: [] }),
  }, ['load', 'loadDashboard', 'dashboard', 'equityLoading'])
  const pending = subject.load()
  await tick()
  await subject.loadDashboard()
  assert.equal(subject.dashboard.value.portfolio.equity, 2)
  assert.equal(subject.equityLoading.value, true)
  points.resolve({ points: [] })
  await pending
})

test('Dashboard merges each resource refresh and stops queued work after unmount', async () => {
  const account = deferred(), points = deferred(), symbols = deferred()
  const calls = { account: 0, points: 0, symbols: 0 }
  const subject = view('Dashboard', {
    getDashboard() { calls.account += 1; return account.promise },
    getEquity() { calls.points += 1; return points.promise },
    getMarketSymbols() { calls.symbols += 1; return symbols.promise },
  }, ['load', 'dashboard', 'equity', 'symbolCount'])
  const pending = subject.load()
  subject.load()
  subject.unmount()
  account.resolve({ portfolio: { equity: 100 } })
  points.resolve({ points: [{ equity: 100 }] })
  symbols.resolve({ symbols: ['BTCUSDT'] })
  await pending
  await subject.load()
  assert.deepEqual(calls, { account: 1, points: 1, symbols: 1 })
  assert.equal(subject.dashboard.value, null)
  assert.equal(subject.equity.value.length, 0)
  assert.equal(subject.symbolCount.value, 0)
})

test('Orders render before settings finish and refresh polling starts immediately', async () => {
  const settings = deferred(), orders = deferred()
  const subject = view('Orders', {
    getSettings: () => settings.promise,
    getOrders: () => orders.promise,
  }, ['orders', 'ordersLoaded', 'loading', 'settingsLoading'])
  await subject.mount()
  orders.resolve({ orders: [{ id: 7, symbol: 'BTCUSDT' }] })
  await tick()
  assert.equal(subject.orders.value[0].id, 7)
  assert.equal(subject.ordersLoaded.value, true)
  assert.equal(subject.loading.value, false)
  assert.equal(subject.settingsLoading.value, true)
  subject.unmount()
  settings.resolve({ symbols: ['BTCUSDT'] })
  await tick()
  assert.deepEqual(subject.timers(), { created: 1, cleared: 1 })
})

test('Orders keep loaded data when optional settings fail', async () => {
  const subject = view('Orders', {
    getSettings: async () => { throw new Error('settings offline') },
    getOrders: async () => ({ orders: [{ id: 8, symbol: 'ETHUSDT' }] }),
  }, ['load', 'loadSettings', 'orders', 'ordersLoaded', 'ordersError', 'settingsError'])
  await Promise.all([subject.load(), subject.loadSettings()])
  assert.equal(subject.orders.value[0].id, 8)
  assert.equal(subject.ordersLoaded.value, true)
  assert.equal(subject.ordersError.value, '')
  assert.equal(subject.settingsError.value, 'settings offline')
})
