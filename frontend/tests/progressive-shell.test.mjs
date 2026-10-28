import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { shareRequest } from '../src/utils/shareRequest.js'

function deferred() {
  let resolve, reject
  const promise = new Promise((done, fail) => { resolve = done; reject = fail })
  return { promise, resolve, reject }
}

const tick = () => new Promise((resolve) => setImmediate(resolve))

function component(path, api, exported) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const hooks = { mount: [], unmount: [] }
  const context = vm.createContext({
    api, ref: (value) => ({ value }), reactive: (value) => value,
    computed: (fn) => ({ get value() { return fn() } }), watch: () => {},
    useRoute: () => ({ meta: {} }),
    onMounted: (fn) => hooks.mount.push(fn), onBeforeUnmount: (fn) => hooks.unmount.push(fn),
    setInterval: () => 1, clearInterval: () => {},
    ElMessage: { error: (message) => assert.fail(message) },
  })
  vm.runInContext(`${source}\nglobalThis.subject = { ${exported.join(', ')} }`, context)
  return { ...context.subject,
    mount: () => hooks.mount.forEach((fn) => fn()),
    unmount: () => hooks.unmount.forEach((fn) => fn()),
  }
}

test('App status and symbol count load independently without inventing an execution mode', async () => {
  const settings = deferred()
  const subject = component('../src/App.vue', {
    getSettings: () => settings.promise,
    getMarketSymbols: async () => ({ symbols: ['BTCUSDT', 'ETHUSDT'] }),
  }, ['symbolCount', 'statusLoaded', 'executionModeLabel'])
  subject.mount()
  await tick()
  assert.equal(subject.symbolCount.value, 2)
  assert.equal(subject.statusLoaded.value, false)
  assert.equal(subject.executionModeLabel.value, '加载中')
  settings.resolve({ execution_mode: 'live', testnet: true })
  await tick()
  assert.equal(subject.executionModeLabel.value, 'TESTNET LIVE')
  subject.unmount()
})

test('Settings does not submit default values before saved settings arrive', async () => {
  const settings = deferred()
  let writes = 0
  const subject = component('../src/views/Settings.vue', {
    getSettings: () => settings.promise,
    saveSettings: () => { writes += 1 }, saveCredentials: () => { writes += 1 },
    adjustCapital: () => { writes += 1 }, testCredentials: () => { writes += 1 },
  }, ['load', 'settingsReady', 'loading', 'saveSettings', 'saveCredentials', 'adjustCapital', 'testCredentials'])
  const pending = subject.load()
  await Promise.all([subject.saveSettings(), subject.saveCredentials(), subject.adjustCapital(), subject.testCredentials()])
  assert.equal(writes, 0)
  assert.equal(subject.loading.value, true)
  assert.equal(subject.settingsReady.value, false)
  settings.resolve({ execution_mode: 'paper', symbols: ['ETHUSDT'] })
  await pending
  assert.equal(subject.loading.value, false)
  assert.equal(subject.settingsReady.value, true)
})

test('Settings can retry a failed initial read without blocking the page', async () => {
  let attempts = 0
  const subject = component('../src/views/Settings.vue', {
    getSettings: async () => {
      if (++attempts === 1) throw new Error('offline')
      return { execution_mode: 'paper' }
    },
  }, ['load', 'loadError', 'settingsReady'])
  await subject.load()
  assert.equal(subject.loadError.value, 'offline')
  assert.equal(subject.settingsReady.value, false)
  await subject.load()
  assert.equal(subject.loadError.value, '')
  assert.equal(subject.settingsReady.value, true)
})

test('Concurrent shared reads deduplicate only pending work, with fresh reads after completion', async () => {
  const response = deferred()
  let calls = 0
  const read = shareRequest(() => { calls += 1; return response.promise })
  const first = read(), second = read()
  assert.equal(first, second)
  await tick()
  assert.equal(calls, 1)
  response.resolve({ symbols: ['BTCUSDT'] })
  await first
  assert.deepEqual(await read(), { symbols: ['BTCUSDT'] })
  assert.equal(calls, 2)
})

test('Failed shared reads can be retried rather than retaining a rejected request', async () => {
  let calls = 0
  const read = shareRequest(async () => {
    if (++calls === 1) throw new Error('offline')
    return 'ready'
  })
  const first = read(), second = read()
  assert.equal(first, second)
  await assert.rejects(first, /offline/)
  assert.equal(await read(), 'ready')
  assert.equal(calls, 2)
})

test('Invalidation starts a fresh read and an older completion cannot clear it', async () => {
  const old = deferred(), fresh = deferred()
  let calls = 0
  const read = shareRequest(() => (++calls === 1 ? old.promise : fresh.promise))
  const oldRead = read()
  await tick()
  read.invalidate()
  const freshRead = read()
  assert.notEqual(freshRead, oldRead)
  await tick()
  old.resolve('old settings')
  await oldRead
  assert.equal(read(), freshRead)
  assert.equal(calls, 2)
  fresh.resolve('new settings')
  assert.equal(await freshRead, 'new settings')
})
