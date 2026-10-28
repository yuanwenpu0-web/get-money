import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const tick = () => new Promise((resolve) => setImmediate(resolve))

function deferred() {
  let resolve
  let reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function view(name, api, exports) {
  const text = readFileSync(new URL(`../src/views/${name}.vue`, import.meta.url), 'utf8')
  assert.match(text, /<template>\s*<div>/)
  const script = text.match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const mounts = []
  const unmounts = []
  const context = vm.createContext({
    api,
    ref: (value) => ({ value }),
    reactive: (value) => value,
    onMounted: (callback) => mounts.push(callback),
    onBeforeUnmount: (callback) => unmounts.push(callback),
    ElMessage: { error: (message) => assert.fail(message), success: () => {} },
    number: (value) => String(value),
  })
  vm.runInContext(`${script}\nglobalThis.subject = { ${exports.join(', ')} }`, context)
  return {
    ...context.subject,
    mount: () => Promise.all(mounts.map((callback) => callback())),
    unmount: () => unmounts.forEach((callback) => callback()),
  }
}

test('Models renders the completed list while settings remain pending and guards training', async () => {
  const list = deferred()
  const settings = deferred()
  let trainCalls = 0
  const subject = view('Models', {
    getModels: () => list.promise,
    getSettings: () => settings.promise,
    trainModel: () => { trainCalls += 1 },
  }, ['models', 'loading', 'settingsLoading', 'settingsReady', 'form', 'train'])
  const pending = subject.mount()
  list.resolve({ models: [{ id: 7 }] })
  await tick()
  assert.equal(subject.models.value[0].id, 7)
  assert.equal(subject.loading.value, false)
  assert.equal(subject.settingsLoading.value, true)
  assert.equal(subject.settingsReady.value, false)
  await subject.train()
  assert.equal(trainCalls, 0)
  settings.resolve({ symbols: ['SOLUSDT'], interval: '1h', auto_train_days: 60, train_horizon: 9 })
  await pending
  assert.equal(subject.form.symbol, 'SOLUSDT')
  assert.equal(subject.form.days, 60)
  assert.equal(subject.settingsReady.value, true)
})

test('Models keeps a list visible after settings failure and retries only that section', async () => {
  const settings = deferred()
  let settingsCalls = 0
  let listCalls = 0
  const subject = view('Models', {
    getModels: async () => { listCalls += 1; return { models: [{ id: 1 }] } },
    getSettings: () => {
      settingsCalls += 1
      return settingsCalls === 1 ? settings.promise : Promise.resolve({ symbols: ['ETHUSDT'], interval: '4h' })
    },
  }, ['models', 'settingsReady', 'settingsError', 'loadSettings'])
  const pending = subject.mount()
  settings.reject(new Error('settings timeout'))
  await pending
  assert.equal(subject.models.value.length, 1)
  assert.equal(subject.settingsError.value, 'settings timeout')
  assert.equal(subject.settingsReady.value, false)
  await subject.loadSettings()
  assert.equal(subject.settingsReady.value, true)
  assert.equal(subject.settingsError.value, '')
  assert.equal(listCalls, 1)
})

test('Models settings response does not overwrite form input or post-unmount data', async () => {
  const settings = deferred()
  const subject = view('Models', {
    getModels: async () => ({ models: [] }),
    getSettings: () => settings.promise,
  }, ['form', 'settingsReady'])
  const pending = subject.mount()
  subject.form.symbol = 'XRPUSDT'
  subject.form.days = 90
  settings.resolve({ symbols: ['ETHUSDT'], interval: '1h', auto_train_days: 60 })
  await pending
  assert.equal(subject.form.symbol, 'XRPUSDT')
  assert.equal(subject.form.days, 90)

  const late = deferred()
  const unmounted = view('Models', {
    getModels: () => late.promise,
    getSettings: async () => ({}),
  }, ['models'])
  const loading = unmounted.mount()
  unmounted.unmount()
  late.resolve({ models: [{ id: 99 }] })
  await loading
  assert.equal(unmounted.models.value.length, 0)
})

test('Models refreshes again after training when an older list request is still pending', async () => {
  const oldList = deferred()
  let listCalls = 0
  const subject = view('Models', {
    getModels: () => {
      listCalls += 1
      return listCalls === 1 ? oldList.promise : Promise.resolve({ models: [{ id: 2 }] })
    },
    getSettings: async () => ({}),
    trainModel: async () => ({ id: 2, metrics: {} }),
  }, ['models', 'train'])
  const initial = subject.mount()
  await tick()
  const training = subject.train()
  await tick()
  oldList.resolve({ models: [{ id: 1 }] })
  await Promise.all([initial, training])
  assert.equal(listCalls, 2)
  assert.equal(subject.models.value[0].id, 2)
})

test('Strategies starts all sections together and displays strategies before other sections', async () => {
  const settings = deferred()
  const symbols = deferred()
  const list = deferred()
  const started = []
  let writeCalls = 0
  const subject = view('Strategies', {
    getStrategies: () => { started.push('strategies'); return list.promise },
    getSettings: () => { started.push('settings'); return settings.promise },
    getMarketSymbols: () => { started.push('symbols'); return symbols.promise },
    saveSettings: () => { writeCalls += 1 },
    runCycle: () => { writeCalls += 1 },
  }, ['strategies', 'loading', 'settingsLoading', 'symbolsLoading', 'settingsReady', 'saveSchedule', 'runCycle'])
  const pending = subject.mount()
  assert.deepEqual(started, ['strategies', 'settings', 'symbols'])
  list.resolve({ strategies: [{ code: 'trend_follow' }] })
  await tick()
  assert.equal(subject.strategies.value[0].code, 'trend_follow')
  assert.equal(subject.loading.value, false)
  assert.equal(subject.settingsLoading.value, true)
  assert.equal(subject.symbolsLoading.value, true)
  await subject.saveSchedule()
  await subject.runCycle()
  assert.equal(writeCalls, 0)
  symbols.resolve({ symbols: [] })
  settings.resolve({ symbols: ['SOLUSDT'], interval: '1h' })
  await pending
  assert.equal(subject.settingsReady.value, true)
})

test('Strategies isolates a failed list from successful settings and symbol data', async () => {
  const list = deferred()
  const subject = view('Strategies', {
    getStrategies: () => list.promise,
    getSettings: async () => ({ symbols: ['ETHUSDT'], interval: '1h', schedule_enabled: false, schedule_seconds: 120 }),
    getMarketSymbols: async () => ({ symbols: ['BTCUSDT', 'ETHUSDT'] }),
  }, ['listError', 'settingsReady', 'form', 'symbolCount', 'scheduleForm'])
  const pending = subject.mount()
  await tick()
  assert.equal(subject.settingsReady.value, true)
  assert.equal(subject.form.symbol, 'ETHUSDT')
  assert.equal(subject.symbolCount.value, 2)
  assert.equal(subject.scheduleForm.seconds, 120)
  list.reject(new Error('strategy timeout'))
  await pending
  assert.equal(subject.listError.value, 'strategy timeout')
})

test('Strategies late settings preserve edits and section requests are deduplicated', async () => {
  const settings = deferred()
  let settingsCalls = 0
  const subject = view('Strategies', {
    getStrategies: async () => ({ strategies: [] }),
    getSettings: () => { settingsCalls += 1; return settings.promise },
    getMarketSymbols: async () => ({ symbols: [] }),
  }, ['form', 'scheduleForm', 'loadSettings'])
  const pending = subject.mount()
  subject.loadSettings()
  assert.equal(settingsCalls, 1)
  subject.form.symbol = 'XRPUSDT'
  subject.scheduleForm.seconds = 180
  settings.resolve({ symbols: ['ETHUSDT'], interval: '1h', schedule_enabled: false, schedule_seconds: 120 })
  await pending
  assert.equal(subject.form.symbol, 'XRPUSDT')
  assert.equal(subject.scheduleForm.seconds, 180)
})
