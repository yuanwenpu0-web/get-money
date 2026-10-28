import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const BINDINGS = [
  { id: 3, strategy: 'absorption_reversal', pool: 'alt', interval: '5m', pool_name: '山寨币', enabled: true },
  { id: 4, strategy: 'absorption_reversal', pool: 'alt', interval: '15m', pool_name: '山寨币', enabled: false },
  { id: 5, strategy: 'absorption_reversal', pool: 'major', interval: '15m', pool_name: '主流币', enabled: true },
]

function view(overrides = {}) {
  const source = readFileSync(new URL('../src/views/Backtest.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const context = vm.createContext({
    api: {
      getSettings: async () => ({}),
      listBacktests: async () => ({ backtests: [], total: 0 }),
      getBindings: async () => ({ bindings: BINDINGS, total: BINDINGS.length }),
      getStrategies: async () => ({ strategies: [], specs: [{ code: 'absorption_reversal', name: '吸收反转' }] }),
      ...overrides,
    },
    ref: (value) => ({ value }), reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {}, onBeforeUnmount: () => {},
    ElMessage: { error: () => {}, success: () => {} },
    setTimeout: () => 1, clearTimeout: () => {},
    window: { scrollTo: () => {} },
  })
  vm.runInContext(`${source}\nglobalThis.subject = {
    loadBindings, applyBinding, onIntervalChange, onStrategyChange,
    form, bindings, strategyOptions, selectedBinding, strategyChoices, bindingOptions,
  }`, context)
  return context.subject
}

test('回测页按策略过滤绑定，选中后同步策略 × 池 × 周期', async () => {
  const subject = view({
    getBindings: async () => ({
      bindings: [...BINDINGS, { id: 9, strategy: 'other_strategy', pool: 'major', interval: '1h', enabled: true }],
    }),
  })
  await subject.loadBindings()
  assert.equal(subject.bindingOptions.value.length, 3)
  assert.ok(subject.bindingOptions.value.every((row) => row.strategy === subject.form.strategy))

  subject.applyBinding(3)
  assert.equal(subject.form.pool, 'alt')
  assert.equal(subject.form.interval, '5m')

  subject.applyBinding(5)
  assert.equal(subject.form.pool, 'major')
  assert.equal(subject.form.interval, '15m')
})

test('清空绑定选择时不再向回测传池参数', async () => {
  const subject = view()
  await subject.loadBindings()
  subject.applyBinding(3)
  assert.equal(subject.form.pool, 'alt')
  subject.applyBinding('')
  assert.equal(subject.form.pool, '')
})

test('手动换到没有绑定的周期时回落到策略默认参数层', async () => {
  const subject = view()
  await subject.loadBindings()
  subject.applyBinding(3)
  subject.form.interval = '1h'
  subject.onIntervalChange()
  assert.equal(subject.form.pool, '')
  assert.equal(subject.selectedBinding.value, '')
})

test('手动换到同池其它已绑定的周期时保留池选择', async () => {
  const subject = view()
  await subject.loadBindings()
  subject.selectedBinding.value = 3  // el-select 的 v-model 先落值，change 再回调
  subject.applyBinding(3)
  subject.form.interval = '15m'
  subject.onIntervalChange()
  assert.equal(subject.form.pool, 'alt')
  assert.equal(subject.selectedBinding.value, 3)
})

test('切换策略会清空原有绑定选择', async () => {
  const subject = view()
  await subject.loadBindings()
  subject.applyBinding(3)
  subject.onStrategyChange()
  assert.equal(subject.form.pool, '')
  assert.equal(subject.selectedBinding.value, '')
})

test('绑定与策略接口失败时退回策略默认参数，不阻塞回测', async () => {
  const subject = view({
    getBindings: async () => { throw new Error('offline') },
    getStrategies: async () => { throw new Error('offline') },
  })
  await subject.loadBindings()
  assert.equal(subject.bindings.value.length, 0)
  assert.equal(subject.strategyOptions.value.length, 0)
  assert.equal(subject.strategyChoices.value[0].code, 'absorption_reversal')
  assert.equal(subject.bindingOptions.value.length, 0)
})
