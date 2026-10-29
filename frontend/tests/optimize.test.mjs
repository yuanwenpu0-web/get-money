import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const OPTIONS = {
  specs: [
    {
      code: 'absorption_reversal',
      name: '吸收反转',
      grid: { conf_mode: [0, 1], target_r: [1.5, 2, 3] },
      grid_size: 6,
    },
  ],
  pools: [{ code: 'major', name: '主流币', kind: 'rule' }],
  default_workers: 20,
  max_workers: 24,
  busy: false,
}

function view(overrides = {}) {
  const source = readFileSync(new URL('../src/views/Optimize.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1]
    .replace(/^import .*$/gm, '')
  const context = vm.createContext({
    api: {
      getOptimizationOptions: async () => OPTIONS,
      listOptimizations: async () => ({ optimizations: [], total: 0 }),
      getSymbolParams: async () => ({ total: 0, sources: {}, items: [] }),
      getOptimizationUniverse: async ({ interval, days }) => ({
        interval,
        days,
        count: 59,
        sample: ['BTCUSDT', 'ETHUSDT'],
      }),
      getOptimization: async () => ({ id: 1, status: 'done', progress: {}, summary: {} }),
      getOptimizationResults: async () => ({ results: [], total: 0 }),
      createOptimization: async () => ({ id: 1, symbol_count: 2, combos: 4, workers: 2 }),
      applyOptimization: async () => ({ mode: 'symbol', written: 2, message: 'ok' }),
      cancelOptimization: async () => ({ ok: true }),
      deleteOptimization: async () => ({ ok: true }),
      deleteSymbolParams: async () => ({ removed: 1 }),
      ...overrides,
    },
    ref: (value) => ({ value }),
    reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {},
    onBeforeUnmount: () => {},
    ElMessage: { error: () => {}, success: () => {}, warning: () => {}, info: () => {} },
    ElMessageBox: { confirm: async () => true },
    setInterval: () => 1,
    clearInterval: () => {},
  })
  vm.runInContext(
    `${source}
globalThis.subject = {
  form, options, runs, run, results, resultsTotal, filters, applyMode, applyOnly,
  spec, gridRows, progressPct, busy, running, canApply, verdictCounts, pooled, coverage,
  loadOptions, loadRuns, loadResults, loadCoverage, startOptimization, refresh,
  previewUniverse, universe,
  applyResults, buildStartPayload, buildApplyPayload, verdictMeta, paramsText,
}`,
    context,
  )
  return context.subject
}

test('启动参数按范围构造：全部 / 池 / 手填币种', async () => {
  const subject = view()
  await subject.loadOptions()

  subject.form.scope = 'all'
  const all = subject.buildStartPayload()
  assert.equal(all.pool, undefined)
  assert.equal(all.symbols, undefined)
  assert.equal(all.days, 90)
  assert.equal(all.train_ratio, 0.7)
  assert.equal(all.delta_source, 'auto')

  subject.form.scope = 'pool'
  subject.form.pool = ''
  assert.throws(() => subject.buildStartPayload(), /币种池/)

  subject.form.pool = 'major'
  assert.equal(subject.buildStartPayload().pool, 'major')

  subject.form.scope = 'symbols'
  subject.form.symbols = ' btcusdt, ethusdt  '
  assert.deepEqual([...subject.buildStartPayload().symbols], ['BTCUSDT', 'ETHUSDT'])

  subject.form.symbols = ''
  assert.throws(() => subject.buildStartPayload(), /至少一个币种/)
})

test('应用参数只带选中的模式与门槛', () => {
  const subject = view()
  subject.applyMode.value = 'symbol'
  subject.applyOnly.value = 'pass'
  assert.deepEqual({ ...subject.buildApplyPayload() }, { mode: 'symbol', only: 'pass' })
  assert.deepEqual(
    { ...subject.buildApplyPayload(['BTCUSDT']), symbols: [...subject.buildApplyPayload(['BTCUSDT']).symbols] },
    { mode: 'symbol', only: 'pass', symbols: ['BTCUSDT'] },
  )

  subject.applyMode.value = 'pool'
  subject.applyOnly.value = 'positive'
  assert.deepEqual({ ...subject.buildApplyPayload() }, { mode: 'pool', only: 'positive' })

  subject.form.pool = 'major'
  assert.equal(subject.buildApplyPayload().pool_prefix, undefined)
})

test('判定文案覆盖后端全部取值', () => {
  const subject = view()
  for (const key of ['pass', 'inconclusive', 'negative', 'insufficient', 'no_trades']) {
    const meta = subject.verdictMeta(key)
    assert.notEqual(meta.label, key, `${key} 应有中文文案`)
    assert.ok(meta.hint.length > 0, `${key} 应有解释`)
  }
  assert.equal(subject.verdictMeta('').label, '—')
  assert.equal(subject.verdictMeta('weird').label, 'weird')
})

test('参数摘要只显示与当前不同的键', () => {
  const subject = view()
  assert.equal(subject.paramsText({}), '沿用当前参数')
  assert.equal(subject.paramsText({ target_r: 3, conf_mode: 0 }), 'target_r=3 · conf_mode=0')
})

test('进度按完成数换算，缺失总量时不报错', async () => {
  const subject = view()
  assert.equal(subject.progressPct.value, 0)
  subject.run.value = { id: 1, status: 'running', progress: { total: 4, done: 1 } }
  assert.equal(subject.progressPct.value, 25)
  subject.run.value = { id: 1, status: 'done', progress: { total: 0, done: 0 } }
  assert.equal(subject.progressPct.value, 0)
})

test('启动时把表单换算成后端请求体', async () => {
  let captured = null
  const subject = view({
    createOptimization: async (payload) => {
      captured = payload
      return { id: 5, symbol_count: 2, combos: 16, workers: 2 }
    },
  })
  await subject.loadOptions()
  subject.form.scope = 'pool'
  subject.form.pool = 'major'
  subject.form.combos = 16
  subject.form.workers = 4
  await subject.startOptimization()
  assert.equal(captured.strategy, 'absorption_reversal')
  assert.equal(captured.pool, 'major')
  assert.equal(captured.combos, 16)
  assert.equal(captured.workers, 4)
  assert.equal(captured.symbols, undefined)
})

test('启动失败时给出提示且不进入轮询', async () => {
  const errors = []
  const subject = view({
    createOptimization: async () => {
      throw new Error('已有寻优任务在运行')
    },
  })
  await subject.loadOptions()
  await subject.startOptimization()
  assert.equal(subject.results.value.length, 0)
  assert.equal(subject.canApply.value, false)
  void errors
})

test('options 加载后默认选中第一个策略并给出搜索空间', async () => {
  const subject = view()
  await subject.loadOptions()
  assert.equal(subject.options.value.specs.length, 1)
  assert.equal(subject.spec.value.code, 'absorption_reversal')
  assert.deepEqual(
    [...subject.gridRows.value.map((row) => row.key)],
    ['conf_mode', 'target_r'],
  )
  assert.equal(subject.busy.value, false)
})

test('无任务时不允许应用', () => {
  const subject = view()
  assert.equal(subject.canApply.value, false)
  subject.run.value = { id: 1, status: 'running' }
  assert.equal(subject.canApply.value, false)
  subject.run.value = { id: 1, status: 'done' }
  assert.equal(subject.canApply.value, true)
})

test('扫描范围预览反映当前周期与天数', async () => {
  const subject = view()
  subject.form.interval = '1h'
  subject.form.days = 180
  await subject.previewUniverse()
  assert.equal(subject.universe.value.count, 59)
  assert.equal(subject.universe.value.interval, '1h')
  assert.equal(subject.universe.value.days, 180)
  assert.deepEqual([...subject.universe.value.sample], ['BTCUSDT', 'ETHUSDT'])
})

test('预览失败时不影响开始寻优', async () => {
  const subject = view({
    getOptimizationUniverse: async () => {
      throw new Error('boom')
    },
  })
  await subject.previewUniverse()
  assert.equal(subject.universe.value, null)
  const payload = subject.buildStartPayload()
  assert.equal(payload.interval, '15m')
})
