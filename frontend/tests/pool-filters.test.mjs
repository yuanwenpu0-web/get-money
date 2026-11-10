import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

test('研究币池维护说明固定筛选口径并保留维护入口', () => {
  const source = readFileSync(new URL('../src/views/Strategies.vue', import.meta.url), 'utf8')
  assert.match(source, /合格币种至少 100 个/)
  assert.match(source, /成交额前 100 中选取波动率前 50/)
  assert.match(source, /成交额前 50 和后 50/)
  assert.match(source, /@click="refreshResearchMaintenance"/)
})

test('币池表单添加波动率窗口与严格成交额条件', () => {
  const source = readFileSync(new URL('../src/components/BindingMatrix.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const errors = []
  const context = vm.createContext({
    PAGE_SIZE: 15,
    defineProps: () => ({ strategies: [] }), defineEmits: () => () => {},
    ref: value => ({ value }), reactive: value => value, computed: fn => ({ get value() { return fn() } }),
    onMounted: () => {}, ElMessage: { error: e => errors.push(e) }, api: {},
  })
  vm.runInContext(`${source}\nglobalThis.subject = { poolForm, metricField, metricMin, metricMax, volatilityWindow, volatilityMaxAge, addMetricFilter }`, context)
  const s = context.subject
  s.metricField.value = 'volatility_1h'
  s.metricMin.value = '0.01'
  s.volatilityWindow.value = 48
  s.volatilityMaxAge.value = 3
  s.addMetricFilter()
  s.metricField.value = 'quote_volume_24h'
  s.metricMin.value = '1000'
  s.addMetricFilter()
  const rule = JSON.parse(s.poolForm.rule)
  assert.equal(rule.filters.volatility_1h.min, 0.01)
  assert.equal(rule.filters.quote_volume_24h.min, 1000)
  assert.equal(rule.volatility_window_hours, 48)
  assert.equal(rule.volatility_max_age_hours, 3)
  assert.deepEqual(errors, [])
})
