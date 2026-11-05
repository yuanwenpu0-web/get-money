import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { barsToCover, intervalMs, KLINE_MAX_LIMIT, locateBar, tradeMarker, zoomForIndexes } from '../src/utils/kline.js'

const MIN5 = 300_000
const tick = () => new Promise((resolve) => setImmediate(resolve))

function ordersView(api) {
  const source = readFileSync(new URL('../src/views/Orders.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const messages = []
  const noop = () => {}
  const context = vm.createContext({
    api,
    ref: (value) => ({ value }),
    reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: noop,
    onBeforeUnmount: noop,
    ElMessage: { info: (text) => messages.push(text), error: (text) => messages.push(text) },
    ElMessageBox: { confirm: async () => {} },
    EChart: {},
    price: (value) => (typeof value === 'number' ? value.toFixed(2) : '—'),
    tsLabel: (value) => String(value),
    money: (value) => String(value),
    pct: (value) => `${value}%`,
    pnlClass: () => '',
    sideLabel: (value) => value,
    statusLabel: (value) => value,
    strategyLabel: (value) => value,
    closeReasonLabel: (value) => value,
    barsToCover, intervalMs, KLINE_MAX_LIMIT, locateBar, tradeMarker, zoomForIndexes,
  })
  vm.runInContext(
    `${source}\nglobalThis.subject = { loadChartBars, loadOlderChartBars, onChartZoom, ensureEntryInWindow, autoZoom, focusEntry, focusExit, switchInterval, prependOlder, chartBars, chartZoom, chartHasMore, chartOlderLoading, chartError, chartLoading, entryPoint, exitPoint, chartOrder, chartInterval, MAX_AUTO_PAGES }`,
    context,
  )
  return { ...context.subject, messages }
}

function page(start, interval, count) {
  const step = intervalMs(interval)
  return Array.from({ length: count }, (_, index) => ({
    time: start + index * step,
    open: 100, high: 101, low: 99, close: 100.5, volume: 1,
  }))
}

test('limit 用 barsToCover 计算且不超过 1500，开平仓都落在窗口内', async () => {
  const base = Date.parse('2026-11-01T00:00:00Z')
  const calls = []
  const api = {
    getKlines: async (symbol, interval, limit) => {
      calls.push({ symbol, interval, limit })
      return { bars: page(base, interval, 600) }
    },
  }
  const subject = ordersView(api)
  subject.chartOrder.value = {
    id: 1, status: 'CLOSED', side: 'LONG',
    created_at: new Date(base).toISOString(),
    entry_price: 100, close_price: 110, closed_at: new Date(base + 100 * MIN5).toISOString(),
  }
  subject.chartInterval.value = '5m'
  await subject.loadChartBars()
  assert.ok(calls.length >= 1)
  assert.ok(calls[0].limit >= 240 && calls[0].limit <= 1500, `limit within bounds, got ${calls[0].limit}`)
  assert.equal(subject.chartBars.value.length, 600)
})

test('切换 5m 后仍定位到开仓/平仓，并自动把两者放进视窗', async () => {
  const base = Date.parse('2026-11-01T00:00:00Z')
  const api = {
    getKlines: async (symbol, interval) => ({ bars: page(base, interval, 600) }),
  }
  const subject = ordersView(api)
  subject.chartOrder.value = {
    id: 1, status: 'CLOSED', side: 'LONG',
    created_at: new Date(base + 37 * MIN5).toISOString(),
    entry_price: 100, close_price: 110, closed_at: new Date(base + 100 * MIN5).toISOString(),
  }
  subject.chartInterval.value = '1h'
  await subject.loadChartBars()

  // 先确认 1h 下能定位
  assert.ok(subject.entryPoint.value && subject.exitPoint.value, '1h 下开平仓都应定位到')

  // 切到 5m：标记不能消失，且自动定位仍覆盖两者
  await subject.switchInterval('5m')
  assert.ok(subject.entryPoint.value, '切到 5m 后开仓点仍在')
  assert.ok(subject.exitPoint.value, '切到 5m 后平仓点仍在')
  assert.equal(subject.entryPoint.value.index, 37)
  assert.equal(subject.exitPoint.value.index, 100)
  const total = subject.chartBars.value.length
  const startIdx = (subject.chartZoom.value.start / 100) * total
  const endIdx = (subject.chartZoom.value.end / 100) * total
  assert.ok(subject.entryPoint.value.index >= startIdx - 1 && subject.entryPoint.value.index <= endIdx + 1, '开仓在视窗内')
  assert.ok(subject.exitPoint.value.index >= startIdx - 1 && subject.exitPoint.value.index <= endIdx + 1, '平仓在视窗内')
})

test('开仓早于本地最早一根时自动翻更早分页，加载完再定位', async () => {
  const base = Date.now() - 5 * MIN5
  const calls = []
  // 第一屏从 B+200m 起（不含 entry）；第一页往前挪到 B+50m（仍不含）；第二页才覆盖到 entry。
  const api = {
    getKlines: async (symbol, interval, limit, end) => {
      calls.push({ end })
      if (end === undefined) return { bars: page(base + 200 * MIN5, '5m', 100) }
      if (end >= base + 50 * MIN5) return { bars: page(base + 50 * MIN5, '5m', 100) }
      return { bars: page(base - 100 * MIN5, '5m', 100) }
    },
  }
  const subject = ordersView(api)
  subject.chartOrder.value = {
    id: 1, status: 'OPEN', side: 'LONG',
    created_at: new Date(base).toISOString(), entry_price: 100, sl: 0, tp: 0,
  }
  subject.chartInterval.value = '5m'
  await subject.loadChartBars()
  // 初始 1 次 + 自动翻页 2 次（覆盖到 entry 后停手）
  assert.equal(calls.length, 3, `请求次数合理，got ${calls.length}`)
  assert.ok(subject.entryPoint.value, '自动翻页后开仓点应定位到')
  assert.equal(subject.chartBars.value.length, 300)
  // 视窗应覆盖开仓点
  const total = subject.chartBars.value.length
  const startIdx = (subject.chartZoom.value.start / 100) * total
  const endIdx = (subject.chartZoom.value.end / 100) * total
  assert.ok(subject.entryPoint.value.index >= startIdx - 1 && subject.entryPoint.value.index <= endIdx + 1)
})

test('自动翻页达到上限后停止，不无限请求', async () => {
  // 每次翻页都返回满页且起点持续前移（有进展），但始终晚于 entry，验证最多 MAX_AUTO_PAGES 次后停手。
  const base = Date.now() - 5 * MIN5
  const calls = []
  const api = {
    getKlines: async (symbol, interval, limit, end) => {
      calls.push(1)
      let start = base + 200 * MIN5
      if (end !== undefined) {
        if (end >= base + 150 * MIN5) start = base + 150 * MIN5
        else if (end >= base + 100 * MIN5) start = base + 100 * MIN5
        else start = base + 50 * MIN5
      }
      return { bars: page(start, '5m', 600) }
    },
  }
  const subject = ordersView(api)
  subject.chartOrder.value = {
    id: 1, status: 'OPEN', side: 'LONG',
    created_at: new Date(base).toISOString(), entry_price: 100, sl: 0, tp: 0,
  }
  subject.chartInterval.value = '5m'
  await subject.loadChartBars()
  assert.equal(calls.length, 1 + subject.MAX_AUTO_PAGES, '达到翻页上限后停止')
  assert.equal(subject.entryPoint.value, null, '始终覆盖不到则不开仓标记')
})

test('拖动到最左仍拼上更早一页并保持视窗内容（与原有行为一致）', async () => {
  const first = page(Date.parse('2026-11-01T00:00:00Z'), '5m', 600)
  const older = page(Date.parse('2026-11-01T00:00:00Z') - 600 * MIN5, '5m', 600)
  const api = {
    getKlines: async (symbol, interval, limit, end) => ({ bars: end === undefined ? first : older }),
  }
  const subject = ordersView(api)
  subject.chartOrder.value = { id: 1, status: 'OPEN', side: 'LONG', created_at: new Date(first[0].time).toISOString(), entry_price: 100, sl: 0, tp: 0 }
  subject.chartInterval.value = '5m'
  await subject.loadChartBars()
  assert.equal(subject.chartBars.value.length, 600)

  subject.onChartZoom({ start: 0, end: 100 })
  await tick()
  await tick()
  assert.equal(subject.chartBars.value.length, 1200)
  // 原来 0~100% 视窗覆盖"后来的 600 根"，拼上更早数据后整体右移到 50~100%
  assert.ok(Math.abs(subject.chartZoom.value.start - 50) < 0.01)
  assert.ok(Math.abs(subject.chartZoom.value.end - 100) < 0.01)
})
