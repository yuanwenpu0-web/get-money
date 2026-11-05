import assert from 'node:assert/strict'
import test from 'node:test'
import { barsToCover, intervalMs, locateBar, tradeMarker, zoomForIndexes } from '../src/utils/kline.js'

const HOUR = 3_600_000
const MIN5 = 300_000

function bars(start, interval, count) {
  const step = intervalMs(interval)
  return Array.from({ length: count }, (_, index) => ({ time: start + index * step, close: 100 }))
}

test('周期换算覆盖分钟/小时/天，未知周期返回 0', () => {
  assert.equal(intervalMs('5m'), MIN5)
  assert.equal(intervalMs('1h'), HOUR)
  assert.equal(intervalMs('1d'), 86_400_000)
  assert.equal(intervalMs('7m'), 0)
})

test('同一笔成交在 5m 与 1h 都能定位到包含它的K线', () => {
  const base = Date.parse('2026-11-01T00:00:00Z')
  const five = bars(base, '5m', 600)
  const hourly = bars(base, '1h', 600)
  const tradeTime = new Date(base + 37 * MIN5 + 60_000).toISOString()
  const a = locateBar(five, tradeTime, '5m')
  const b = locateBar(hourly, tradeTime, '1h')
  assert.equal(a.index, 37)
  assert.equal(a.exact, true)
  assert.equal(b.index, 3)
  assert.equal(b.exact, true)
})

test('成交时间早于首根返回 null，晚于末根但在容忍范围内退到末根', () => {
  const base = Date.parse('2026-11-01T00:00:00Z')
  const hourly = bars(base, '1h', 10)
  assert.equal(locateBar(hourly, new Date(base - HOUR).toISOString(), '1h'), null)
  const late = locateBar(hourly, new Date(base + 10 * HOUR + 60_000).toISOString(), '1h')
  assert.equal(late.index, 9)
  assert.equal(locateBar(hourly, new Date(base + 40 * HOUR).toISOString(), '1h'), null)
})

test('缺口行情退到不晚于成交时间的最近一根，并报告偏移', () => {
  const base = Date.parse('2026-11-01T00:00:00Z')
  const hourly = [{ time: base }, { time: base + 5 * HOUR }]
  const found = locateBar(hourly, new Date(base + 3 * HOUR).toISOString(), '1h')
  assert.equal(found.index, 0)
  assert.equal(found.exact, false)
  assert.equal(found.offsetBars, 3)
})

test('窗口根数按成交跨度计算，5m 长持仓会请求更多根并受上限约束', () => {
  const now = Date.parse('2026-11-06T00:00:00Z')
  const entry = new Date(now - 3 * 86_400_000).toISOString()
  const five = barsToCover('5m', [entry, new Date(now - 3_600_000).toISOString()], now)
  const hour = barsToCover('1h', [entry], now)
  assert.ok(five > 800, `5m needs many bars, got ${five}`)
  assert.ok(hour < five)
  assert.ok(barsToCover('5m', ['2025-01-01T00:00:00Z'], now) <= 1500)
  assert.ok(barsToCover('5m', [], now) >= 240)
})

test('自动定位把开平仓都放进视窗，并保留上下文', () => {
  const centered = zoomForIndexes(1000, [400, 420])
  assert.ok(centered.start < 40)
  assert.ok(centered.end > 42)
  const edge = zoomForIndexes(1000, [995])
  assert.equal(edge.end, 100)
  assert.ok(edge.start < 99.5)
  const empty = zoomForIndexes(1000, [])
  assert.ok(empty.end === 100 && empty.start >= 80)
  assert.deepEqual(zoomForIndexes(0, [1]), { start: 0, end: 100 })
})

test('买点用向上三角，卖点用向下三角', () => {
  assert.equal(tradeMarker({ side: 'LONG' }).symbolRotate, 0)
  assert.equal(tradeMarker({ side: 'LONG', action: 'exit' }).symbolRotate, 180)
  assert.equal(tradeMarker({ side: 'SHORT' }).symbolRotate, 180)
})
