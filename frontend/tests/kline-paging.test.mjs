import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const tick = () => new Promise((resolve) => setImmediate(resolve))

function marketView(api) {
  const source = readFileSync(new URL('../src/views/Market.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const messages = []
  const context = vm.createContext({
    api,
    ref: (value) => ({ value }),
    reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {},
    onBeforeUnmount: () => {},
    ElMessage: { info: (text) => messages.push(text), error: (text) => messages.push(text) },
  })
  vm.runInContext(
    `${source}\nglobalThis.subject = { loadBars, loadOlderBars, onChartZoom, bars, zoomWindow, hasMoreBars, olderLoading, query }`,
    context,
  )
  return { ...context.subject, messages }
}

let nextTime = Date.UTC(2026, 0, 1)
function page(count) {
  const rows = []
  for (let index = 0; index < count; index += 1) {
    const time = nextTime
    nextTime += 15 * 60000
    rows.push({ time, open: 100, high: 101, low: 99, close: 100.5, volume: 1 })
  }
  return rows
}

test('K线拖到最左会拼上更早一页，并保持原视窗内容不变', async () => {
  const calls = []
  const first = page(600)
  const older = page(600)
  const api = {
    getKlines: async (symbol, interval, limit, end) => {
      calls.push({ symbol, interval, limit, end })
      return { bars: end === undefined ? first : older }
    },
  }
  const subject = marketView(api)

  await subject.loadBars()
  assert.equal(subject.bars.value.length, 600)
  assert.deepEqual({ ...subject.zoomWindow.value }, { start: 40, end: 100 })
  assert.equal(subject.hasMoreBars.value, true)

  subject.onChartZoom({ start: 0, end: 100 })
  await tick()
  await tick()

  // 第二次请求带上了最早一根K线的时间游标
  assert.equal(calls.length, 2)
  assert.equal(calls[1].end, first[0].time - 1)
  assert.equal(subject.bars.value.length, 1200)
  // 原来 0~100% 的视窗覆盖的是「后来的 600 根」，拼上新数据后要整体右移到 50~100%
  assert.ok(Math.abs(subject.zoomWindow.value.start - 50) < 0.01)
  assert.ok(Math.abs(subject.zoomWindow.value.end - 100) < 0.01)
  assert.equal(subject.olderLoading.value, false)
  assert.equal(subject.hasMoreBars.value, true)
})

test('拖动幅度不够时不触发翻页', async () => {
  const calls = []
  const api = {
    getKlines: async () => {
      calls.push(1)
      return { bars: page(600) }
    },
  }
  const subject = marketView(api)

  await subject.loadBars()
  subject.onChartZoom({ start: 12, end: 60 })
  await tick()
  assert.equal(calls.length, 1)
  assert.equal(subject.bars.value.length, 600)
})

test('本地没有更早数据时给出提示并停止继续请求', async () => {
  const api = {
    getKlines: async (symbol, interval, limit, end) => ({ bars: end === undefined ? page(600) : [] }),
  }
  const subject = marketView(api)

  await subject.loadBars()
  subject.onChartZoom({ start: 0, end: 100 })
  await tick()
  await tick()

  assert.equal(subject.hasMoreBars.value, false)
  assert.equal(subject.bars.value.length, 600)
  assert.deepEqual(subject.messages, ['已经到本地最早的数据了'])

  // 已经到头后不再发请求
  const before = subject.bars.value.length
  subject.onChartZoom({ start: 0, end: 100 })
  await tick()
  assert.equal(subject.bars.value.length, before)
})

test('重新选币/切周期会把视窗和翻页状态复位', async () => {
  const api = { getKlines: async () => ({ bars: page(600) }) }
  const subject = marketView(api)

  await subject.loadBars()
  subject.zoomWindow.value = { start: 0, end: 30 }
  subject.hasMoreBars.value = false

  await subject.loadBars()
  assert.deepEqual({ ...subject.zoomWindow.value }, { start: 40, end: 100 })
  assert.equal(subject.hasMoreBars.value, true)
})
