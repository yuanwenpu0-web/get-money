import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

import { number, price, tsLabel } from '../src/utils/format.js'
import { barsToCover, locateBar, tradeMarker, zoomForIndexes } from '../src/utils/kline.js'

const MIN = 60_000
const BASE = Date.UTC(2026, 10, 1)

function makeBars(start, step, count) {
  return Array.from({ length: count }, (_, index) => ({
    time: start + index * step,
    open: 100,
    high: 101,
    low: 99,
    close: 100,
    volume: 1,
  }))
}

// 与 views 里同样的 vm 载入方式：import 行被剥掉，因此把 kline/format 工具注入上下文。
function view(api) {
  const source = readFileSync(new URL('../src/views/Market.vue', import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1]
    .replace(/^import .*$/gm, '')
  const messages = []
  const context = vm.createContext({
    api,
    ref: (value) => ({ value }),
    reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {},
    onBeforeUnmount: () => {},
    ElMessage: { info: (text) => messages.push(text), error: (text) => messages.push(text), success: (text) => messages.push(text) },
    barsToCover, locateBar, tradeMarker, zoomForIndexes,
    number, price, tsLabel,
  })
  vm.runInContext(
    `${source}\nglobalThis.subject = { loadBars, loadTrades, changeQuery, switchInterval, locateLatestTrade, chartOption, tradeHint, trades, tradesError, showTrades, zoomWindow, bars, query }`,
    context,
  )
  return { ...context.subject, messages }
}

function markerSeries(subject) {
  return subject.chartOption.value.series.find((item) => item.name === '我的成交')
}

const ORDERS = [
  {
    id: 1, symbol: 'ETHUSDT', side: 'LONG', status: 'CLOSED', entry_price: 100,
    created_at: new Date(BASE + 30 * MIN).toISOString(),
    close_price: 105, closed_at: new Date(BASE + 150 * MIN).toISOString(),
  },
  {
    id: 2, symbol: 'ETHUSDT', side: 'SHORT', status: 'OPEN', entry_price: 101.5,
    created_at: new Date(BASE + 200 * MIN).toISOString(), close_price: 0, closed_at: '',
  },
]

function tradingApi(orders = ORDERS) {
  const calls = []
  return {
    calls,
    getOrders: async (params) => {
      calls.push({ kind: 'orders', ...params })
      return { orders }
    },
    getKlines: async (symbol, interval, limit, end) => {
      calls.push({ kind: 'klines', symbol, interval, limit, end })
      const step = interval === '5m' ? 5 * MIN : MIN
      return { bars: makeBars(BASE, step, 300) }
    },
  }
}

test('按 symbol 拉取成交，并在 1m / 5m 都定位到正确的K线', async () => {
  const api = tradingApi()
  const subject = view(api)
  subject.query.symbol = 'ETHUSDT'
  await subject.changeQuery()

  assert.deepEqual(api.calls[0], { kind: 'klines', symbol: 'ETHUSDT', interval: '1m', limit: 600, end: undefined })
  assert.deepEqual(api.calls[1], { kind: 'orders', symbol: 'ETHUSDT', page: 1, page_size: 100 })

  const minute = markerSeries(subject).data
  // 开仓、平仓各一个点，持仓中的空单只有开仓点
  assert.deepEqual([...minute].map((point) => [...point.value]), [[30, 100], [150, 105], [200, 101.5]])
  assert.deepEqual([...minute].map((point) => point.symbolRotate), [0, 180, 180])
  assert.match(minute[0].label.formatter, /买/)
  assert.match(minute[0].label.formatter, /100\.00/)
  assert.match(minute[1].label.formatter, /平多/)
  assert.match(minute[1].label.formatter, /105\.00/)
  assert.match(minute[2].label.formatter, /卖/)
  // 浏览时不强行改视窗
  assert.deepEqual({ ...subject.zoomWindow.value }, { start: 40, end: 100 })

  await subject.switchInterval('5m')
  const five = markerSeries(subject).data
  assert.deepEqual([...five].map((point) => [...point.value]), [[6, 100], [30, 105], [40, 101.5]])
  assert.deepEqual({ ...subject.zoomWindow.value }, { start: 40, end: 100 })
  // 窗口足够覆盖成交时不额外翻页
  assert.equal(api.calls.filter((call) => call.kind === 'klines' && call.end !== undefined).length, 0)

  subject.showTrades.value = false
  assert.equal(markerSeries(subject), undefined)
  subject.showTrades.value = true
  assert.equal(markerSeries(subject).data.length, 3)
})

test('订单接口失败只影响标记，K线照常显示', async () => {
  const subject = view({
    getOrders: async () => { throw new Error('orders down') },
    getKlines: async () => ({ bars: makeBars(BASE, MIN, 300) }),
  })
  await subject.loadBars()
  await subject.loadTrades()

  assert.equal(subject.bars.value.length, 300)
  assert.equal(subject.tradesError.value, 'orders down')
  assert.equal(markerSeries(subject), undefined)
  assert.match(subject.tradeHint.value, /成交标记不可用/)
  assert.match(subject.tradeHint.value, /K线不受影响/)
  // 失败只给小字说明，不弹 ElMessage
  assert.deepEqual(subject.messages, [])

  await subject.locateLatestTrade()
  assert.equal(subject.bars.value.length, 300)
  assert.equal(subject.messages.length, 1)
})

test('定位最近成交会先补更早分页，再用 zoomForIndexes 缩放', async () => {
  const older = {
    id: 9, symbol: 'BTCUSDT', side: 'LONG', status: 'CLOSED', entry_price: 99,
    created_at: new Date(BASE - 60 * MIN).toISOString(),
    close_price: 103, closed_at: new Date(BASE - 30 * MIN).toISOString(),
  }
  const calls = []
  const subject = view({
    getOrders: async (params) => {
      calls.push({ kind: 'orders', ...params })
      return { orders: [older] }
    },
    getKlines: async (symbol, interval, limit, end) => {
      calls.push({ kind: 'klines', symbol, interval, limit, end })
      if (end === undefined) return { bars: makeBars(BASE, MIN, 300) }
      return { bars: makeBars(BASE - 300 * MIN, MIN, 300) }
    },
  })
  await subject.loadBars()
  await subject.loadTrades()
  assert.deepEqual({ ...subject.zoomWindow.value }, { start: 40, end: 100 })

  await subject.locateLatestTrade()

  // 成交在首根之前：补了一页历史，然后才定位
  assert.equal(subject.bars.value.length, 600)
  assert.equal(calls.filter((call) => call.kind === 'klines' && call.end !== undefined).length, 1)
  assert.deepEqual({ ...subject.zoomWindow.value }, zoomForIndexes(600, [240, 270]))
  assert.ok(subject.zoomWindow.value.start < 40)
  assert.deepEqual(subject.messages, [])
})
