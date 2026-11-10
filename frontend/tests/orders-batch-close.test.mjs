import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const SOURCE = readFileSync(new URL('../src/views/Orders.vue', import.meta.url), 'utf8')
  .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')

const EXPORTS = [
  'orders', 'selectedIds', 'selectedSet', 'selectedCount', 'selectableOpen',
  'pageSelectableIds', 'allPageSelected', 'somePageSelected',
  'setSelectedIds', 'clearSelection', 'toggleRowSelection', 'togglePageSelection',
  'selectAllMatching', 'closeSelected', 'changeFilters', 'load',
  'filters', 'pagination', 'orderSummary', 'summary',
]

function ordersView(api, confirms = []) {
  const messages = []
  const context = vm.createContext({
    PAGE_SIZE: 15,
    api,
    ref: (value) => ({ value }),
    reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {},
    onBeforeUnmount: () => {},
    ElMessage: {
      info: (text) => messages.push(['info', text]),
      success: (text) => messages.push(['success', text]),
      warning: (text) => messages.push(['warning', text]),
      error: (text) => messages.push(['error', text]),
    },
    ElMessageBox: { confirm: async (...args) => { confirms.push(args); return true } },
    EChart: {}, TablePager: {},
    price: (value) => String(value), tsLabel: (value) => String(value),
    money: (value) => String(value), pct: (value) => `${value}%`,
    pnlClass: () => '', sideLabel: (value) => value, statusLabel: (value) => value,
    strategyLabel: (value) => value, closeReasonLabel: (value) => value,
    barsToCover: () => 100, intervalMs: () => 3600000, KLINE_MAX_LIMIT: 1500,
    locateBar: () => null, tradeMarker: () => ({}), zoomForIndexes: () => ({ start: 0, end: 100 }),
    setInterval: () => 1, clearInterval: () => {},
  })
  vm.runInContext(`${SOURCE}\nglobalThis.subject = { ${EXPORTS.join(', ')} }`, context)
  return { ...context.subject, messages, confirms }
}

const emptyList = {
  getOrders: async () => ({ orders: [], total: 0, summary: { open: 0, selectable_open: 0 } }),
}

test('勾选只针对持仓单，表头全选本页，翻页保留、切筛选清空', async () => {
  const subject = ordersView(emptyList)
  subject.orders.value = [
    { id: 1, status: 'OPEN' },
    { id: 2, status: 'OPEN' },
    { id: 3, status: 'CLOSED' },
  ]

  assert.deepEqual([...subject.pageSelectableIds.value], [1, 2])
  assert.equal(subject.allPageSelected.value, false)
  assert.equal(subject.somePageSelected.value, false)

  subject.togglePageSelection(true)
  assert.deepEqual([...subject.selectedIds.value], [1, 2])
  assert.equal(subject.allPageSelected.value, true)

  // 取消单个 → 表头变成半选
  subject.toggleRowSelection({ id: 2 }, false)
  assert.deepEqual([...subject.selectedIds.value], [1])
  assert.equal(subject.somePageSelected.value, true)
  assert.equal(subject.allPageSelected.value, false)

  // 翻页只换数据，不清勾选
  subject.orders.value = [{ id: 9, status: 'OPEN' }]
  assert.deepEqual([...subject.selectedIds.value], [1])

  subject.togglePageSelection(false)
  assert.deepEqual([...subject.selectedIds.value], [1])
  await subject.changeFilters()
  assert.deepEqual([...subject.selectedIds.value], [])
})

test('按筛选全选只传筛选参数，并合并服务端展开的跨页 ID', async () => {
  const calls = []
  const subject = ordersView({
    ...emptyList,
    getOpenOrderIds: async (params) => { calls.push(params); return { order_ids: [1, 2, 3, 4], count: 4 } },
  })
  subject.filters.status = 'OPEN'
  subject.filters.symbol = 'BTCUSDT'
  subject.filters.strategy = 'absorption_reversal'
  subject.setSelectedIds([9])

  await subject.selectAllMatching()

  assert.deepEqual(calls.map((item) => ({ ...item })), [{ status: 'OPEN', symbol: 'BTCUSDT', strategy: 'absorption_reversal' }])
  assert.deepEqual([...subject.selectedIds.value], [9, 1, 2, 3, 4])
  assert.equal(subject.selectedCount.value, 5)
  assert.ok(subject.messages.some(([, text]) => text.includes('含其他分页')))
})

test('筛选下没有持仓时全选给出提示，不改动已有勾选', async () => {
  const subject = ordersView({ ...emptyList, getOpenOrderIds: async () => ({ order_ids: [], count: 0 }) })
  subject.setSelectedIds([7])
  await subject.selectAllMatching()
  assert.deepEqual([...subject.selectedIds.value], [7])
  assert.ok(subject.messages.some(([level, text]) => level === 'info' && text.includes('没有可平仓持仓')))
})

test('批量平仓只提交勾选 ID，成功后清空，失败的保留待重试', async () => {
  const calls = []
  const confirms = []
  const subject = ordersView({
    ...emptyList,
    batchCloseOrders: async (ids) => {
      calls.push(ids)
      return { closed_count: 2, failed_count: 1, skipped_count: 1, failed: [{ id: 4 }], realized_pnl: 12.5 }
    },
  }, confirms)
  subject.orders.value = [{ id: 1, status: 'OPEN' }]
  subject.setSelectedIds([1, 2, 3, 4])

  await subject.closeSelected()

  assert.deepEqual(calls.map((ids) => [...ids]), [[1, 2, 3, 4]])
  assert.equal(confirms.length, 1)
  assert.ok(confirms[0][0].includes('4 笔'))
  assert.ok(confirms[0][0].includes('3 笔不在当前页'))
  assert.deepEqual([...subject.selectedIds.value], [4])
  assert.ok(subject.messages.some(([level, text]) => level === 'warning' && text.includes('1 笔失败')))
})

test('确认框取消或空勾选都不发请求', async () => {
  const calls = []
  const subject = ordersView({ ...emptyList, batchCloseOrders: async (ids) => { calls.push(ids) } })
  await subject.closeSelected()
  assert.deepEqual(calls, [])
  assert.ok(subject.messages.some(([level, text]) => level === 'info' && text.includes('请先勾选')))
})

test('列表刷新后被平掉的勾选自动失效', async () => {
  const subject = ordersView({
    getOrders: async () => ({
      orders: [{ id: 1, status: 'CLOSED' }, { id: 2, status: 'OPEN' }],
      total: 2,
      summary: { open: 1, selectable_open: 1 },
    }),
  })
  subject.setSelectedIds([1, 2])
  await subject.load()
  assert.deepEqual([...subject.selectedIds.value], [2])
})
