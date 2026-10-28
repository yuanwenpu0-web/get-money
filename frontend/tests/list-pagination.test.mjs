import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const tick = () => new Promise((resolve) => setImmediate(resolve))

function deferred() {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function view(name, api, exported) {
  const script = readFileSync(new URL(`../src/views/${name}.vue`, import.meta.url), 'utf8')
    .match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '')
  const unmounts = []
  const context = vm.createContext({
    api, ref: (value) => ({ value }), reactive: (value) => value,
    computed: (getter) => ({ get value() { return getter() } }),
    onMounted: () => {}, onBeforeUnmount: (fn) => unmounts.push(fn),
    ElMessage: { error: (message) => assert.fail(message), success: () => {} },
    ElMessageBox: { confirm: async () => true },
    setInterval: () => 1, clearInterval: () => {}, number: (number) => String(number),
  })
  vm.runInContext(`${script}\nglobalThis.subject = { ${exported.join(', ')} }`, context)
  return { ...context.subject, unmount: () => unmounts.forEach((fn) => fn()) }
}

function page(items, total, number, pageSize) {
  return { total, page: number, page_size: pageSize, pages: Math.ceil(total / pageSize), ...items }
}

test('Orders requests only its page and uses filtered totals rather than page-only statistics', async () => {
  const calls = []
  const subject = view('Orders', {
    getOrders: async (params) => {
      calls.push({ ...params })
      return { ...page({ orders: [{ id: 1, status: 'CLOSED', realized_pnl: 5 }] }, 241, params.page, params.page_size),
        summary: { open: 120, closed: 121, realized_pnl: 850.5, unrealized_pnl: 42.5 } }
    },
  }, ['load', 'filters', 'pagination', 'summary'])
  subject.filters.status = ''
  await subject.load()
  assert.deepEqual(calls, [{ page: 1, page_size: 20 }])
  assert.equal(subject.pagination.total, 241)
  assert.equal(subject.summary.value.closed, 121)
  assert.equal(subject.summary.value.realized, 850.5)
  assert.equal(subject.summary.value.unrealized, 42.5)
})

test('Orders page changes retain filters and filter/page-size changes return to page one', async () => {
  const calls = []
  const subject = view('Orders', {
    getOrders: async (params) => {
      calls.push({ ...params })
      return page({ orders: [{ id: params.page }] }, 200, params.page, params.page_size)
    },
  }, ['load', 'filters', 'pagination', 'changePage', 'changePageSize', 'changeFilters'])
  await subject.load()
  await subject.changePage(3)
  assert.deepEqual(calls.at(-1), { page: 3, page_size: 20, status: 'OPEN' })
  subject.filters.symbol = 'ETHUSDT'
  await subject.changeFilters()
  assert.deepEqual(calls.at(-1), { page: 1, page_size: 20, status: 'OPEN', symbol: 'ETHUSDT' })
  await subject.changePage(2)
  await subject.changePageSize(50)
  assert.deepEqual(calls.at(-1), { page: 1, page_size: 50, status: 'OPEN', symbol: 'ETHUSDT' })
})

test('Orders does not apply a response from a page that is no longer selected', async () => {
  const requests = []
  const subject = view('Orders', {
    getOrders: (params) => {
      const response = deferred()
      requests.push({ params, ...response })
      return response.promise
    },
  }, ['load', 'orders', 'pagination', 'changePage'])
  const pending = subject.load()
  const next = subject.changePage(2)
  assert.equal(next, pending)
  requests[0].resolve(page({ orders: [{ id: 1 }] }, 100, 1, 20))
  await tick()
  assert.equal(subject.orders.value.length, 0)
  assert.equal(requests[1].params.page, 2)
  requests[1].resolve(page({ orders: [{ id: 2 }] }, 100, 2, 20))
  await pending
  assert.equal(subject.orders.value[0].id, 2)
  assert.equal(subject.pagination.page, 2)
})

test('Orders refreshes a valid previous page when the last page becomes empty', async () => {
  const requests = []
  const subject = view('Orders', {
    getOrders: async (params) => {
      requests.push({ ...params })
      return page({ orders: params.page === 1 ? [{ id: 1 }] : [] }, 20, params.page, params.page_size)
    },
  }, ['load', 'orders', 'pagination'])
  subject.pagination.page = 2
  await subject.load()
  assert.deepEqual(requests.map((params) => params.page), [2, 1])
  assert.equal(subject.pagination.page, 1)
  assert.equal(subject.orders.value[0].id, 1)
})

test('Unloading the order list stops a queued page request and ignores the in-flight response', async () => {
  const response = deferred()
  let requests = 0
  const subject = view('Orders', { getOrders: () => { requests += 1; return response.promise } }, ['load', 'orders', 'changePage'])
  const pending = subject.load()
  subject.changePage(2)
  subject.unmount()
  response.resolve(page({ orders: [{ id: 1 }] }, 60, 1, 20))
  await pending
  assert.equal(requests, 1)
  assert.equal(subject.orders.value.length, 0)
})
