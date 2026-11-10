import axios from 'axios'
import { shareRequest } from './utils/shareRequest'

const http = axios.create({ baseURL: '/api', timeout: 180000 })

http.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail = error?.response?.data?.detail || error?.message || '请求失败'
    return Promise.reject(new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)))
  },
)

const getSettings = shareRequest(() => http.get('/settings', { params: { include_stats: false } }))
const getMarketSymbols = shareRequest(() => http.get('/market/symbols'))

async function updateSettings(request) {
  const data = await request
  getSettings.invalidate()
  getMarketSymbols.invalidate()
  return data
}

export default {
  health: () => http.get('/health'),

  getSettings,
  saveSettings: (payload) => updateSettings(http.put('/settings', payload)),
  adjustCapital: (payload) => updateSettings(http.post('/settings/capital', payload)),
  getCredentials: () => http.get('/settings/credentials'),
  saveCredentials: (payload) => updateSettings(http.put('/settings/credentials', payload)),
  testCredentials: () => http.post('/settings/credentials/test'),

  getDashboard: (params) => http.get('/trading/dashboard', { params: { include_market: false, ...params } }),
  getOrders: (params) => http.get('/trading/orders', { params }),
  getOrderMarkers: (params) => http.get('/trading/orders/markers', { params }),
  getOrderStrategies: () => http.get('/trading/orders/strategies'),
  getOrder: (id) => http.get(`/trading/orders/${id}`),
  // 按当前搜索条件展开所有可平仓订单 ID（跨分页），用于「按筛选全选」
  getOpenOrderIds: (params) => http.get('/trading/orders/open-ids', { params }),
  batchCloseOrders: (orderIds) => http.post('/trading/orders/batch-close', { order_ids: orderIds }),
  closeOrder: (id) => http.post(`/trading/orders/${id}/close`),
  patchOrder: (id, payload) => http.patch(`/trading/orders/${id}`, payload),
  reverseOrder: (id) => http.post(`/trading/orders/${id}/reverse`),
  getEquity: (limit = 1200) => http.get('/trading/equity', { params: { limit } }),
  getLogs: (params) => http.get('/trading/logs', { params }),
  runCycle: () => http.post('/trading/run-once'),
  // 交易所为真相源：环境、账户快照、下单成败回执与收益基准
  getExchangeState: () => http.get('/trading/exchange'),
  getExchangeEvents: (params) => http.get('/trading/exchange-events', { params }),
  getExchangeStatus: () => http.get('/settings/exchange'),
  rebaseExchange: () => http.post('/settings/exchange/rebase'),

  // end 传毫秒时间戳时只取该时间之前的K线，用于向左翻页加载更早历史
  getKlines: (symbol, interval, limit = 500, end = null) =>
    http.get('/market/klines', { params: { symbol, interval, limit, ...(end ? { end } : {}) } }),
  getMarketSources: (params) => http.get('/market/sources', { params }),
  getMarketSymbols,
  syncMarket: (payload) => http.post('/market/sync', payload),
  previewSignal: (payload) => http.post('/market/preview', payload),

  getIndicators: (params) => http.get('/indicators', { params }),
  getIndicatorResearch: () => http.get('/indicators/research/latest'),
  getIndicator: (id) => http.get('/indicators/' + encodeURIComponent(id)),

  getStrategies: () => http.get('/strategies'),
  updateStrategy: (code, payload) => http.put(`/strategies/${code}`, payload),
  // 某个策略下的全部绑定：信号预演的参数层下拉用它，不依赖绑定表格的当前页
  getStrategyBindings: (code, params) => http.get(`/strategies/${code}/bindings`, { params }),

  // 币种池与「策略 × 池」绑定：参数的真正归属地
  getPools: () => http.get('/pools'),
  getResearchPoolMaintenance: () => http.get('/pools/research/maintenance'),
  refreshResearchPools: () => http.post('/pools/research/maintenance'),
  createPool: (payload) => http.post('/pools', payload),
  updatePool: (code, payload) => http.put(`/pools/${code}`, payload),
  deletePool: (code) => http.delete(`/pools/${code}`),
  getPoolMembers: (code) => http.get(`/pools/${code}/members`),

  getBindings: (params) => http.get('/bindings', { params }),
  // 绑定层出场覆盖：可选出场方法与支撑线指标
  getBindingExitOptions: () => http.get('/bindings/exit-options'),
  createBinding: (payload) => http.post('/bindings', payload),
  updateBinding: (id, payload) => http.put(`/bindings/${id}`, payload),
  deleteBinding: (id) => http.delete(`/bindings/${id}`),

  createBacktest: (payload) => http.post('/backtests', payload),
  listBacktests: (params) => http.get('/backtests', { params }),
  getBacktest: (id, params = {}) => http.get(`/backtests/${id}`, { params: { include_trades: false, ...params } }),
  getBacktestTrades: (id, params) => http.get(`/backtests/${id}/trades`, { params }),

  // 参数寻优：一键跑全币种 → 每币最优参数 → 一键上线
  getOptimizationOptions: () => http.get('/optimizations/options'),
  getOptimizationUniverse: (params) => http.get('/optimizations/universe', { params }),
  createOptimization: (payload) => http.post('/optimizations', payload),
  listOptimizations: (params) => http.get('/optimizations', { params }),
  getOptimization: (id) => http.get(`/optimizations/${id}`),
  cancelOptimization: (id) => http.post(`/optimizations/${id}/cancel`),
  deleteOptimization: (id) => http.delete(`/optimizations/${id}`),
  getOptimizationResults: (id, params) => http.get(`/optimizations/${id}/results`, { params }),
  applyOptimization: (id, payload) => http.post(`/optimizations/${id}/apply`, payload),

  // 币种级参数覆盖：寻优结果的落地形态
  getSymbolParams: (params) => http.get('/symbol-params', { params }),
  saveSymbolParams: (payload) => http.put('/symbol-params', payload),
  deleteSymbolParams: (payload) => http.delete('/symbol-params', { data: payload }),
  // 清空覆盖：服务端一次删完（分页后前端只看到当前页，不能逐页拼 symbols 删）
  clearSymbolParams: (params) => http.delete('/symbol-params/all', { params }),
}


