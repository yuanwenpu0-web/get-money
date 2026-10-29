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

  getDashboard: () => http.get('/trading/dashboard', { params: { include_market: false } }),
  getOrders: (params) => http.get('/trading/orders', { params }),
  getOrder: (id) => http.get(`/trading/orders/${id}`),
  closeOrder: (id) => http.post(`/trading/orders/${id}/close`),
  patchOrder: (id, payload) => http.patch(`/trading/orders/${id}`, payload),
  reverseOrder: (id) => http.post(`/trading/orders/${id}/reverse`),
  getEquity: (limit = 1200) => http.get('/trading/equity', { params: { limit } }),
  getLogs: (limit = 80) => http.get('/trading/logs', { params: { limit } }),
  runCycle: () => http.post('/trading/run-once'),

  getKlines: (symbol, interval, limit = 500) => http.get('/market/klines', { params: { symbol, interval, limit } }),
  getMarketSources: (params) => http.get('/market/sources', { params }),
  getMarketSymbols,
  syncMarket: (payload) => http.post('/market/sync', payload),
  previewSignal: (payload) => http.post('/market/preview', payload),

  getIndicators: () => http.get('/indicators'),
  getIndicator: (id) => http.get('/indicators/' + encodeURIComponent(id)),

  getStrategies: () => http.get('/strategies'),
  updateStrategy: (code, payload) => http.put(`/strategies/${code}`, payload),

  // 币种池与「策略 × 池」绑定：参数的真正归属地
  getPools: () => http.get('/pools'),
  createPool: (payload) => http.post('/pools', payload),
  updatePool: (code, payload) => http.put(`/pools/${code}`, payload),
  deletePool: (code) => http.delete(`/pools/${code}`),
  getPoolMembers: (code) => http.get(`/pools/${code}/members`),

  getBindings: (params) => http.get('/bindings', { params }),
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
}


