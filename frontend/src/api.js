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

  getStrategies: () => http.get('/strategies'),
  updateStrategy: (code, payload) => http.put(`/strategies/${code}`, payload),

  getModels: (params) => http.get('/models', { params }),
  trainModel: (payload) => http.post('/models/train', payload),
  deleteModel: (id) => http.delete(`/models/${id}`),

  createBacktest: (payload) => http.post('/backtests', payload),
  listBacktests: (params) => http.get('/backtests', { params }),
  getBacktest: (id, params = {}) => http.get(`/backtests/${id}`, { params: { include_trades: false, ...params } }),
  getBacktestTrades: (id, params) => http.get(`/backtests/${id}/trades`, { params }),
}
