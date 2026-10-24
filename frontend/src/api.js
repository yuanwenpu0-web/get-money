import axios from 'axios'

const http = axios.create({ baseURL: '/api', timeout: 180000 })

http.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail = error?.response?.data?.detail || error?.message || '请求失败'
    return Promise.reject(new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)))
  },
)

export default {
  health: () => http.get('/health'),

  getSettings: () => http.get('/settings'),
  saveSettings: (payload) => http.put('/settings', payload),
  adjustCapital: (payload) => http.post('/settings/capital', payload),
  getCredentials: () => http.get('/settings/credentials'),
  saveCredentials: (payload) => http.put('/settings/credentials', payload),
  testCredentials: () => http.post('/settings/credentials/test'),

  getDashboard: () => http.get('/trading/dashboard'),
  getOrders: (params) => http.get('/trading/orders', { params }),
  getOrder: (id) => http.get(`/trading/orders/${id}`),
  closeOrder: (id) => http.post(`/trading/orders/${id}/close`),
  patchOrder: (id, payload) => http.patch(`/trading/orders/${id}`, payload),
  reverseOrder: (id) => http.post(`/trading/orders/${id}/reverse`),
  getEquity: (limit = 1200) => http.get('/trading/equity', { params: { limit } }),
  getLogs: (limit = 80) => http.get('/trading/logs', { params: { limit } }),
  runCycle: () => http.post('/trading/run-once'),

  getKlines: (symbol, interval, limit = 500) => http.get('/market/klines', { params: { symbol, interval, limit } }),
  getMarketSources: () => http.get('/market/sources'),
  syncMarket: (payload) => http.post('/market/sync', payload),
  previewSignal: (payload) => http.post('/market/preview', payload),

  getStrategies: () => http.get('/strategies'),
  updateStrategy: (code, payload) => http.put(`/strategies/${code}`, payload),

  getModels: () => http.get('/models'),
  trainModel: (payload) => http.post('/models/train', payload),
  deleteModel: (id) => http.delete(`/models/${id}`),

  createBacktest: (payload) => http.post('/backtests', payload),
  listBacktests: () => http.get('/backtests'),
  getBacktest: (id) => http.get(`/backtests/${id}`),
}
