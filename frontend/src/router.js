import { createRouter, createWebHashHistory } from 'vue-router'

import Dashboard from './views/Dashboard.vue'
import Orders from './views/Orders.vue'
import Strategies from './views/Strategies.vue'
import Backtest from './views/Backtest.vue'
import Models from './views/Models.vue'
import Market from './views/Market.vue'
import Settings from './views/Settings.vue'

const routes = [
  { path: '/', redirect: '/dashboard' },
  { path: '/dashboard', component: Dashboard, meta: { title: '总览' } },
  { path: '/orders', component: Orders, meta: { title: '订单' } },
  { path: '/strategies', component: Strategies, meta: { title: '策略' } },
  { path: '/backtest', component: Backtest, meta: { title: '回测' } },
  { path: '/models', component: Models, meta: { title: '模型' } },
  { path: '/market', component: Market, meta: { title: '行情' } },
  { path: '/settings', component: Settings, meta: { title: '设置' } },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
