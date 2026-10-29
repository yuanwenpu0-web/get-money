import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/dashboard' },
  { path: '/dashboard', component: () => import('./views/Dashboard.vue'), meta: { title: '总览' } },
  { path: '/orders', component: () => import('./views/Orders.vue'), meta: { title: '订单' } },
  { path: '/indicators', component: () => import('./views/Indicators.vue'), meta: { title: '指标' } },
  { path: '/strategies', component: () => import('./views/Strategies.vue'), meta: { title: '策略' } },
  { path: '/optimize', component: () => import('./views/Optimize.vue'), meta: { title: '参数寻优' } },
  { path: '/backtest', component: () => import('./views/Backtest.vue'), meta: { title: '回测' } },
  { path: '/market', component: () => import('./views/Market.vue'), meta: { title: '行情' } },
  { path: '/settings', component: () => import('./views/Settings.vue'), meta: { title: '设置' } },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})

