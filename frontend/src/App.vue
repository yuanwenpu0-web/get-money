<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import api from './api'
import { symbolsLabel, tsLabel } from './utils/format'

const route = useRoute()
const scheduler = ref({ running: false, next_run: null })
const settings = ref({})
const symbolCount = ref(0)
let timer = null

const menu = [
  { path: '/dashboard', label: '总览', hint: '收益 · 持仓 · 状态' },
  { path: '/orders', label: '订单', hint: '持仓与历史成交' },
  { path: '/strategies', label: '策略', hint: '信号与评分' },
  { path: '/backtest', label: '回测', hint: '策略验证 · 收益对比' },
  { path: '/models', label: '模型', hint: 'LightGBM / XGBoost' },
  { path: '/market', label: '行情', hint: '本地K线与数据源' },
  { path: '/settings', label: '设置', hint: 'API · 资金 · 风控' },
]

const title = computed(() => route.meta.title || '总览')

const executionModeLabel = computed(() => {
  if (settings.value.execution_mode === 'live') {
    return settings.value.testnet ? 'TESTNET LIVE' : 'LIVE'
  }
  return 'PAPER'
})

const executionModeTagType = computed(() => {
  if (settings.value.execution_mode === 'live') {
    return settings.value.testnet ? 'warning' : 'danger'
  }
  return 'success'
})

const modeHint = computed(() => {
  if (settings.value.execution_mode === 'live') {
    return settings.value.testnet ? '订单提交币安测试网 · 不涉及真实资金' : '真实资金实盘 · 主网真实成交'
  }
  return '订单本地模拟 · 不触达真实资金'
})

const pageSub = computed(() => {
  if (settings.value.execution_mode === 'live') {
    return `本地K线 · 策略评分 · 币安${settings.value.testnet ? '测试网' : '主网'}实盘 · 自动巡检`
  }
  return '本地K线 · 策略评分 · 模拟撮合 · 自动巡检'
})

async function loadStatus() {
  try {
    const data = await api.getSettings()
    settings.value = data
    scheduler.value = data.scheduler || { running: false }
    loadSymbolCount()
  } catch (error) {
    /* 顶部状态静默失败 */
  }
}

async function loadSymbolCount() {
  try {
    const data = await api.getMarketSymbols()
    symbolCount.value = data.symbols?.length || 0
  } catch (error) {
    /* 币种数量接口不可用时展示“全部永续”不带数量 */
  }
}

onMounted(() => {
  loadStatus()
  timer = setInterval(loadStatus, 30000)
})
onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">Q</div>
        <div>
          <div class="brand-name">Quant Desk</div>
          <div class="brand-sub">CRYPTO SYSTEMATIC</div>
        </div>
      </div>
      <nav class="nav">
        <router-link
          v-for="item in menu"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          active-class="active"
        >
          <span class="nav-label">{{ item.label }}</span>
          <span class="nav-hint">{{ item.hint }}</span>
        </router-link>
      </nav>
      <div class="sidebar-footer">
        <div class="mode-row">
          <span class="status-dot" :class="{ live: scheduler.running }"></span>
          {{ scheduler.running ? '定时监测运行中' : '定时监测已暂停' }}
        </div>
        <div class="muted" :class="{ down: executionModeLabel === 'LIVE' }">{{ modeHint }}</div>
        <div v-if="scheduler.next_run" class="muted">下次巡检 {{ tsLabel(scheduler.next_run) }}</div>
        <div class="muted">{{ symbolsLabel(settings, symbolCount.value) }} · {{ settings.interval || '1m' }}</div>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <div class="page-title">{{ title }}</div>
          <div class="page-sub">{{ pageSub }}</div>
        </div>
        <div class="topbar-right">
          <el-tag size="small" effect="plain" :type="executionModeTagType">{{ executionModeLabel }}</el-tag>
          <span class="muted">数据源 币安接口</span>
          <span class="muted">{{ tsLabel(Date.now()) }}</span>
        </div>
      </header>
      <section class="content">
        <router-view />
      </section>
    </main>
  </div>
</template>
