<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import api from './api'
import { symbolsLabel, tsLabel } from './utils/format'

const route = useRoute()
const scheduler = ref({ running: false, next_run: null })
const settings = ref({})
const symbolCount = ref(0)
const statusLoaded = ref(false)
const statusError = ref('')
const statusLoading = ref(true)
let timer = null
let statusPromise = null
let symbolsPromise = null
let disposed = false

const menu = [
  { path: '/dashboard', label: '总览', hint: '收益 · 持仓 · 状态' },
  { path: '/orders', label: '订单', hint: '持仓与历史成交' },
  { path: '/indicators', label: '指标', hint: 'Pine 指标目录' },
  { path: '/strategies', label: '策略', hint: '信号与评分' },
  { path: '/optimize', label: '寻优', hint: '全币种参数搜索' },
  { path: '/backtest', label: '回测', hint: '策略验证 · 收益对比' },
  { path: '/market', label: '行情', hint: '本地K线与数据源' },
  { path: '/settings', label: '设置', hint: 'API · 资金 · 风控' },
]

const title = computed(() => route.meta.title || '总览')

const executionModeLabel = computed(() => {
  if (!statusLoaded.value) return statusError.value ? '状态未知' : '加载中'
  if (settings.value.execution_mode === 'live') {
    return settings.value.testnet ? 'TESTNET LIVE' : 'LIVE'
  }
  return 'PAPER'
})

const executionModeTagType = computed(() => {
  if (!statusLoaded.value) return 'info'
  if (settings.value.execution_mode === 'live') {
    return settings.value.testnet ? 'warning' : 'danger'
  }
  return 'success'
})

const modeHint = computed(() => {
  if (!statusLoaded.value) return statusError.value ? '运行状态暂不可用' : '正在获取运行状态'
  if (settings.value.execution_mode === 'live') {
    return settings.value.testnet ? '订单提交币安测试网 · 不涉及真实资金' : '真实资金实盘 · 主网真实成交'
  }
  return '订单本地模拟 · 不触达真实资金'
})

const pageSub = computed(() => {
  if (!statusLoaded.value) return '本地K线 · 策略评分 · 交易与风控'
  if (settings.value.execution_mode === 'live') {
    return `本地K线 · 策略评分 · 币安${settings.value.testnet ? '测试网' : '主网'}实盘 · 自动巡检`
  }
  return '本地K线 · 策略评分 · 模拟撮合 · 自动巡检'
})

function loadStatus() {
  if (disposed) return Promise.resolve()
  if (statusPromise) return statusPromise
  statusLoading.value = true
  statusPromise = (async () => {
    try {
      const data = await api.getSettings()
      if (disposed) return
      settings.value = data
      scheduler.value = data.scheduler || { running: false }
      statusLoaded.value = true
      statusError.value = ''
    } catch (error) {
      if (!disposed) statusError.value = error.message
    } finally {
      if (!disposed) statusLoading.value = false
      statusPromise = null
    }
  })()
  return statusPromise
}

function loadSymbolCount() {
  if (disposed) return Promise.resolve()
  if (symbolsPromise) return symbolsPromise
  symbolsPromise = (async () => {
    try {
      const data = await api.getMarketSymbols()
      if (!disposed) symbolCount.value = data.symbols?.length || 0
    } catch (error) {
      /* 币种数量失败时保留已有状态。 */
    } finally {
      symbolsPromise = null
    }
  })()
  return symbolsPromise
}

onMounted(() => {
  loadStatus()
  loadSymbolCount()
  timer = setInterval(() => { loadStatus(); loadSymbolCount() }, 30000)
})
onBeforeUnmount(() => {
  disposed = true
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
          {{ statusLoaded ? (scheduler.running ? '定时监测运行中' : '定时监测已暂停') : '运行状态加载中…' }}
        </div>
        <div class="muted" :class="{ down: executionModeLabel === 'LIVE' }">{{ modeHint }}</div>
        <div v-if="scheduler.next_run" class="muted">下次巡检 {{ tsLabel(scheduler.next_run) }}</div>
        <div v-if="statusLoaded" class="muted">{{ symbolsLabel(settings, symbolCount.value) }} · {{ settings.interval }}</div>
        <div v-if="statusError" class="muted" role="status">
          状态刷新失败
          <el-button size="small" link :loading="statusLoading" @click="loadStatus">重试</el-button>
        </div>
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

