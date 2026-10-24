<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import StatCard from '../components/StatCard.vue'
import { closeReasonLabel, money, pct, pnlClass, price, sideLabel, statusLabel, strategyLabel, tsLabel } from '../utils/format'

const dashboard = ref(null)
const equity = ref([])
const loading = ref(false)
const running = ref(false)
let timer = null

const portfolio = computed(() => dashboard.value?.portfolio || {})
const positions = computed(() => dashboard.value?.positions || [])
const settings = computed(() => dashboard.value?.settings || {})
const logs = computed(() => dashboard.value?.logs || [])

const equityOption = computed(() => {
  const points = equity.value
  return {
    backgroundColor: 'transparent',
    grid: { left: 58, right: 18, top: 24, bottom: 46 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15,20,28,0.95)',
      borderColor: 'rgba(148,163,184,0.2)',
      textStyle: { color: '#e6ecf5', fontSize: 11 },
      formatter: (params) => {
        const item = params[0]
        return `${item.axisValue}<br/>权益 ${money(item.value)}`
      },
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: points.map((item) => shortTime(item.time)),
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.2)' } },
      axisLabel: { color: '#64748b', fontSize: 10, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } },
      axisLabel: { color: '#64748b', fontSize: 10, formatter: (value) => `$${Math.round(value)}` },
    },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 8, borderColor: 'transparent', backgroundColor: 'rgba(148,163,184,0.08)', fillerColor: 'rgba(52,211,153,0.15)', handleStyle: { color: '#34d399' }, textStyle: { color: '#64748b' } }],
    series: [
      {
        type: 'line',
        name: '权益',
        data: points.map((item) => item.equity),
        showSymbol: false,
        lineStyle: { width: 2, color: '#34d399' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(52,211,153,0.25)' },
              { offset: 1, color: 'rgba(52,211,153,0)' },
            ],
          },
        },
      },
    ],
  }
})

function shortTime(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false })
}

function logText(message) {
  try {
    const data = JSON.parse(message)
    if (Array.isArray(data.decisions) && data.decisions.length) return data.decisions.join(' · ')
    if (data.message) return data.message
  } catch (error) {
    /* 普通文本日志 */
  }
  return String(message || '').slice(0, 240)
}

async function load() {
  loading.value = true
  try {
    const [dash, points] = await Promise.all([api.getDashboard(), api.getEquity(1500)])
    dashboard.value = dash
    equity.value = points.points || []
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

async function runNow() {
  running.value = true
  try {
    const result = await api.runCycle()
    if (result.ok) {
      ElMessage.success(`巡检完成：开仓 ${result.opened?.length || 0} 笔`)
    } else {
      ElMessage.warning(result.message || '巡检未完成')
    }
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    running.value = false
  }
}

onMounted(() => {
  load()
  timer = setInterval(load, 30000)
})
onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div v-loading="loading">
    <div class="row" style="margin-bottom: 14px">
      <div>
        <div class="panel-title">账户与收益</div>
        <div class="panel-sub">
          {{ settings.symbols?.join(' / ') }} · {{ settings.interval }} · 自动巡检
          {{ settings.schedule_enabled ? `${settings.schedule_seconds}s` : '已关闭' }}
        </div>
      </div>
      <div class="spacer"></div>
      <el-button size="small" :loading="running" type="primary" @click="runNow">立即巡检</el-button>
      <el-button size="small" @click="load">刷新</el-button>
    </div>

    <div class="grid stats-grid">
      <StatCard label="总权益" :value="money(portfolio.equity)" :sub="`收益基准 ${money(portfolio.initial_capital)}`" />
      <StatCard label="可用现金" :value="money(portfolio.cash)" :sub="`占用保证金 ${money(portfolio.margin_used)}`" />
      <StatCard
        label="未实现盈亏"
        :value="money(portfolio.unrealized_pnl)"
        :tone="pnlClass(portfolio.unrealized_pnl)"
        :sub="`${portfolio.open_positions || 0} 个持仓`"
      />
      <StatCard
        label="累计已实现"
        :value="money(portfolio.realized_pnl)"
        :tone="pnlClass(portfolio.realized_pnl)"
        :sub="`${portfolio.closed_trades || 0} 笔已平仓`"
      />
      <StatCard
        label="总收益率"
        :value="pct(portfolio.total_return_pct)"
        :tone="pnlClass(portfolio.total_return_pct)"
        sub="按手动资金基准计算"
      />
      <StatCard label="胜率" :value="`${portfolio.win_rate ?? 0}%`" :sub="`${portfolio.closed_trades || 0} 笔样本`" />
    </div>

    <div class="two-col section-gap">
      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">权益曲线</div>
            <div class="panel-sub">每次巡检自动记录；盈利自动累加到总资金</div>
          </div>
          <el-tag size="small" effect="plain">{{ equity.length }} 个采样点</el-tag>
        </div>
        <EChart v-if="equity.length > 1" :option="equityOption" height="330px" />
        <div v-else class="empty-hint">暂无权益快照。点击“立即巡检”生成第一笔记录。</div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">系统状态</div>
            <div class="panel-sub">调度、数据与最近巡检</div>
          </div>
          <el-tag size="small" :type="dashboard?.scheduler?.running ? 'success' : 'info'" effect="plain">
            {{ dashboard?.scheduler?.running ? 'RUNNING' : 'PAUSED' }}
          </el-tag>
        </div>
        <div class="kv-grid">
          <div class="kv"><span class="kv-key">执行模式</span><span class="kv-value">{{ settings.execution_mode === 'paper' ? '本地模拟' : settings.execution_mode }}</span></div>
          <div class="kv"><span class="kv-key">定时周期</span><span class="kv-value mono">{{ settings.schedule_seconds }}s</span></div>
          <div class="kv"><span class="kv-key">交易对</span><span class="kv-value">{{ settings.symbols?.join(' / ') }}</span></div>
          <div class="kv"><span class="kv-key">最新信号</span><span class="kv-value">{{ dashboard?.recent_signals?.[0] ? tsLabel(dashboard.recent_signals[0].time) : '—' }}</span></div>
        </div>
        <div class="timeline-log" style="margin-top: 16px">
          <div v-for="log in logs" :key="log.id" class="log-item" :class="{ error: log.level === 'error' }">
            <div class="log-time">{{ tsLabel(log.created_at) }} · {{ log.job }}</div>
            <div class="log-text">{{ logText(log.message) }}</div>
          </div>
          <div v-if="!logs.length" class="empty-hint">暂无日志</div>
        </div>
      </div>
    </div>

    <div class="panel section-gap">
      <div class="panel-head">
        <div>
          <div class="panel-title">当前持仓</div>
          <div class="panel-sub">止损止盈由巡检监控，可在订单页手动调整</div>
        </div>
        <router-link to="/orders"><el-button size="small" text type="primary">管理订单</el-button></router-link>
      </div>
      <el-table :data="positions" class="table-compact" empty-text="暂无持仓">
        <el-table-column prop="symbol" label="币种" width="100" />
        <el-table-column label="方向" width="80">
          <template #default="{ row }">
            <el-tag size="small" :type="row.side === 'LONG' ? 'success' : 'danger'" effect="plain">{{ sideLabel(row.side) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="数量" width="110"><template #default="{ row }"><span class="mono">{{ row.qty }}</span></template></el-table-column>
        <el-table-column label="开仓 / 标记" min-width="150">
          <template #default="{ row }"><span class="mono">{{ price(row.entry_price) }} / {{ price(row.mark_price) }}</span></template>
        </el-table-column>
        <el-table-column label="杠杆" width="70"><template #default="{ row }">×{{ row.leverage }}</template></el-table-column>
        <el-table-column label="保证金" width="110"><template #default="{ row }"><span class="mono">{{ money(row.margin) }}</span></template></el-table-column>
        <el-table-column label="未实现" width="120">
          <template #default="{ row }">
            <span class="mono" :class="pnlClass(row.unrealized_pnl)">{{ money(row.unrealized_pnl) }}</span>
            <span class="muted"> {{ pct(row.unrealized_pct) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="止损 / 止盈" min-width="150">
          <template #default="{ row }"><span class="mono">{{ price(row.sl) }} / {{ price(row.tp) }}</span></template>
        </el-table-column>
        <el-table-column label="策略" min-width="130">
          <template #default="{ row }">{{ strategyLabel(row.strategy) }}</template>
        </el-table-column>
      </el-table>
    </div>

    <div class="two-col section-gap">
      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">最近信号</div>
            <div class="panel-sub">评分来自 ML 概率与趋势指标共振</div>
          </div>
        </div>
        <el-table :data="dashboard?.recent_signals || []" class="table-compact" empty-text="暂无信号">
          <el-table-column label="时间" min-width="135">
            <template #default="{ row }">{{ tsLabel(row.time) }}</template>
          </el-table-column>
          <el-table-column prop="symbol" label="币种" width="90" />
          <el-table-column label="方向" width="76">
            <template #default="{ row }">
              <el-tag size="small" :type="row.direction === 'LONG' ? 'success' : row.direction === 'SHORT' ? 'danger' : 'info'" effect="plain">
                {{ row.direction === 'LONG' ? '做多' : row.direction === 'SHORT' ? '做空' : '观望' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="评分" width="80"><template #default="{ row }"><span class="mono">{{ Number(row.score).toFixed(3) }}</span></template></el-table-column>
          <el-table-column label="概率" width="80"><template #default="{ row }"><span class="mono">{{ Number(row.prob_up).toFixed(3) }}</span></template></el-table-column>
          <el-table-column label="杠杆 / 仓位" width="110">
            <template #default="{ row }"><span class="mono">×{{ row.leverage }} · {{ (row.position_pct * 100).toFixed(1) }}%</span></template>
          </el-table-column>
          <el-table-column label="策略" min-width="120"><template #default="{ row }">{{ strategyLabel(row.strategy) }}</template></el-table-column>
        </el-table>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">最近订单</div>
            <div class="panel-sub">全部为本地模拟成交</div>
          </div>
        </div>
        <el-table :data="dashboard?.recent_orders || []" class="table-compact" empty-text="暂无订单">
          <el-table-column prop="id" label="#" width="52" />
          <el-table-column prop="symbol" label="币种" width="90" />
          <el-table-column label="方向" width="70">
            <template #default="{ row }"><span :class="row.side === 'LONG' ? 'up' : 'down'">{{ sideLabel(row.side) }}</span></template>
          </el-table-column>
          <el-table-column label="状态" width="80">
            <template #default="{ row }"><el-tag size="small" effect="plain" :type="row.status === 'OPEN' ? 'warning' : 'info'">{{ statusLabel(row.status) }}</el-tag></template>
          </el-table-column>
          <el-table-column label="已实现" width="100">
            <template #default="{ row }"><span class="mono" :class="pnlClass(row.realized_pnl)">{{ row.status === 'CLOSED' ? money(row.realized_pnl) : '—' }}</span></template>
          </el-table-column>
          <el-table-column label="平仓原因" min-width="100">
            <template #default="{ row }">{{ closeReasonLabel(row.close_reason) }}</template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>
