<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import EChart from '../components/EChart.vue'
import StatCard from '../components/StatCard.vue'
import { money, number, pct, pnlClass, price, tsLabel } from '../utils/format'
import { signalReason } from '../utils/strategy'

const settings = ref({})
const history = ref([])
const activeRun = ref(null)
const result = ref(null)
const running = ref(false)
const loadingHistory = ref(false)
let pollTimer = null

const form = reactive({
  symbol: 'BTCUSDT',
  interval: '1m',
  strategy: 'ml_ensemble',
  algo: 'lightgbm',
  days: 30,
  initial_cash: 10000,
  fee_pct: 0.04,
  slippage_pct: 0.02,
  threshold: null,
  optimize: true,
})

const metrics = computed(() => result.value?.metrics || {})
const equityCurve = computed(() => result.value?.equity_curve || [])
const trades = computed(() => result.value?.trades || [])
const comparison = computed(() => metrics.value.baseline ? [
  { name: '基础信号（相同风控与成本）', ...metrics.value.baseline },
  { name: '优化策略（样本外）', ...metrics.value },
] : [])
const blockedSignals = computed(() => Object.entries(metrics.value.blocked_signals || {}).map(([reason, count]) => ({ reason: signalReason(reason), count })))
function profitFactor(value) {
  return value.profit_factor_infinite ? '∞' : value.profit_factor == null ? '—' : number(value.profit_factor)
}

const equityOption = computed(() => {
  const points = equityCurve.value
  return {
    backgroundColor: 'transparent',
    grid: { left: 58, right: 18, top: 24, bottom: 46 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15,20,28,0.95)',
      borderColor: 'rgba(148,163,184,0.2)',
      textStyle: { color: '#e6ecf5', fontSize: 11 },
      formatter: (params) => `${params[0].axisValue}<br/>权益 ${money(params[0].value)}`,
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
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 8, borderColor: 'transparent', backgroundColor: 'rgba(148,163,184,0.08)', fillerColor: 'rgba(59,130,246,0.18)', handleStyle: { color: '#60a5fa' }, textStyle: { color: '#64748b' } }],
    series: [
      {
        type: 'line',
        data: points.map((item) => item.equity),
        showSymbol: false,
        lineStyle: { width: 2, color: '#60a5fa' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(96,165,250,0.24)' },
              { offset: 1, color: 'rgba(96,165,250,0)' },
            ],
          },
        },
      },
    ],
  }
})

function shortTime(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value).slice(5, 16)
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false })
}

async function loadBase() {
  try {
    const config = await api.getSettings()
    settings.value = config
    form.symbol = config.symbols?.[0] || 'BTCUSDT'
    form.interval = config.interval || '1m'
    form.fee_pct = Number(((config.fee_rate ?? 0.0004) * 100).toFixed(4))
    form.slippage_pct = Number(((config.slippage ?? 0.0002) * 100).toFixed(4))
  } catch (error) {
    ElMessage.error(error.message)
  }
  await loadHistory()
}

async function loadHistory() {
  loadingHistory.value = true
  try {
    const data = await api.listBacktests()
    history.value = data.backtests || []
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loadingHistory.value = false
  }
}

async function poll(runId) {
  try {
    const data = await api.getBacktest(runId)
    activeRun.value = data
    if (data.status === 'done') {
      result.value = data
      running.value = false
      ElMessage.success('回测完成')
      await loadHistory()
      return
    }
    if (data.status === 'failed') {
      running.value = false
      ElMessage.error(data.error || '回测失败')
      await loadHistory()
      return
    }
    pollTimer = setTimeout(() => poll(runId), 1800)
  } catch (error) {
    running.value = false
    ElMessage.error(error.message)
  }
}

async function submit() {
  running.value = true
  result.value = null
  try {
    const payload = {
      symbol: form.symbol,
      interval: form.interval,
      strategy: form.strategy,
      algo: form.algo,
      days: Number(form.days),
      initial_cash: Number(form.initial_cash),
      fee: Number(form.fee_pct) / 100,
      slippage: Number(form.slippage_pct) / 100,
      threshold: form.threshold == null ? null : Number(form.threshold),
      optimize: form.optimize,
    }
    const created = await api.createBacktest(payload)
    activeRun.value = { id: created.id, status: 'running' }
    poll(created.id)
  } catch (error) {
    running.value = false
    ElMessage.error(error.message)
  }
}

async function loadRun(row) {
  try {
    result.value = await api.getBacktest(row.id)
    activeRun.value = result.value
    window.scrollTo({ top: 0, behavior: 'smooth' })
  } catch (error) {
    ElMessage.error(error.message)
  }
}

onMounted(loadBase)
onBeforeUnmount(() => {
  if (pollTimer) clearTimeout(pollTimer)
})
</script>

<template>
  <div>
    <div class="panel" style="margin-bottom: 14px">
      <div class="panel-head">
        <div>
          <div class="panel-title">策略验证与收益对比</div>
          <div class="panel-sub">收盘信号于下一根开盘撮合，计入双边成本、杠杆与风险限额；独立测试集评估</div>
        </div>
        <el-tag size="small" effect="plain" type="info">统一顺序撮合</el-tag>
      </div>

      <div class="form-grid">
        <div>
          <div class="muted" style="margin-bottom: 4px">币种</div>
          <el-select v-model="form.symbol" style="width: 100%">
            <el-option v-for="symbol in settings.symbols || ['BTCUSDT', 'ETHUSDT']" :key="symbol" :label="symbol" :value="symbol" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">K线周期</div>
          <el-select v-model="form.interval" style="width: 100%">
            <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">策略</div>
          <el-select v-model="form.strategy" style="width: 100%">
            <el-option label="ML 趋势共振" value="ml_ensemble" />
            <el-option label="SuperTrend 趋势跟随" value="trend_follow" />
            <el-option label="RSI 均值回归" value="mean_reversion" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">模型</div>
          <el-select v-model="form.algo" :disabled="form.strategy !== 'ml_ensemble'" style="width: 100%">
            <el-option label="LightGBM" value="lightgbm" />
            <el-option label="XGBoost" value="xgboost" />
            <el-option label="纯规则（不训练）" value="rule" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">回测天数</div>
          <el-input-number v-model="form.days" :min="1" :max="180" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">初始资金</div>
          <el-input-number v-model="form.initial_cash" :min="1000" :step="1000" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">手续费（单边 %）</div>
          <el-input-number v-model="form.fee_pct" :min="0" :max="1" :step="0.01" :precision="4" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">滑点（单边 %）</div>
          <el-input-number v-model="form.slippage_pct" :min="0" :max="1" :step="0.01" :precision="4" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">开仓评分阈值（留空使用策略配置）</div>
          <el-input-number v-model="form.threshold" :min="0.5" :max="1" :step="0.01" :precision="2" placeholder="使用策略配置" style="width: 100%" />
        </div>
      </div>

      <div class="row" style="margin-top: 14px">
        <el-switch v-model="form.optimize" active-text="验证集选参" />
        <el-button type="primary" :loading="running" @click="submit">开始回测</el-button>
        <span class="muted" v-if="running">回测在后台运行，通常需要数秒到一分钟</span>
        <span class="muted" v-else>{{ form.optimize ? '前 60% 训练、15% 验证选参、后 25% 测试；边界隔离预测跨度' : '前 75% 训练、后 25% 测试；边界隔离预测跨度' }}</span>
      </div>
    </div>

    <template v-if="result">
      <div class="grid stats-grid backtest-stats">
        <StatCard label="总收益" :value="pct(metrics.total_return_pct)" :tone="pnlClass(metrics.total_return_pct)" :sub="`${money(metrics.initial_cash)} → ${money(metrics.final_equity)}`" />
        <StatCard label="最大回撤" :value="pct(metrics.max_drawdown_pct)" :tone="pnlClass(metrics.max_drawdown_pct)" :sub="'权益峰值回落'" />
        <StatCard label="夏普" :value="number(metrics.sharpe)" sub="按日收益年化" />
        <StatCard label="胜率" :value="`${metrics.win_rate ?? 0}%`" :sub="`${metrics.total_trades || 0} 笔交易`" />
        <StatCard label="盈利因子" :value="profitFactor(metrics)" sub="总盈利 / 总亏损" />
        <StatCard label="手续费" :value="money(metrics.total_fees)" :tone="pnlClass(-Number(metrics.total_fees || 0))" :sub="`滑点 ${(metrics.slippage * 100).toFixed(3)}% · 费率 ${(metrics.fee_rate * 100).toFixed(3)}%`" />
        <StatCard label="买入持有" :value="pct(metrics.buy_hold_return_pct)" :tone="pnlClass(metrics.buy_hold_return_pct)" sub="测试区间基准" />
        <StatCard label="回测引擎" :value="metrics.engine === 'sequential' ? '顺序撮合' : metrics.engine" :sub="`${metrics.algo} · ${metrics.signal_bars} 个信号K线`" />
      </div>

      <div v-if="comparison.length" class="panel section-gap">
        <div class="panel-head">
          <div>
            <div class="panel-title">样本外对比</div>
            <div class="panel-sub">两组使用相同数据、模型、撮合与风险限制；基础信号关闭行情和成本过滤。收益差 {{ number(metrics.return_improvement_pct) }} 个百分点</div>
          </div>
        </div>
        <el-table :data="comparison" class="table-compact">
          <el-table-column prop="name" label="方案" min-width="240" />
          <el-table-column label="净收益"><template #default="{ row }"><span :class="pnlClass(row.total_return_pct)">{{ pct(row.total_return_pct) }}</span></template></el-table-column>
          <el-table-column label="最大回撤"><template #default="{ row }">{{ pct(row.max_drawdown_pct) }}</template></el-table-column>
          <el-table-column prop="total_trades" label="成交数" />
          <el-table-column label="手续费"><template #default="{ row }">{{ money(row.total_fees) }}</template></el-table-column>
          <el-table-column label="盈利因子"><template #default="{ row }">{{ profitFactor(row) }}</template></el-table-column>
        </el-table>
        <div v-for="warning in metrics.warnings || []" :key="warning" class="form-hint">{{ warning }}</div>
        <div class="form-hint">测试区间：{{ metrics.split?.test_start }} → {{ metrics.split?.test_end }}；边界清除 {{ metrics.split?.purge_bars }} 根标签。</div>
        <el-collapse style="margin-top: 12px">
          <el-collapse-item title="参数选择与信号过滤" name="diagnostics">
            <template v-if="metrics.validation?.enabled">
              <p>验证区间：{{ metrics.validation.start }} → {{ metrics.validation.end }}。选择 {{ metrics.validation.selected }}；{{ metrics.validation.status === 'insufficient_trades' ? '成交不足，保留当前参数' : '按净收益与回撤评分选出' }}。结果仅用于本次回测。</p>
              <el-table :data="metrics.validation.candidates || []" size="small">
                <el-table-column prop="name" label="候选" />
                <el-table-column prop="total_return_pct" label="验证净收益 %" />
                <el-table-column prop="max_drawdown_pct" label="回撤 %" />
                <el-table-column prop="total_trades" label="成交数" />
                <el-table-column prop="objective" label="评分" />
                <el-table-column label="样本量"><template #default="{ row }">{{ row.eligible ? '足够' : '不足 10 笔' }}</template></el-table-column>
              </el-table>
            </template>
            <el-table :data="blockedSignals" size="small" style="margin-top: 12px">
              <el-table-column prop="reason" label="过滤原因" />
              <el-table-column prop="count" label="K线数" width="100" />
            </el-table>
            <pre class="mono" style="white-space: pre-wrap">{{ JSON.stringify(metrics.parameters, null, 2) }}</pre>
          </el-collapse-item>
        </el-collapse>
      </div>

      <div class="panel section-gap">
        <div class="panel-head">
          <div>
            <div class="panel-title">权益曲线</div>
            <div class="panel-sub">{{ metrics.start }} → {{ metrics.end }} · {{ metrics.bars }} 根K线 · 阈值 {{ metrics.threshold }}</div>
          </div>
          <el-tag size="small" effect="plain">{{ result.symbol }} · {{ result.interval }} · {{ result.strategy }}</el-tag>
        </div>
        <EChart v-if="equityCurve.length > 1" :option="equityOption" height="330px" />
        <div v-else class="empty-hint">无权益曲线</div>
      </div>

      <div class="panel section-gap">
        <div class="panel-head">
          <div>
            <div class="panel-title">成交明细</div>
            <div class="panel-sub">最多展示 500 笔，含手续费与退出原因</div>
          </div>
        </div>
        <el-table :data="trades" class="table-compact" max-height="420" empty-text="无成交">
          <el-table-column label="开仓" min-width="140"><template #default="{ row }">{{ tsLabel(row.entry_time) }}</template></el-table-column>
          <el-table-column label="平仓" min-width="140"><template #default="{ row }">{{ tsLabel(row.exit_time) }}</template></el-table-column>
          <el-table-column prop="side" label="方向" width="80" />
          <el-table-column label="开 / 平" min-width="130"><template #default="{ row }"><span class="mono">{{ price(row.entry_price) }} / {{ price(row.exit_price) }}</span></template></el-table-column>
          <el-table-column label="数量" width="110"><template #default="{ row }"><span class="mono">{{ number(row.size, 6) }}</span></template></el-table-column>
          <el-table-column label="盈亏" width="110"><template #default="{ row }"><span class="mono" :class="pnlClass(row.pnl)">{{ money(row.pnl) }}</span></template></el-table-column>
          <el-table-column label="收益率" width="90"><template #default="{ row }"><span class="mono" :class="pnlClass(row.return_pct)">{{ pct(row.return_pct) }}</span></template></el-table-column>
          <el-table-column label="手续费" width="90"><template #default="{ row }"><span class="mono">{{ money(row.fees) }}</span></template></el-table-column>
          <el-table-column prop="status" label="退出" min-width="100" />
        </el-table>
      </div>
    </template>

    <div class="panel section-gap">
      <div class="panel-head">
        <div>
          <div class="panel-title">回测历史</div>
          <div class="panel-sub">点击查看详细曲线与成交</div>
        </div>
        <el-button size="small" :loading="loadingHistory" @click="loadHistory">刷新</el-button>
      </div>
      <el-table :data="history" class="table-compact" empty-text="暂无回测记录" @row-click="loadRun">
        <el-table-column prop="id" label="#" width="56" />
        <el-table-column prop="symbol" label="币种" width="96" />
        <el-table-column prop="interval" label="周期" width="70" />
        <el-table-column prop="strategy" label="策略" min-width="120" />
        <el-table-column prop="algo" label="模型" width="90" />
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <el-tag size="small" effect="plain" :type="row.status === 'done' ? 'success' : row.status === 'failed' ? 'danger' : 'warning'">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="收益 / 回撤" min-width="150">
          <template #default="{ row }">
            <span v-if="row.metrics?.total_return_pct !== undefined" class="mono" :class="pnlClass(row.metrics.total_return_pct)">
              {{ pct(row.metrics.total_return_pct) }} / {{ pct(row.metrics.max_drawdown_pct) }}
            </span>
            <span v-else class="muted">{{ row.error ? row.error.slice(0, 40) : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="交易数" width="80"><template #default="{ row }">{{ row.metrics?.total_trades ?? '—' }}</template></el-table-column>
        <el-table-column label="创建时间" min-width="150"><template #default="{ row }">{{ tsLabel(row.created_at) }}</template></el-table-column>
      </el-table>
    </div>
  </div>
</template>

<style scoped>
.backtest-stats { grid-template-columns: repeat(4, minmax(0, 1fr)); }
@media (max-width: 900px) {
  .backtest-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
