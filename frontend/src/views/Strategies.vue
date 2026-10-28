<script setup>
import { onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { price, symbolsLabel } from '../utils/format'
import { signalReason } from '../utils/strategy'
import TradingViewCatalog from '../components/TradingViewCatalog.vue'

const strategies = ref([])
const localStrategies = ref([])
const localSaved = ref({})
const localLoading = ref(false)
const localError = ref('')
const localSaving = ref({})
const settings = ref({})
const symbolCount = ref(0)
const loading = ref(false)
const listError = ref('')
const settingsLoading = ref(false)
const settingsReady = ref(false)
const settingsError = ref('')
const symbolsLoading = ref(false)
const symbolsError = ref('')
const formEdited = ref(false)
const scheduleEdited = ref(false)
const previewing = ref(false)
const running = ref(false)
const preview = ref(null)
const runResult = ref(null)
const runVisible = ref(false)

const form = reactive({ symbol: 'BTCUSDT', interval: '1m', strategy: 'ml_ensemble' })
const editVisible = ref(false)
const editCode = ref('')
const editName = ref('')
const editParams = ref('{}')
const editSaving = ref(false)
const scheduleForm = reactive({ enabled: true, seconds: 60 })
const scheduleSaving = ref(false)
const initialForm = { ...form }
let listPromise = null
let localPromise = null
let settingsPromise = null
let symbolsPromise = null
let disposed = false

const intervalOptions = ['1m', '5m', '15m', '1h', '4h', '1d']

function loadStrategies() {
  if (disposed) return Promise.resolve()
  if (listPromise) return listPromise
  loading.value = true
  listError.value = ''
  listPromise = api.getStrategies().then((data) => {
    if (!disposed) strategies.value = data.strategies || []
  }).catch((error) => {
    if (!disposed) listError.value = error.message
  }).finally(() => {
    loading.value = false
    listPromise = null
  })
  return listPromise
}

function loadLocalStrategies() {
  if (disposed) return Promise.resolve()
  if (localPromise) return localPromise
  if (Object.values(localSaving.value).some(Boolean)) return Promise.resolve()
  localLoading.value = true
  localError.value = ''
  localPromise = Promise.resolve().then(() => api.getLocalStrategies()).then((data) => {
    if (!disposed) {
      localStrategies.value = (data.strategies || []).map(copyLocalStrategy)
      localSaved.value = Object.fromEntries(localStrategies.value.map((row) => [row.code, copyLocalStrategy(row)]))
    }
  }).catch((error) => {
    if (!disposed) localError.value = error.message
  }).finally(() => {
    localLoading.value = false
    localPromise = null
  })
  return localPromise
}

function copyLocalStrategy(row) {
  return { ...row, bound_strategies: [...(row.bound_strategies || [])], params: { ...(row.params || {}) } }
}

function localDirty(row) {
  const saved = localSaved.value[row.code]
  if (!saved) return false
  return row.enabled !== saved.enabled ||
    JSON.stringify([...row.bound_strategies].sort()) !== JSON.stringify([...saved.bound_strategies].sort())
}

async function saveLocalStrategy(row) {
  if (localLoading.value || localSaving.value[row.code]) return
  localSaving.value = { ...localSaving.value, [row.code]: true }
  try {
    const saved = await api.updateLocalStrategy(row.code, {
      enabled: row.enabled,
      bound_strategies: row.bound_strategies,
    })
    if (disposed) return
    const index = localStrategies.value.findIndex((item) => item.code === row.code)
    if (index >= 0) localStrategies.value[index] = copyLocalStrategy(saved)
    localSaved.value = { ...localSaved.value, [row.code]: copyLocalStrategy(saved) }
    ElMessage.success(`${row.name} 配置已保存`)
  } catch (error) {
    if (disposed) return
    let verified = false
    try {
      const latest = await api.getLocalStrategies()
      const serverRow = (latest.strategies || []).find((item) => item.code === row.code)
      if (serverRow) {
        localSaved.value = { ...localSaved.value, [row.code]: copyLocalStrategy(serverRow) }
        verified = true
      }
    } catch (_) {
      // Retain the last confirmed state if the server cannot be reached.
    }
    if (disposed) return
    const index = localStrategies.value.findIndex((item) => item.code === row.code)
    const saved = localSaved.value[row.code]
    if (index >= 0 && saved) localStrategies.value[index] = copyLocalStrategy(saved)
    ElMessage.error(verified
      ? `保存请求失败，已重新读取服务器配置：${error.message}`
      : `保存请求失败，无法确认服务器当前状态，已恢复上次确认的配置，请重新加载：${error.message}`)
  } finally {
    localSaving.value = { ...localSaving.value, [row.code]: false }
  }
}

function loadSettings() {
  if (disposed) return Promise.resolve()
  if (settingsPromise) return settingsPromise
  settingsLoading.value = true
  settingsError.value = ''
  settingsPromise = api.getSettings().then((config) => {
    if (disposed) return
    settings.value = config
    if (!settingsReady.value) {
      if (!formEdited.value && Object.keys(initialForm).every((key) => form[key] === initialForm[key])) {
        form.symbol = config.symbols?.[0] || 'BTCUSDT'
        form.interval = config.interval || '1m'
      }
      if (!scheduleEdited.value && scheduleForm.enabled === true && scheduleForm.seconds === 60) {
        scheduleForm.enabled = Boolean(config.schedule_enabled)
        scheduleForm.seconds = Number(config.schedule_seconds || 60)
      }
    }
    settingsReady.value = true
  }).catch((error) => {
    if (!disposed) settingsError.value = error.message
  }).finally(() => {
    settingsLoading.value = false
    settingsPromise = null
  })
  return settingsPromise
}

function loadSymbols() {
  if (disposed) return Promise.resolve()
  if (symbolsPromise) return symbolsPromise
  symbolsLoading.value = true
  symbolsError.value = ''
  symbolsPromise = api.getMarketSymbols().then((data) => {
    if (!disposed) symbolCount.value = data.symbols?.length || 0
  }).catch((error) => {
    if (!disposed) symbolsError.value = error.message
  }).finally(() => {
    symbolsLoading.value = false
    symbolsPromise = null
  })
  return symbolsPromise
}

function load() {
  return Promise.all([loadStrategies(), loadSettings(), loadSymbols(), loadLocalStrategies()])
}

async function refreshStrategies() {
  if (listPromise) await listPromise
  return loadStrategies()
}

async function refreshSettings() {
  if (settingsPromise) await settingsPromise
  return loadSettings()
}

async function toggle(row) {
  try {
    await api.updateStrategy(row.code, { enabled: row.enabled })
    ElMessage.success(`${row.name} 已${row.enabled ? '启用' : '停用'}`)
  } catch (error) {
    row.enabled = !row.enabled
    ElMessage.error(error.message)
  }
}

function openEdit(row) {
  editCode.value = row.code
  editName.value = row.name
  const editable = { ...(row.effective_params || row.params || {}) }
  delete editable.fee_rate
  delete editable.slippage
  editParams.value = JSON.stringify(editable, null, 2)
  editVisible.value = true
}

async function saveEdit() {
  let parsed
  try {
    parsed = JSON.parse(editParams.value)
  } catch (error) {
    ElMessage.error('参数必须是合法 JSON')
    return
  }
  editSaving.value = true
  try {
    await api.updateStrategy(editCode.value, { params: parsed })
    ElMessage.success('策略参数已保存')
    editVisible.value = false
    await refreshStrategies()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    editSaving.value = false
  }
}

async function runPreview(code) {
  if (code) form.strategy = code
  previewing.value = true
  try {
    preview.value = await api.previewSignal({ ...form })
  } catch (error) {
    preview.value = null
    ElMessage.error(error.message)
  } finally {
    previewing.value = false
  }
}

async function runCycle() {
  if (!settingsReady.value || running.value) return
  running.value = true
  try {
    const result = await api.runCycle()
    runResult.value = result
    runVisible.value = true
    load()
    if (result.ok) ElMessage.success('巡检完成')
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    running.value = false
  }
}

async function saveSchedule() {
  if (!settingsReady.value || scheduleSaving.value) return
  scheduleSaving.value = true
  try {
    await api.saveSettings({
      schedule_enabled: scheduleForm.enabled,
      schedule_seconds: Number(scheduleForm.seconds),
    })
    ElMessage.success('定时任务设置已生效')
    refreshSettings()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    scheduleSaving.value = false
  }
}

function directionText(direction) {
  return direction === 'LONG' ? '做多' : direction === 'SHORT' ? '做空' : '观望'
}

onMounted(load)
onBeforeUnmount(() => { disposed = true })
</script>

<template>
  <div>
    <TradingViewCatalog />

    <div class="panel" style="margin-bottom: 14px">
      <div class="panel-head">
        <div>
          <div class="panel-title">自动巡检</div>
          <div class="panel-sub">定时同步本地K线、重算特征、管理持仓并调整止损止盈</div>
        </div>
        <div class="row">
          <el-switch v-model="scheduleForm.enabled" :disabled="!settingsReady" active-text="启用" inactive-text="暂停" @update:model-value="scheduleEdited = true" />
          <el-input-number v-model="scheduleForm.seconds" :disabled="!settingsReady" :min="10" :max="3600" :step="10" size="small" style="width: 120px" @update:model-value="scheduleEdited = true" />
          <span class="muted">秒</span>
          <el-button size="small" :disabled="!settingsReady" :loading="scheduleSaving" @click="saveSchedule">应用</el-button>
          <el-button size="small" type="primary" :disabled="!settingsReady" :loading="running" @click="runCycle">立即巡检</el-button>
        </div>
      </div>
      <div v-if="settingsLoading && !settingsReady" class="muted" role="status">正在加载巡检配置…</div>
      <div v-if="settingsError" class="row" role="alert" style="margin-bottom: 12px">
        <span class="down">巡检配置加载失败：{{ settingsError }}</span>
        <el-button size="small" :loading="settingsLoading" @click="loadSettings">重试</el-button>
      </div>
      <div v-if="settingsReady" class="row">
        <el-tag size="small" effect="plain" :type="settings.scheduler?.running ? 'success' : 'info'">
          {{ settings.scheduler?.running ? '调度运行中' : '调度已暂停' }}
        </el-tag>
        <span class="muted">交易对 {{ symbolsLabel(settings, symbolCount.value) }} · 周期 {{ settings.interval }}</span>
        <span class="muted">下次运行 {{ settings.scheduler?.next_run || '—' }}</span>
        <span v-if="symbolsLoading" class="muted" role="status">正在加载币种数量…</span>
        <el-button v-else-if="symbolsError" size="small" link @click="loadSymbols">重试币种数量</el-button>
      </div>
    </div>

    <div class="panel" style="margin-bottom: 14px">
      <div class="panel-head">
        <div>
          <div class="panel-title">信号预演</div>
          <div class="panel-sub">不落单，仅按当前已收盘K线计算方向、评分、止损止盈与推荐杠杆</div>
        </div>
      </div>
      <div class="row">
        <el-select v-model="form.symbol" size="small" style="width: 130px" @update:model-value="formEdited = true">
          <el-option v-for="symbol in settings.symbols || ['BTCUSDT', 'ETHUSDT']" :key="symbol" :label="symbol" :value="symbol" />
        </el-select>
        <el-select v-model="form.interval" size="small" style="width: 100px" @update:model-value="formEdited = true">
          <el-option v-for="item in intervalOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <el-select v-model="form.strategy" size="small" style="width: 190px" @update:model-value="formEdited = true">
          <el-option v-for="row in strategies" :key="row.code" :label="row.name" :value="row.code" />
        </el-select>
        <el-button size="small" type="primary" :loading="previewing" @click="runPreview()">计算信号</el-button>
      </div>

      <div v-if="preview" class="signal-box" style="margin-top: 14px">
        <div v-if="preview.signal" class="row">
          <div>
            <div class="muted">方向</div>
            <div class="signal-direction" :class="preview.signal.direction === 'LONG' ? 'up' : preview.signal.direction === 'SHORT' ? 'down' : 'flat'">
              {{ directionText(preview.signal.direction) }}
            </div>
          </div>
          <div class="spacer"></div>
          <div class="kv-grid" style="flex: 2">
            <div class="kv"><span class="kv-key">价格</span><span class="kv-value mono">{{ price(preview.signal.price) }}</span></div>
            <div class="kv"><span class="kv-key">评分</span><span class="kv-value mono">{{ preview.signal.score }}</span></div>
            <div class="kv"><span class="kv-key">上涨概率</span><span class="kv-value mono">{{ preview.signal.prob_up }}</span></div>
            <div class="kv"><span class="kv-key">推荐杠杆</span><span class="kv-value mono">×{{ preview.signal.leverage }}</span></div>
            <div class="kv"><span class="kv-key">仓位占比</span><span class="kv-value mono">{{ (preview.signal.position_pct * 100).toFixed(1) }}%</span></div>
            <div class="kv"><span class="kv-key">止损</span><span class="kv-value mono">{{ price(preview.signal.sl) }}</span></div>
            <div class="kv"><span class="kv-key">止盈</span><span class="kv-value mono">{{ price(preview.signal.tp) }}</span></div>
            <div class="kv"><span class="kv-key">ATR 占比</span><span class="kv-value mono">{{ (preview.signal.atr_pct * 100).toFixed(3) }}%</span></div>
            <div class="kv"><span class="kv-key">预计止损风险（含成本）</span><span class="kv-value mono">{{ ((preview.signal.reason?.risk_pct || 0) * 100).toFixed(2) }}%</span></div>
            <div class="kv"><span class="kv-key">净盈亏比</span><span class="kv-value mono">{{ preview.signal.reason?.net_reward_risk?.toFixed(2) ?? '—' }}</span></div>
          </div>
        </div>
        <div v-else class="empty-hint">当前数据不足，无法生成信号（需至少 120 根K线）。</div>

        <el-divider style="margin: 14px 0" />
        <div class="kv-grid">
          <div class="kv"><span class="kv-key">RSI(14)</span><span class="kv-value mono">{{ preview.indicators?.rsi_14?.toFixed(2) ?? '—' }}</span></div>
          <div class="kv"><span class="kv-key">RSI(7)</span><span class="kv-value mono">{{ preview.indicators?.rsi_7?.toFixed(2) ?? '—' }}</span></div>
          <div class="kv"><span class="kv-key">MACD 柱</span><span class="kv-value mono">{{ preview.indicators?.macd_hist ?? '—' }}</span></div>
          <div class="kv"><span class="kv-key">SuperTrend</span><span class="kv-value">{{ preview.indicators?.supertrend_dir === 1 ? '多头' : preview.indicators?.supertrend_dir === -1 ? '空头' : '—' }}</span></div>
          <div class="kv"><span class="kv-key">ATR 占比</span><span class="kv-value mono">{{ preview.indicators?.atr_pct ? `${(preview.indicators.atr_pct * 100).toFixed(3)}%` : '—' }}</span></div>
          <div class="kv"><span class="kv-key">ADX</span><span class="kv-value mono">{{ preview.indicators?.adx_14?.toFixed(1) ?? '—' }}</span></div>
          <div class="kv"><span class="kv-key">EMA 9/21</span><span class="kv-value mono">{{ preview.indicators?.ema_ratio_9_21?.toFixed(4) ?? '—' }}</span></div>
          <div class="kv"><span class="kv-key">量能 Z</span><span class="kv-value mono">{{ preview.indicators?.volume_z?.toFixed(2) ?? '—' }}</span></div>
          <div class="kv"><span class="kv-key">K线时间</span><span class="kv-value mono">{{ preview.indicators?.time ?? '—' }}</span></div>
        </div>
        <div v-if="preview.signal?.reason" class="form-hint mono" style="margin-top: 10px">
          {{ signalReason(preview.signal.reason.block_reason) }}
        </div>
      </div>
    </div>

    <div class="panel" style="margin-bottom: 14px">
      <div class="panel-head">
        <div>
          <div class="panel-title">TradingView 本地策略</div>
          <div class="panel-sub">默认停用。启用并绑定系统策略后，只有双方在同一根已收盘 K 线上同时出现同方向入场信号才允许开仓；多个绑定的已启用本地策略须全部同方向触发。</div>
        </div>
        <el-button size="small" :loading="localLoading" :disabled="Object.values(localSaving).some(Boolean)" @click="loadLocalStrategies">重新加载</el-button>
      </div>
      <div class="form-hint" style="margin-bottom: 12px">本地策略仅作为入场过滤，仓位、止损止盈和平仓仍由系统策略与风控管理。原版策略回测与系统实际出场规则不同。修改后请点击保存配置，重新加载会放弃未保存修改；未绑定任何系统策略时不会影响开仓。</div>
      <div v-if="localError" class="row" role="alert">
        <span class="down">本地策略加载失败：{{ localError }}</span>
        <el-button size="small" @click="loadLocalStrategies">重试</el-button>
      </div>
      <div v-else-if="localLoading && !localStrategies.length" class="muted" role="status">正在加载本地策略…</div>
      <div v-else-if="!localStrategies.length" class="muted">暂无本地策略</div>
      <div v-for="row in localStrategies" :key="row.code" class="signal-box" style="margin-top: 10px">
        <div class="row">
          <strong>{{ row.name }}</strong>
          <span class="muted mono">{{ row.code }}</span>
          <el-tag v-if="localDirty(row)" size="small" type="warning">未保存</el-tag>
          <div class="spacer"></div>
          <span class="muted">已保存：{{ localSaved[row.code]?.enabled ? '启用' : '停用' }}</span>
          <el-switch v-model="row.enabled" :disabled="localLoading || localSaving[row.code]" active-text="启用" inactive-text="停用" />
        </div>
        <div class="muted" style="margin-top: 6px">{{ row.description }}</div>
        <div class="muted mono" style="margin-top: 6px">来源：{{ row.source }}</div>
        <div class="row" style="margin-top: 10px">
          <span class="muted">绑定系统策略</span>
          <el-select v-model="row.bound_strategies" multiple collapse-tags collapse-tags-tooltip placeholder="选择系统策略" style="min-width: 260px" :disabled="localLoading || localSaving[row.code]">
            <el-option v-for="system in strategies" :key="system.code" :label="system.name" :value="system.code" />
          </el-select>
          <el-button size="small" type="primary" :disabled="localLoading" :loading="localSaving[row.code]" @click="saveLocalStrategy(row)">保存配置</el-button>
        </div>
      </div>
    </div>

    <div class="row" style="margin-bottom: 12px">
      <div class="panel-title">策略列表</div>
      <div class="spacer"></div>
      <el-button size="small" :loading="loading" @click="loadStrategies">刷新</el-button>
    </div>
    <div v-if="listError" class="row" role="alert" style="margin-bottom: 12px">
      <span class="down">策略列表加载失败：{{ listError }}</span>
      <el-button size="small" :loading="loading" @click="loadStrategies">重试</el-button>
    </div>
    <div v-if="loading && !strategies.length" class="panel empty-hint" role="status">正在加载策略…</div>
    <div v-else-if="!strategies.length && !listError" class="panel empty-hint">暂无策略</div>
    <div class="three-col">
      <div v-for="row in strategies" :key="row.code" class="panel strategy-card">
        <div class="strategy-head">
          <div>
            <div class="strategy-name">{{ row.name }}</div>
            <div class="muted mono" style="margin-top: 2px">{{ row.code }}</div>
          </div>
          <el-switch v-model="row.enabled" @change="toggle(row)" />
        </div>
        <div class="strategy-desc">{{ row.description }}</div>
        <div class="tag-row">
          <el-tag v-for="(value, key) in row.params" :key="key" size="small" effect="plain" type="info">
            {{ key }}={{ value }}
          </el-tag>
          <el-tag v-if="!Object.keys(row.params || {}).length" size="small" effect="plain" type="info">默认参数</el-tag>
        </div>
        <div class="spacer"></div>
        <div class="row">
          <el-button size="small" type="primary" plain :loading="previewing" @click="runPreview(row.code)">预演信号</el-button>
          <el-button size="small" @click="openEdit(row)">编辑参数</el-button>
        </div>
      </div>
    </div>

    <el-dialog v-model="editVisible" :title="`编辑参数 · ${editName}`" width="560px">
      <div class="muted" style="margin-bottom: 8px">显示当前生效参数。保存的字段会覆盖全局设置；删除某个字段可恢复继承默认值。risk_per_trade_pct 为单笔权益风险比例，cooldown_bars 为平仓后冷却根数，regime_filter / cost_filter 控制行情与成本过滤。</div>
      <el-input v-model="editParams" type="textarea" :rows="9" class="mono" />
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" :loading="editSaving" @click="saveEdit">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="runVisible" title="巡检结果" width="620px">
      <template v-if="runResult">
        <el-alert :type="runResult.ok ? 'success' : 'warning'" :closable="false" :title="runResult.ok ? `完成 · 开仓 ${runResult.opened?.length || 0} 笔` : runResult.message" />
        <div class="timeline-log" style="margin-top: 14px">
          <div v-for="(item, index) in runResult.decisions || []" :key="index" class="log-item">
            <div class="log-text">{{ item }}</div>
          </div>
          <div v-for="(item, index) in runResult.managed || []" :key="`m${index}`" class="log-item">
            <div class="log-text">订单 #{{ item.order_id }}：{{ item.actions?.join(', ') }}</div>
          </div>
          <div v-for="(item, index) in [...(runResult.sync_errors || []), ...(runResult.signal_errors || [])]" :key="`e${index}`" class="log-item error">
            <div class="log-text">{{ item }}</div>
          </div>
        </div>
      </template>
    </el-dialog>
  </div>
</template>
