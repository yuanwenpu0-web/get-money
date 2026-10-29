<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { price, symbolsLabel } from '../utils/format'
import { signalReason } from '../utils/strategy'
import BindingMatrix from '../components/BindingMatrix.vue'

const strategies = ref([])
const bindings = ref([])
const previewBinding = ref('')
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

const form = reactive({ symbol: 'BTCUSDT', interval: '1m', strategy: 'absorption_reversal', pool: '' })
const editVisible = ref(false)
const editCode = ref('')
const editName = ref('')
const editParams = ref('{}')
const editSaving = ref(false)
// 三级止盈比例均以初始仓位为基数；界面用百分数，API 用 0–1 小数。
const supertrendExitDefaults = {
  sl_atr_mult: 1.5, tp1_r: 1, tp2_r: 2, target_r: 3,
  tp1_fraction: 100 / 6, tp2_fraction: 100 / 3, tp3_fraction: 100 / 2,
}
const supertrendExitForm = reactive({ ...supertrendExitDefaults })
const supertrendExitKeys = Object.keys(supertrendExitForm)
const scheduleForm = reactive({ enabled: true, seconds: 60 })
const scheduleSaving = ref(false)
const initialForm = { ...form }
let listPromise = null
let settingsPromise = null
let symbolsPromise = null
let bindingsPromise = null
let disposed = false

const intervalOptions = ['1m', '5m', '15m', '1h', '4h', '1d']

// 预演可以选一条「策略 × 池」绑定作为参数层；选中后周期跟着绑定走，
// 因为后端取参数时要求 (策略, 池, 周期) 三元组精确命中。
const previewBindingOptions = computed(() => bindings.value.filter((row) => row.strategy === form.strategy))

function onPreviewStrategyChange() {
  previewBinding.value = ''
  form.pool = ''
}

function onPreviewBindingChange(bindingId) {
  const row = bindings.value.find((item) => String(item.id) === String(bindingId))
  if (!row) {
    form.pool = ''
    return
  }
  form.pool = row.pool
  form.interval = row.interval
}

function onPreviewIntervalChange() {
  formEdited.value = true
  if (!form.pool) return
  const hit = bindings.value.some((row) => (
    row.strategy === form.strategy && row.pool === form.pool && row.interval === form.interval
  ))
  if (hit) return
  form.pool = ''
  previewBinding.value = ''
}

function loadBindings() {
  if (disposed) return Promise.resolve()
  if (bindingsPromise) return bindingsPromise
  bindingsPromise = api.getBindings().then((data) => {
    if (!disposed) bindings.value = data.bindings || []
  }).catch(() => {
    if (!disposed) bindings.value = []
  }).finally(() => {
    bindingsPromise = null
  })
  return bindingsPromise
}

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
  return Promise.all([loadStrategies(), loadSettings(), loadSymbols(), loadBindings()])
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
  if (row.code === 'supertrend_current') {
    Object.assign(supertrendExitForm, supertrendExitDefaults)
    for (const key of supertrendExitKeys) {
      const value = editable[key]
      if (value != null) supertrendExitForm[key] = Number(value) * (key.endsWith('_fraction') ? 100 : 1)
      delete editable[key]
    }
  }
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
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
    ElMessage.error('参数必须是 JSON 对象')
    return
  }
  if (editCode.value === 'supertrend_current') {
    const f = supertrendExitForm
    const numbers = supertrendExitKeys.map((key) => Number(f[key]))
    if (supertrendExitKeys.some((key) => f[key] == null) || numbers.some((value) => !Number.isFinite(value)) || f.sl_atr_mult <= 0 ||
        f.tp1_r <= 0 || f.tp1_r >= f.tp2_r || f.tp2_r >= f.target_r ||
        [f.tp1_fraction, f.tp2_fraction, f.tp3_fraction].some((value) => value < 0 || value > 100) ||
        Math.abs(f.tp1_fraction + f.tp2_fraction + f.tp3_fraction - 100) > 0.000001) {
      ElMessage.error('止盈目标须依次递增，各档平仓比例须在 0%–100% 且合计 100%')
      return
    }
    for (const key of supertrendExitKeys) {
      parsed[key] = Number(f[key]) / (key.endsWith('_fraction') ? 100 : 1)
    }
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
    preview.value = await api.previewSignal({ ...form, pool: form.pool || null })
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

    <BindingMatrix :strategies="strategies" @change="loadBindings" />

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
        <el-select v-model="form.interval" size="small" style="width: 100px" @update:model-value="onPreviewIntervalChange">
          <el-option v-for="item in intervalOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <el-select v-model="form.strategy" size="small" style="width: 190px" @update:model-value="onPreviewStrategyChange">
          <el-option v-for="row in strategies" :key="row.code" :label="row.name" :value="row.code" />
        </el-select>
        <el-select
          v-model="previewBinding"
          size="small"
          clearable
          placeholder="参数层：策略默认"
          style="width: 200px"
          @change="onPreviewBindingChange"
        >
          <el-option
            v-for="row in previewBindingOptions"
            :key="row.id"
            :label="`${row.pool_name || row.pool} × ${row.interval}${row.enabled ? '' : '（巡检已停用）'}`"
            :value="row.id"
          />
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
        <div v-if="row.code === 'simple_zigzag_double'" class="form-hint">
          模拟盘已支持：挂单触价成交（与回测同时序），反向通道为止损，反手同周期开新仓；实盘暂不支持。
        </div>
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
      <div v-if="editCode === 'simple_zigzag_double'" class="form-hint" style="margin-bottom: 12px">
        <div>策略参数使用下方 JSON 修改；开始与结束时间为 UTC 的 Unix 毫秒时间戳。</div>
        <div><span class="mono">long_length / short_length</span>：多、空方向的三重 EMA 长度。</div>
        <div><span class="mono">long_detection / short_detection</span>：确定多、空转折点时使用的最高／最低价回看 K 线数。</div>
        <div><span class="mono">use_long / use_short</span>：是否启用多、空方向（true 或 false）。</div>
        <div><span class="mono">fixed_lot</span> 为 false 时，<span class="mono">lot_long / lot_short</span> 是权益百分比；为 true 时，它们是固定下单数量。</div>
        <div><span class="mono">timestamp_start / timestamp_end</span>：允许创建入场订单的时间范围；结束后按脚本关闭仓位。</div>
        <div><span class="mono">strict_position</span>：按脚本的虚拟持仓状态过滤入场请求，默认关闭。</div>
        <div>模拟盘执行口径：T-1 收盘提交通道挂单、T 根内触价成交，反向通道为止损，反手同周期开新仓；仓位经系统风控（risk_per_trade_pct）截断，杠杆 1。</div>
      </div>
      <div v-if="editCode === 'supertrend_current'" style="margin-bottom: 16px">
        <div class="muted" style="margin-bottom: 10px">1R 为信号 K 线收盘价到止损价的距离，止损距离按信号 K 线 ATR 计算。各档平仓比例按初始仓位计算，总和须为 100%；回测实际成交价可能因下一根开盘和滑点而不同。</div>
        <el-alert title="三级分批止盈目前用于回测，模拟盘及实盘暂不支持。" type="warning" :closable="false" style="margin-bottom: 10px" />
        <div class="form-grid">
          <div>
            <div class="muted" style="margin-bottom: 4px">止损距离（ATR 倍数）</div>
            <el-input-number v-model="supertrendExitForm.sl_atr_mult" :min="0.1" :max="10" :step="0.1" style="width: 100%" />
          </div>
        </div>
        <div v-for="(keys, index) in [['tp1_r', 'tp1_fraction'], ['tp2_r', 'tp2_fraction'], ['target_r', 'tp3_fraction']]" :key="index" class="form-grid" style="margin-top: 10px">
          <div>
            <div class="muted" style="margin-bottom: 4px">TP{{ index + 1 }} 目标（R）</div>
            <el-input-number v-model="supertrendExitForm[keys[0]]" :min="0.1" :max="50" :step="0.1" style="width: 100%" />
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">TP{{ index + 1 }} 平仓比例（%）</div>
            <el-input-number v-model="supertrendExitForm[keys[1]]" :min="0" :max="100" :step="1" style="width: 100%" />
          </div>
        </div>
      </div>
      <div class="muted" style="margin-bottom: 8px">{{ editCode === 'supertrend_current' ? '其他参数（JSON）。' : '显示当前生效参数。' }}保存的字段会覆盖全局设置；删除某个字段可恢复继承默认值。risk_per_trade_pct 为单笔权益风险比例，cooldown_bars 为平仓后冷却根数，regime_filter / cost_filter 控制行情与成本过滤。</div>
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
