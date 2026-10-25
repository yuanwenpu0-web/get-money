<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { price } from '../utils/format'
import { signalReason } from '../utils/strategy'

const strategies = ref([])
const settings = ref({})
const loading = ref(false)
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

const intervalOptions = ['1m', '5m', '15m', '1h', '4h', '1d']

async function load() {
  loading.value = true
  try {
    const [list, config] = await Promise.all([api.getStrategies(), api.getSettings()])
    strategies.value = list.strategies || []
    settings.value = config
    form.symbol = config.symbols?.[0] || 'BTCUSDT'
    form.interval = config.interval || '1m'
    scheduleForm.enabled = Boolean(config.schedule_enabled)
    scheduleForm.seconds = Number(config.schedule_seconds || 60)
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
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
    await load()
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
  running.value = true
  try {
    const result = await api.runCycle()
    runResult.value = result
    runVisible.value = true
    await load()
    if (result.ok) ElMessage.success('巡检完成')
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    running.value = false
  }
}

async function saveSchedule() {
  scheduleSaving.value = true
  try {
    await api.saveSettings({
      schedule_enabled: scheduleForm.enabled,
      schedule_seconds: Number(scheduleForm.seconds),
    })
    ElMessage.success('定时任务设置已生效')
    await load()
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
</script>

<template>
  <div v-loading="loading">
    <div class="panel" style="margin-bottom: 14px">
      <div class="panel-head">
        <div>
          <div class="panel-title">自动巡检</div>
          <div class="panel-sub">定时同步本地K线、重算特征、管理持仓并调整止损止盈</div>
        </div>
        <div class="row">
          <el-switch v-model="scheduleForm.enabled" active-text="启用" inactive-text="暂停" />
          <el-input-number v-model="scheduleForm.seconds" :min="10" :max="3600" :step="10" size="small" style="width: 120px" />
          <span class="muted">秒</span>
          <el-button size="small" :loading="scheduleSaving" @click="saveSchedule">应用</el-button>
          <el-button size="small" type="primary" :loading="running" @click="runCycle">立即巡检</el-button>
        </div>
      </div>
      <div class="row">
        <el-tag size="small" effect="plain" :type="settings.scheduler?.running ? 'success' : 'info'">
          {{ settings.scheduler?.running ? '调度运行中' : '调度已暂停' }}
        </el-tag>
        <span class="muted">交易对 {{ (settings.symbols || []).join(' / ') }} · 周期 {{ settings.interval }}</span>
        <span class="muted">下次运行 {{ settings.scheduler?.next_run || '—' }}</span>
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
        <el-select v-model="form.symbol" size="small" style="width: 130px">
          <el-option v-for="symbol in settings.symbols || ['BTCUSDT', 'ETHUSDT']" :key="symbol" :label="symbol" :value="symbol" />
        </el-select>
        <el-select v-model="form.interval" size="small" style="width: 100px">
          <el-option v-for="item in intervalOptions" :key="item" :label="item" :value="item" />
        </el-select>
        <el-select v-model="form.strategy" size="small" style="width: 190px">
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
