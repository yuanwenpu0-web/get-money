<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { money, tsLabel } from '../utils/format'

const settings = ref({})
const loading = ref(false)
const settingsReady = ref(false)
const loadError = ref('')
let loadPromise = null
let disposed = false
const saving = ref(false)
const credentialInfo = ref({})
const credentialSaving = ref(false)
const testing = ref(false)
const testResult = ref(null)
const capitalLoading = ref(false)

const form = reactive({
  execution_mode: 'paper',
  proxy_url: '',
  symbol_scope: 'all',
  data_source: 'binance',
  symbols: ['BTCUSDT', 'ETHUSDT'],
  interval: '1m',
  testnet: true,
  fee_pct: 0.04,
  slippage_pct: 0.02,
  base_pct: 8,
  max_pct: 25,
  min_score: 0.55,
  heavy_score: 0.75,
  max_leverage: 10,
  sl_atr_mult: 1.5,
  tp_atr_mult: 2.5,
  min_sl_pct_pct: 0.25,
  min_tp_pct_pct: 0.5,
  trailing_atr_mult: 1.2,
  min_hold_bars: 30,
  max_hold_bars: 120,
  trailing_enabled: true,
  risk_pct: 0.5,
  cooldown_bars: 10,
  trail_activation_atr: 1.5,
  train_horizon: 5,
  auto_train_days: 40,
  auto_train_workers: 2,
  schedule_enabled: true,
  schedule_seconds: 60,
})

const credentialForm = reactive({ api_key: '', api_secret: '', testnet: true })
const capitalForm = reactive({ mode: 'add', amount: 1000, note: '' })

function applySettings(data) {
  settings.value = data
  credentialInfo.value = data.credentials || {}
  form.execution_mode = data.execution_mode === 'live' ? 'live' : 'paper'
  form.proxy_url = data.proxy_url || ''
  form.symbol_scope = data.symbol_scope === 'custom' ? 'custom' : 'all'
  form.data_source = data.data_source || 'binance'
  form.symbols = data.symbols || ['BTCUSDT', 'ETHUSDT']
  form.interval = data.interval || '1m'
  form.testnet = Boolean(data.testnet)
  form.fee_pct = Number(((data.fee_rate ?? 0.0004) * 100).toFixed(4))
  form.slippage_pct = Number(((data.slippage ?? 0.0002) * 100).toFixed(4))
  form.base_pct = Number(((data.base_position_pct || 0.08) * 100).toFixed(2))
  form.max_pct = Number(((data.max_position_pct || 0.25) * 100).toFixed(2))
  form.min_score = data.min_score ?? 0.55
  form.heavy_score = data.heavy_score ?? 0.75
  form.max_leverage = data.max_leverage ?? 10
  form.sl_atr_mult = data.sl_atr_mult ?? 1.5
  form.tp_atr_mult = data.tp_atr_mult ?? 2.5
  form.min_sl_pct_pct = Number(((data.min_sl_pct || 0.0025) * 100).toFixed(4))
  form.min_tp_pct_pct = Number(((data.min_tp_pct || 0.005) * 100).toFixed(4))
  form.trailing_atr_mult = data.trailing_atr_mult ?? 1.2
  form.min_hold_bars = data.min_hold_bars ?? 30
  form.max_hold_bars = data.max_hold_bars ?? 120
  form.trailing_enabled = data.trailing_enabled ?? true
  form.risk_pct = Number(((data.risk_per_trade_pct ?? 0.005) * 100).toFixed(2))
  form.cooldown_bars = data.cooldown_bars ?? 10
  form.trail_activation_atr = data.trail_activation_atr ?? 1.5
  form.train_horizon = data.train_horizon ?? 5
  form.auto_train_days = data.auto_train_days ?? 40
  form.auto_train_workers = data.auto_train_workers ?? 2
  form.schedule_enabled = Boolean(data.schedule_enabled)
  form.schedule_seconds = Number(data.schedule_seconds || 60)
  credentialForm.testnet = Boolean(data.testnet)
}

function load() {
  if (disposed) return Promise.resolve()
  if (loadPromise) return loadPromise
  loading.value = true
  loadPromise = (async () => {
    try {
      const data = await api.getSettings()
      if (disposed) return
      applySettings(data)
      settingsReady.value = true
      loadError.value = ''
    } catch (error) {
      if (!disposed) loadError.value = error.message || '设置加载失败'
    }
  })().finally(() => {
    loading.value = false
    loadPromise = null
  })
  return loadPromise
}

async function saveSettings() {
  if (!settingsReady.value || loading.value) return
  if (form.symbol_scope === 'custom' && !form.symbols.length) {
    ElMessage.error('自定义列表至少保留一个交易对')
    return
  }
  saving.value = true
  try {
    const payload = {
      execution_mode: form.execution_mode,
      proxy_url: form.proxy_url.trim(),
      symbol_scope: form.symbol_scope,
      data_source: form.data_source,
      symbols: form.symbols.map((item) => String(item).trim().toUpperCase()),
      interval: form.interval,
      testnet: form.testnet,
      fee_rate: Number(form.fee_pct) / 100,
      slippage: Number(form.slippage_pct) / 100,
      base_position_pct: Number(form.base_pct) / 100,
      max_position_pct: Number(form.max_pct) / 100,
      min_score: Number(form.min_score),
      heavy_score: Number(form.heavy_score),
      max_leverage: Number(form.max_leverage),
      sl_atr_mult: Number(form.sl_atr_mult),
      tp_atr_mult: Number(form.tp_atr_mult),
      min_sl_pct: Number(form.min_sl_pct_pct) / 100,
      min_tp_pct: Number(form.min_tp_pct_pct) / 100,
      trailing_atr_mult: Number(form.trailing_atr_mult),
      min_hold_bars: Number(form.min_hold_bars),
      max_hold_bars: Number(form.max_hold_bars),
      trailing_enabled: form.trailing_enabled,
      risk_per_trade_pct: Number(form.risk_pct) / 100,
      cooldown_bars: Number(form.cooldown_bars),
      trail_activation_atr: Number(form.trail_activation_atr),
      train_horizon: Number(form.train_horizon),
      auto_train_days: Number(form.auto_train_days),
      auto_train_workers: Number(form.auto_train_workers),
      schedule_enabled: form.schedule_enabled,
      schedule_seconds: Number(form.schedule_seconds),
    }
    applySettings(await api.saveSettings(payload))
    ElMessage.success('设置已保存')
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    saving.value = false
  }
}

async function saveCredentials() {
  if (!settingsReady.value || loading.value) return
  if (!credentialForm.api_key.trim() || !credentialForm.api_secret.trim()) {
    ElMessage.error('请填写 API key 与 secret')
    return
  }
  credentialSaving.value = true
  try {
    credentialInfo.value = await api.saveCredentials({
      api_key: credentialForm.api_key.trim(),
      api_secret: credentialForm.api_secret.trim(),
      testnet: credentialForm.testnet,
    })
    credentialForm.api_key = ''
    credentialForm.api_secret = ''
    ElMessage.success('API 密钥已加密保存')
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    credentialSaving.value = false
  }
}

async function testCredentials() {
  if (!settingsReady.value || loading.value) return
  testing.value = true
  testResult.value = null
  try {
    testResult.value = await api.testCredentials()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    testing.value = false
  }
}

async function adjustCapital() {
  if (!settingsReady.value || loading.value) return
  capitalLoading.value = true
  try {
    await api.adjustCapital({ mode: capitalForm.mode, amount: Number(capitalForm.amount), note: capitalForm.note })
    ElMessage.success('资金已调整')
    capitalForm.note = ''
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    capitalLoading.value = false
  }
}

async function confirmExecutionMode(value) {
  if (value !== 'live' || credentialInfo.value.configured) return
  try {
    await ElMessageBox.confirm(
      '尚未配置币安 API Key，实盘下单将无法执行。仍要切换到实盘下单模式吗？',
      '实盘模式提醒',
      { confirmButtonText: '仍要切换', cancelButtonText: '返回模拟', type: 'warning' },
    )
  } catch {
    form.execution_mode = 'paper'
  }
}

const credentialPanelSub = computed(() =>
  form.execution_mode === 'live'
    ? '用于实盘下单与账户校验；密钥加密存储'
    : '仅用于账户校验；当前为本地模拟模式',
)

const executionModeAlertType = computed(() => (form.testnet ? 'warning' : 'error'))
const executionModeAlertText = computed(() =>
  form.testnet
    ? '测试网实盘：订单将提交到币安测试网撮合，不涉及真实资金。建议先在测试网验证策略表现后再切主网。'
    : '真实资金风险：订单将用真实资金在币安主网成交，止损止盈可能因跳空或流动性不足失效。请先在测试网充分验证策略表现。',
)

watch(
  () => credentialForm.testnet,
  (value) => {
    form.testnet = value
  },
)

onMounted(load)
onBeforeUnmount(() => { disposed = true })
</script>

<template>
  <div>
    <div class="row" style="margin-bottom: 14px">
      <div>
        <div class="panel-title">系统设置</div>
        <div class="panel-sub">所有金额单位 USDT；盈亏会自动累加到总资金，也可手动调整</div>
      </div>
      <div class="spacer"></div>
      <span v-if="loading" class="muted" role="status">正在读取已保存设置…</span>
      <el-button type="primary" :disabled="!settingsReady || loading" :loading="saving" @click="saveSettings">保存风控与系统设置</el-button>
    </div>

    <div v-if="loadError" class="panel" role="alert" style="margin-bottom: 14px">
      <span class="down">设置读取失败：{{ loadError }}</span>
      <el-button size="small" link type="primary" :loading="loading" @click="load">重试</el-button>
    </div>
    <el-form :disabled="!settingsReady || loading" @submit.prevent>
    <div class="two-col">
      <div style="display: flex; flex-direction: column; gap: 14px">
        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">交易模式</div>
              <div class="panel-sub">本地模拟仅在系统内记账；实盘下单会真实提交到币安合约</div>
            </div>
            <el-tag size="small" effect="plain" :type="form.execution_mode === 'live' ? (form.testnet ? 'warning' : 'danger') : 'info'">
              {{ settingsReady ? (form.execution_mode === 'live' ? (form.testnet ? '测试网实盘' : '真实资金实盘') : '本地模拟') : '读取中…' }}
            </el-tag>
          </div>
          <el-radio-group v-model="form.execution_mode" @change="confirmExecutionMode">
            <el-radio-button value="paper">本地模拟</el-radio-button>
            <el-radio-button value="live">实盘下单</el-radio-button>
          </el-radio-group>
          <el-alert
            v-if="form.execution_mode === 'live'"
            :type="executionModeAlertType"
            :closable="false"
            show-icon
            style="margin-top: 12px"
            :title="executionModeAlertText"
          />
          <el-alert
            v-if="form.execution_mode === 'live' && !credentialInfo.configured"
            type="warning"
            :closable="false"
            show-icon
            style="margin-top: 8px"
            title="尚未配置币安 API Key，实盘下单将无法执行。请在下方“币安 API”面板完成配置。"
          />
          <div class="form-hint">切换后需点击右上角“保存风控与系统设置”才会生效；巡检产生的新订单按此模式执行，已有持仓不受影响。</div>
          <div style="margin-top: 14px">
            <div class="muted" style="margin-bottom: 4px">网络代理（proxy_url）</div>
            <el-input
              v-model="form.proxy_url"
              placeholder="http://127.0.0.1:7890（支持 http / https / socks5，留空 = 直连）"
              clearable
            />
            <div class="form-hint">用于访问币安接口（行情、下单与账户请求均走该代理）；本地可直连或留空。国内访问币安建议配置本地代理端口。</div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">币安 API</div>
              <div class="panel-sub">{{ credentialPanelSub }}</div>
            </div>
            <el-tag size="small" effect="plain" :type="credentialInfo.configured ? 'success' : 'info'">
              {{ settingsReady ? (credentialInfo.configured ? '已配置' : '未配置') : '读取中…' }}
            </el-tag>
          </div>
          <el-alert type="warning" :closable="false" show-icon style="margin-bottom: 12px"
            title="API Secret 使用本地 Fernet 密钥加密存储，不会通过接口返回明文。" />
          <div class="form-grid" style="grid-template-columns: 1fr">
            <div>
              <div class="muted" style="margin-bottom: 4px">API Key</div>
              <el-input v-model="credentialForm.api_key" :placeholder="credentialInfo.api_key || '粘贴币安 API Key'" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">API Secret</div>
              <el-input v-model="credentialForm.api_secret" type="password" show-password placeholder="粘贴币安 API Secret" />
            </div>
            <div class="row">
              <el-switch v-model="credentialForm.testnet" active-text="使用测试网" inactive-text="主网" />
              <span class="muted">上次更新 {{ credentialInfo.updated_at ? tsLabel(credentialInfo.updated_at) : '—' }}</span>
            </div>
          </div>
          <div class="row" style="margin-top: 12px">
            <el-button size="small" :loading="credentialSaving" @click="saveCredentials">保存密钥</el-button>
            <el-button size="small" :loading="testing" @click="testCredentials">测试连接</el-button>
          </div>
          <div v-if="testResult" class="form-hint" style="margin-top: 10px">
            <span :class="testResult.ok ? 'up' : 'down'">
              {{ testResult.ok ? '连接正常' : `连接失败：${testResult.error}` }}
            </span>
            <span v-if="testResult.ok"> · 签名校验 {{ testResult.signed_ok ? '通过' : '未执行（无密钥）' }}</span>
            <div v-if="testResult.balances?.length" class="mono" style="margin-top: 4px">
              {{ testResult.balances.map((item) => `${item.asset}:${item.free}`).join('  ') }}
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">资金与仓位</div>
              <div class="panel-sub">盈利自动累加；新开仓按权益 × 仓位占比分配保证金</div>
            </div>
          </div>
          <div class="grid stats-grid" style="grid-template-columns: repeat(3, 1fr)">
            <div class="stat-card"><div class="stat-label">当前权益</div><div class="stat-value">{{ money(settings.total_capital) }}</div></div>
            <div class="stat-card"><div class="stat-label">可用现金</div><div class="stat-value">{{ money(settings.cash) }}</div></div>
            <div class="stat-card"><div class="stat-label">收益基准</div><div class="stat-value">{{ money(settings.initial_capital) }}</div></div>
          </div>
          <div class="form-grid" style="margin-top: 14px">
            <div>
              <div class="muted" style="margin-bottom: 4px">调整方式</div>
              <el-select v-model="capitalForm.mode" style="width: 100%">
                <el-option label="追加资金" value="add" />
                <el-option label="提取资金" value="withdraw" />
                <el-option label="设为新的总资金" value="set" />
              </el-select>
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">金额（USDT）</div>
              <el-input-number v-model="capitalForm.amount" :min="1" :step="500" style="width: 100%" />
            </div>
          </div>
          <div class="row" style="margin-top: 12px">
            <el-input v-model="capitalForm.note" size="small" placeholder="备注（可选）" style="max-width: 240px" />
            <el-button size="small" type="primary" :loading="capitalLoading" @click="adjustCapital">执行调整</el-button>
          </div>
          <div class="form-hint">“设为新的总资金”会按当前权益差额调整可用现金，并重置收益基准。</div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 14px">
        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">数据源与交易对</div>
              <div class="panel-sub">通过币安公共接口同步行情</div>
            </div>
          </div>
          <div class="form-grid">
            <div>
              <div class="muted" style="margin-bottom: 4px">行情来源</div>
              <el-radio-group v-model="form.data_source">
                <el-radio-button value="binance">币安接口</el-radio-button>
              </el-radio-group>
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">K线周期</div>
              <el-select v-model="form.interval" style="width: 100%">
                <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
              </el-select>
            </div>
          </div>
          <div style="margin-top: 12px">
            <div class="muted" style="margin-bottom: 4px">币种范围</div>
            <el-radio-group v-model="form.symbol_scope">
              <el-radio-button value="all">全部 U 本位永续</el-radio-button>
              <el-radio-button value="custom">自定义列表</el-radio-button>
            </el-radio-group>
            <div style="margin-top: 8px">
              <div class="muted" style="margin-bottom: 4px">交易对（自定义列表）</div>
              <el-select
                v-model="form.symbols"
                multiple
                filterable
                allow-create
                default-first-option
                :disabled="form.symbol_scope === 'all'"
                style="width: 100%"
              >
                <el-option v-for="item in ['BTCUSDT', 'ETHUSDT']" :key="item" :label="item" :value="item" />
              </el-select>
              <div v-if="form.symbol_scope === 'all'" class="form-hint">
                全部模式下不启用自定义列表：巡检默认覆盖币安所有 U 本位永续合约（TRADING 状态，排除杠杆代币），单轮巡检耗时随币种数量增加。切到“自定义列表”后可编辑。
              </div>
              <div v-else class="form-hint">巡检仅覆盖此处列出的合约；可直接输入任意币种添加。</div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">策略风控</div>
              <div class="panel-sub">按评分分配仓位，并按止损距离与双边成本限制单笔风险</div>
            </div>
          </div>
          <div class="form-grid">
            <div>
              <div class="muted" style="margin-bottom: 4px">开仓评分阈值</div>
              <el-input-number v-model="form.min_score" :min="0.5" :max="1" :step="0.01" :precision="2" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">单笔止损风险（% 权益，含成本）</div>
              <el-input-number v-model="form.risk_pct" :min="0.01" :max="5" :step="0.1" :precision="2" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">平仓后冷却K线数</div>
              <el-input-number v-model="form.cooldown_bars" :min="0" :max="10000" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">移动止损启动盈利（ATR 倍数）</div>
              <el-input-number v-model="form.trail_activation_atr" :min="0.1" :max="20" :step="0.1" :precision="1" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">重仓评分阈值</div>
              <el-input-number v-model="form.heavy_score" :min="0.5" :max="1" :step="0.01" :precision="2" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">基础仓位（% 权益）</div>
              <el-input-number v-model="form.base_pct" :min="1" :max="100" :step="1" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">单仓上限（%）</div>
              <el-input-number v-model="form.max_pct" :min="1" :max="100" :step="1" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">最大杠杆</div>
              <el-input-number v-model="form.max_leverage" :min="1" :max="125" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">最小持仓K线数</div>
              <el-input-number v-model="form.min_hold_bars" :min="0" :max="10000" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">最大持仓K线数</div>
              <el-input-number v-model="form.max_hold_bars" :min="0" :max="100000" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">止损 ATR 倍数</div>
              <el-input-number v-model="form.sl_atr_mult" :min="0.1" :max="10" :step="0.1" :precision="1" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">止盈 ATR 倍数</div>
              <el-input-number v-model="form.tp_atr_mult" :min="0.2" :max="20" :step="0.1" :precision="1" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">移动止损 ATR 倍数</div>
              <el-input-number v-model="form.trailing_atr_mult" :min="0.1" :max="10" :step="0.1" :precision="1" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">最小止损距离（%）</div>
              <el-input-number v-model="form.min_sl_pct_pct" :min="0.01" :max="20" :step="0.05" :precision="2" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">最小止盈距离（%）</div>
              <el-input-number v-model="form.min_tp_pct_pct" :min="0.01" :max="50" :step="0.05" :precision="2" style="width: 100%" />
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
              <div class="muted" style="margin-bottom: 4px">训练预测跨度</div>
              <el-input-number v-model="form.train_horizon" :min="1" :max="60" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">训练数据天数</div>
              <el-input-number v-model="form.auto_train_days" :min="7" :max="365" style="width: 100%" />
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">自动训练并发数</div>
              <el-input-number v-model="form.auto_train_workers" :min="1" :max="8" style="width: 100%" />
              <div class="form-hint">默认 2；实际并发按 CPU 数量限制，内存较小时可设为 1。</div>
            </div>
          </div>
          <div class="row" style="margin-top: 12px">
            <el-switch v-model="form.trailing_enabled" active-text="模拟盘与回测启用移动止损" inactive-text="固定止损" />
          </div>
          <div class="form-hint">评分反映信号强度，不等于胜率。仓位同时受评分、资金和单笔风险限制；跳空时实际损失可能超过预算。策略 JSON 中的同名参数优先于全局设置。</div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">定时巡检</div>
              <div class="panel-sub">持续监测K线形态，管理止损止盈，必要时反手</div>
            </div>
            <el-tag size="small" effect="plain" :type="settings.scheduler?.running ? 'success' : 'info'">
              {{ settingsReady ? (settings.scheduler?.running ? '运行中' : '已暂停') : '读取中…' }}
            </el-tag>
          </div>
          <div class="row">
            <el-switch v-model="form.schedule_enabled" active-text="启用" inactive-text="暂停" />
            <span class="muted">周期</span>
            <el-input-number v-model="form.schedule_seconds" :min="10" :max="3600" :step="10" style="width: 130px" />
            <span class="muted">秒</span>
            <span class="muted">下次 {{ settings.scheduler?.next_run ? tsLabel(settings.scheduler.next_run) : '—' }}</span>
          </div>
          <div class="form-hint">保存后立即生效。周期过短会增加币安/数据库读取频率，1 分钟线建议 30-60 秒。</div>
        </div>
      </div>
    </div>
    </el-form>
  </div>
</template>
