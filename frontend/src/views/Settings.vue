<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { money, tsLabel } from '../utils/format'

const settings = ref({})
const loading = ref(false)
const saving = ref(false)
const credentialInfo = ref({})
const credentialSaving = ref(false)
const testing = ref(false)
const testResult = ref(null)
const capitalLoading = ref(false)

const form = reactive({
  data_source: 'nofx',
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
  backtest_trailing: false,
  train_horizon: 5,
  auto_train_days: 40,
  schedule_enabled: true,
  schedule_seconds: 60,
})

const credentialForm = reactive({ api_key: '', api_secret: '', testnet: true })
const capitalForm = reactive({ mode: 'add', amount: 1000, note: '' })

function applySettings(data) {
  settings.value = data
  credentialInfo.value = data.credentials || {}
  form.data_source = data.data_source || 'nofx'
  form.symbols = data.symbols || ['BTCUSDT', 'ETHUSDT']
  form.interval = data.interval || '1m'
  form.testnet = Boolean(data.testnet)
  form.fee_pct = Number(((data.fee_rate || 0.0004) * 100).toFixed(4))
  form.slippage_pct = Number(((data.slippage || 0.0002) * 100).toFixed(4))
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
  form.backtest_trailing = Boolean(data.backtest_trailing)
  form.train_horizon = data.train_horizon ?? 5
  form.auto_train_days = data.auto_train_days ?? 40
  form.schedule_enabled = Boolean(data.schedule_enabled)
  form.schedule_seconds = Number(data.schedule_seconds || 60)
  credentialForm.testnet = Boolean(data.testnet)
}

async function load() {
  loading.value = true
  try {
    applySettings(await api.getSettings())
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

async function saveSettings() {
  if (!form.symbols.length) {
    ElMessage.error('至少保留一个交易对')
    return
  }
  saving.value = true
  try {
    const payload = {
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
      backtest_trailing: form.backtest_trailing,
      train_horizon: Number(form.train_horizon),
      auto_train_days: Number(form.auto_train_days),
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

onMounted(load)
</script>

<template>
  <div v-loading="loading">
    <div class="row" style="margin-bottom: 14px">
      <div>
        <div class="panel-title">系统设置</div>
        <div class="panel-sub">所有金额单位 USDT；盈亏会自动累加到总资金，也可手动调整</div>
      </div>
      <div class="spacer"></div>
      <el-button type="primary" :loading="saving" @click="saveSettings">保存风控与系统设置</el-button>
    </div>

    <div class="two-col">
      <div style="display: flex; flex-direction: column; gap: 14px">
        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">币安 API</div>
              <div class="panel-sub">仅用于账户校验；当前阶段所有订单本地模拟</div>
            </div>
            <el-tag size="small" effect="plain" :type="credentialInfo.configured ? 'success' : 'info'">
              {{ credentialInfo.configured ? '已配置' : '未配置' }}
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
              <div class="panel-sub">默认读取 nofx-new 已有K线，导入后存本地 SQLite</div>
            </div>
          </div>
          <div class="form-grid">
            <div>
              <div class="muted" style="margin-bottom: 4px">行情来源</div>
              <el-radio-group v-model="form.data_source">
                <el-radio-button value="nofx">nofx 本地库</el-radio-button>
                <el-radio-button value="binance">币安接口</el-radio-button>
              </el-radio-group>
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">K线周期</div>
              <el-select v-model="form.interval" style="width: 100%">
                <el-option v-for="item in settings.intervals || ['1m', '5m', '15m', '1h', '4h', '1d']" :key="item" :label="item" :value="item" />
              </el-select>
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">交易对</div>
              <el-select v-model="form.symbols" multiple filterable allow-create default-first-option style="width: 100%">
                <el-option v-for="item in ['BTCUSDT', 'ETHUSDT']" :key="item" :label="item" :value="item" />
              </el-select>
            </div>
            <div>
              <div class="muted" style="margin-bottom: 4px">nofx 数据源</div>
              <el-tag size="small" effect="plain" :type="settings.nofx?.available ? 'success' : 'danger'">
                {{ settings.nofx?.available ? `在线 · ${settings.nofx.symbol_count} 币种` : '不可用' }}
              </el-tag>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">策略风控</div>
              <div class="panel-sub">评分阈值决定方向；仓位占比达到重仓阈值时按上限开仓</div>
            </div>
          </div>
          <div class="form-grid">
            <div>
              <div class="muted" style="margin-bottom: 4px">开仓评分阈值</div>
              <el-input-number v-model="form.min_score" :min="0.5" :max="1" :step="0.01" :precision="2" style="width: 100%" />
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
          </div>
          <div class="row" style="margin-top: 12px">
            <el-switch v-model="form.backtest_trailing" active-text="回测启用移动止损" inactive-text="回测固定止损" />
          </div>
          <div class="form-hint">评分越接近 1，说明 ML 方向概率与指标共振越强；达到重仓阈值时仓位直接取单仓上限。杠杆按 ATR 波动率反比推荐并受上限约束。</div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">定时巡检</div>
              <div class="panel-sub">持续监测K线形态，管理止损止盈，必要时反手</div>
            </div>
            <el-tag size="small" effect="plain" :type="settings.scheduler?.running ? 'success' : 'info'">
              {{ settings.scheduler?.running ? '运行中' : '已暂停' }}
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
  </div>
</template>
