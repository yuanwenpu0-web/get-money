<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { number, tsLabel } from '../utils/format'

const models = ref([])
const settings = ref({})
const loading = ref(false)
const training = ref(false)
const form = reactive({ symbol: 'BTCUSDT', interval: '1m', algo: 'lightgbm', horizon: 5, days: 40 })

async function load() {
  loading.value = true
  try {
    const [modelList, config] = await Promise.all([api.getModels(), api.getSettings()])
    models.value = modelList.models || []
    settings.value = config
    form.symbol = config.symbols?.[0] || 'BTCUSDT'
    form.interval = config.interval || '1m'
    form.horizon = Number(config.train_horizon || 5)
    form.days = Number(config.auto_train_days || 40)
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

async function train() {
  training.value = true
  try {
    const record = await api.trainModel({ ...form, horizon: Number(form.horizon), days: Number(form.days) })
    ElMessage.success(`模型训练完成：AUC ${number(record.metrics?.auc, 4)}，准确率 ${number(record.metrics?.accuracy, 4)}`)
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    training.value = false
  }
}

async function remove(row) {
  try {
    await ElMessageBox.confirm(`删除模型 #${row.id}（${row.symbol} ${row.algo}）？删除后 ML 策略会回退到规则信号。`, '确认删除', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    await api.deleteModel(row.id)
    ElMessage.success('已删除')
    await load()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message)
  }
}

onMounted(load)
</script>

<template>
  <div v-loading="loading">
    <div class="two-col">
      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">训练新模型</div>
            <div class="panel-sub">使用本地K线生成特征，按时间前 80% 训练、后 20% 评估</div>
          </div>
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
            <div class="muted" style="margin-bottom: 4px">算法</div>
            <el-select v-model="form.algo" style="width: 100%">
              <el-option label="LightGBM" value="lightgbm" />
              <el-option label="XGBoost" value="xgboost" />
            </el-select>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">预测跨度（K线数）</div>
            <el-input-number v-model="form.horizon" :min="1" :max="60" style="width: 100%" />
          </div>
          <div>
            <div class="muted" style="margin-bottom: 4px">训练数据天数</div>
            <el-input-number v-model="form.days" :min="7" :max="365" style="width: 100%" />
          </div>
        </div>
        <div class="row" style="margin-top: 14px">
          <el-button type="primary" :loading="training" @click="train">开始训练</el-button>
          <span class="muted">1 分钟线 40 天约 5 万行，训练耗时通常 2-5 秒</span>
        </div>
        <div class="form-hint" style="margin-top: 10px">
          标签为未来 N 根K线收盘价是否上涨。ML 趋势共振策略会自动使用同币种/周期下最新模型；无模型时使用规则回退。
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">环境检查</div>
            <div class="panel-sub">依赖与数据准备情况</div>
          </div>
        </div>
        <div class="kv-grid">
          <div class="kv"><span class="kv-key">TA-Lib</span><span class="kv-value" :class="settings.features?.talib ? 'up' : 'down'">{{ settings.features?.talib ? '可用' : '缺失' }}</span></div>
          <div class="kv"><span class="kv-key">pandas-ta-classic</span><span class="kv-value" :class="settings.features?.pandas_ta ? 'up' : 'down'">{{ settings.features?.pandas_ta ? '可用' : '缺失' }}</span></div>
          <div class="kv"><span class="kv-key">回测引擎</span><span class="kv-value">顺序撮合</span></div>
          <div class="kv"><span class="kv-key">已有模型</span><span class="kv-value">{{ models.length }}</span></div>
          <div class="kv"><span class="kv-key">本地K线</span><span class="kv-value">{{ settings.data_stats?.length || 0 }} 组</span></div>
          <div class="kv"><span class="kv-key">nofx 数据源</span><span class="kv-value" :class="settings.nofx?.available ? 'up' : 'down'">{{ settings.nofx?.available ? '在线' : '不可用' }}</span></div>
        </div>
      </div>
    </div>

    <div class="panel section-gap">
      <div class="panel-head">
        <div>
          <div class="panel-title">模型列表</div>
          <div class="panel-sub">每次训练生成独立文件，可对比指标后保留最优</div>
        </div>
        <el-button size="small" @click="load">刷新</el-button>
      </div>
      <el-table :data="models" class="table-compact" empty-text="还没有模型，先训练一个">
        <el-table-column prop="id" label="#" width="56" />
        <el-table-column prop="symbol" label="币种" width="96" />
        <el-table-column prop="interval" label="周期" width="70" />
        <el-table-column prop="algo" label="算法" width="100" />
        <el-table-column label="准确率" width="90"><template #default="{ row }"><span class="mono">{{ number(row.metrics?.accuracy, 4) }}</span></template></el-table-column>
        <el-table-column label="AUC" width="80"><template #default="{ row }"><span class="mono">{{ number(row.metrics?.auc, 4) }}</span></template></el-table-column>
        <el-table-column label="F1" width="80"><template #default="{ row }"><span class="mono">{{ number(row.metrics?.f1, 4) }}</span></template></el-table-column>
        <el-table-column label="训练 / 测试" width="130">
          <template #default="{ row }"><span class="mono">{{ row.metrics?.train_rows }} / {{ row.metrics?.test_rows }}</span></template>
        </el-table-column>
        <el-table-column prop="rows" label="总样本" width="90" />
        <el-table-column prop="horizon" label="跨度" width="70" />
        <el-table-column label="创建时间" min-width="150"><template #default="{ row }">{{ tsLabel(row.created_at) }}</template></el-table-column>
        <el-table-column label="操作" width="90" fixed="right">
          <template #default="{ row }"><el-button size="small" link type="danger" @click="remove(row)">删除</el-button></template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>
