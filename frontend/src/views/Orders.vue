<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { closeReasonLabel, money, pct, pnlClass, price, sideLabel, statusLabel, strategyLabel, tsLabel } from '../utils/format'

const orders = ref([])
const settings = ref({})
const loading = ref(false)
const filters = reactive({ status: 'OPEN', symbol: '' })
let timer = null
let loadPromise = null
let reloadRequested = false
let disposed = false

const editVisible = ref(false)
const editForm = reactive({ id: null, side: '', mark: 0, sl: 0, tp: 0 })
const editSaving = ref(false)

const summary = computed(() => {
  const closed = orders.value.filter((row) => row.status === 'CLOSED')
  return {
    open: orders.value.filter((row) => row.status === 'OPEN').length,
    closed: closed.length,
    realized: closed.reduce((total, row) => total + Number(row.realized_pnl || 0), 0),
    unrealized: orders.value.filter((row) => row.status === 'OPEN').reduce((total, row) => total + Number(row.unrealized_pnl || 0), 0),
  }
})

const symbols = computed(() => {
  const list = new Set(settings.value.symbols || [])
  orders.value.forEach((row) => list.add(row.symbol))
  return [...list]
})

function load() {
  if (disposed) return Promise.resolve()
  reloadRequested = true
  if (loadPromise) return loadPromise
  loading.value = true
  loadPromise = (async () => {
    do {
      reloadRequested = false
      try {
        const params = { limit: 300 }
        if (filters.status) params.status = filters.status
        if (filters.symbol) params.symbol = filters.symbol
        const data = await api.getOrders(params)
        orders.value = data.orders || []
      } catch (error) {
        ElMessage.error(error.message)
      }
    } while (reloadRequested && !disposed)
  })().finally(() => {
    loadPromise = null
    loading.value = false
  })
  return loadPromise
}

async function loadSettings() {
  try {
    settings.value = await api.getSettings()
  } catch (error) {
    /* 忽略 */
  }
}

async function closeOrder(row) {
  try {
    await ElMessageBox.confirm(
      `按当前标记价 ${price(row.mark_price)} 平仓 ${row.symbol} ${sideLabel(row.side)}？`,
      '确认平仓',
      { type: 'warning', confirmButtonText: '平仓', cancelButtonText: '取消' },
    )
    await api.closeOrder(row.id)
    ElMessage.success('已平仓')
    await load()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message)
  }
}

async function reverseOrder(row) {
  try {
    await ElMessageBox.confirm(
      `先平掉当前 ${sideLabel(row.side)}，再按相反方向开同等保证金仓位？此操作按市价模拟成交。`,
      '确认反手',
      { type: 'warning', confirmButtonText: '反手', cancelButtonText: '取消' },
    )
    await api.reverseOrder(row.id)
    ElMessage.success('已反手')
    await load()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message)
  }
}

function openEdit(row) {
  editForm.id = row.id
  editForm.side = row.side
  editForm.mark = row.mark_price
  editForm.sl = row.sl
  editForm.tp = row.tp
  editVisible.value = true
}

async function saveEdit() {
  editSaving.value = true
  try {
    await api.patchOrder(editForm.id, { sl: Number(editForm.sl), tp: Number(editForm.tp) })
    ElMessage.success('止损止盈已更新')
    editVisible.value = false
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    editSaving.value = false
  }
}

onMounted(async () => {
  await Promise.all([loadSettings(), load()])
  if (!disposed) timer = setInterval(load, 30000)
})
onBeforeUnmount(() => {
  disposed = true
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div v-loading="loading">
    <div class="grid stats-grid" style="margin-bottom: 14px">
      <div class="stat-card"><div class="stat-label">持仓中</div><div class="stat-value">{{ summary.open }}</div><div class="stat-sub">最多每币种 1 单</div></div>
      <div class="stat-card"><div class="stat-label">当前筛选已平仓</div><div class="stat-value">{{ summary.closed }}</div><div class="stat-sub">含止损 / 止盈 / 信号反转</div></div>
      <div class="stat-card"><div class="stat-label">已实现盈亏</div><div class="stat-value" :class="pnlClass(summary.realized)">{{ money(summary.realized) }}</div><div class="stat-sub">已扣手续费</div></div>
      <div class="stat-card"><div class="stat-label">未实现盈亏</div><div class="stat-value" :class="pnlClass(summary.unrealized)">{{ money(summary.unrealized) }}</div><div class="stat-sub">按最新标记价</div></div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">订单列表</div>
          <div class="panel-sub">修改止损止盈后由巡检任务继续监控</div>
        </div>
        <div class="row">
          <el-select v-model="filters.status" size="small" style="width: 130px" @change="load">
            <el-option label="持仓中" value="OPEN" />
            <el-option label="已平仓" value="CLOSED" />
            <el-option label="全部" value="" />
          </el-select>
          <el-select v-model="filters.symbol" size="small" clearable placeholder="全部币种" style="width: 130px" @change="load">
            <el-option v-for="symbol in symbols" :key="symbol" :label="symbol" :value="symbol" />
          </el-select>
          <el-button size="small" @click="load">刷新</el-button>
        </div>
      </div>

      <el-table :data="orders" class="table-compact" empty-text="暂无订单">
        <el-table-column prop="id" label="#" width="56" />
        <el-table-column prop="symbol" label="币种" width="96" />
        <el-table-column label="方向" width="76">
          <template #default="{ row }">
            <el-tag size="small" :type="row.side === 'LONG' ? 'success' : 'danger'" effect="plain">{{ sideLabel(row.side) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="86">
          <template #default="{ row }">
            <el-tag size="small" effect="plain" :type="row.status === 'OPEN' ? 'warning' : 'info'">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="数量" width="100"><template #default="{ row }"><span class="mono">{{ row.qty }}</span></template></el-table-column>
        <el-table-column label="开仓价" width="96"><template #default="{ row }"><span class="mono">{{ price(row.entry_price) }}</span></template></el-table-column>
        <el-table-column label="标记 / 平仓价" width="120">
          <template #default="{ row }"><span class="mono">{{ price(row.status === 'OPEN' ? row.mark_price : row.close_price) }}</span></template>
        </el-table-column>
        <el-table-column label="杠杆" width="62"><template #default="{ row }">×{{ row.leverage }}</template></el-table-column>
        <el-table-column label="保证金" width="96"><template #default="{ row }"><span class="mono">{{ money(row.margin) }}</span></template></el-table-column>
        <el-table-column label="止损 / 止盈" min-width="145">
          <template #default="{ row }"><span class="mono">{{ price(row.sl) }} / {{ price(row.tp) }}</span></template>
        </el-table-column>
        <el-table-column label="盈亏" width="130">
          <template #default="{ row }">
            <span class="mono" :class="pnlClass(row.status === 'OPEN' ? row.unrealized_pnl : row.realized_pnl)">
              {{ money(row.status === 'OPEN' ? row.unrealized_pnl : row.realized_pnl) }}
            </span>
            <div class="muted" v-if="row.status === 'OPEN'">浮盈 {{ pct(row.unrealized_pnl / Math.max(row.margin, 1) * 100) }}</div>
          </template>
        </el-table-column>
        <el-table-column label="手续费" width="96"><template #default="{ row }"><span class="mono">{{ money(Number(row.entry_fee || 0) + Number(row.exit_fee || 0)) }}</span></template></el-table-column>
        <el-table-column label="策略 / 平仓原因" min-width="160">
          <template #default="{ row }">
            <div>{{ strategyLabel(row.strategy) }}</div>
            <div class="muted">{{ row.status === 'CLOSED' ? closeReasonLabel(row.close_reason) : '持仓监控中' }}</div>
          </template>
        </el-table-column>
        <el-table-column label="时间" min-width="150">
          <template #default="{ row }">
            <div class="muted">开 {{ tsLabel(row.created_at) }}</div>
            <div class="muted" v-if="row.closed_at">平 {{ tsLabel(row.closed_at) }}</div>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="190" fixed="right">
          <template #default="{ row }">
            <template v-if="row.status === 'OPEN'">
              <el-button size="small" link type="primary" @click="openEdit(row)">改止损止盈</el-button>
              <el-button size="small" link type="warning" @click="reverseOrder(row)">反手</el-button>
              <el-button size="small" link type="danger" @click="closeOrder(row)">平仓</el-button>
            </template>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <el-dialog v-model="editVisible" title="调整止损止盈" width="420px">
      <div class="form-grid" style="grid-template-columns: 1fr">
        <div>
          <div class="muted">当前标记价 {{ price(editForm.mark) }} · {{ sideLabel(editForm.side) }}</div>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">止损价</div>
          <el-input-number v-model="editForm.sl" :precision="2" :step="10" style="width: 100%" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">止盈价</div>
          <el-input-number v-model="editForm.tp" :precision="2" :step="10" style="width: 100%" />
        </div>
      </div>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" :loading="editSaving" @click="saveEdit">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>
