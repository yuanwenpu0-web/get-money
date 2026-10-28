<script setup>
/**
 * 策略 × 币种池 绑定矩阵。
 *
 * 一条绑定 = (策略, 池, 周期) + 一套参数覆盖。同一个策略可以在「主流 15m」「山寨 5m」
 * 「山寨 1m」上各跑一套参数，互不影响；参数的最终生效值是
 * 全局设置 < 策略默认 < 本条绑定（后端 resolve_params 负责合并与校验）。
 */
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const props = defineProps({
  strategies: { type: Array, default: () => [] },
})

// 绑定变化后通知父级刷新（策略页的「信号预演」要用最新的绑定列表做参数层下拉）
const emit = defineEmits(['change'])

const bindings = ref([])
const pools = ref([])
const loading = ref(false)
const error = ref('')
const saving = reactive({})
const intervalOptions = ['1m', '5m', '15m', '1h', '4h', '1d']

const bindingVisible = ref(false)
const bindingSaving = ref(false)
const bindingForm = reactive({
  strategy: 'absorption_reversal',
  pool: '',
  interval: '15m',
  priority: 100,
  enabled: true,
  params: '{}',
})

const paramsVisible = ref(false)
const paramsSaving = ref(false)
const paramsTarget = ref(null)
const paramsText = ref('{}')

const poolVisible = ref(false)
const poolSaving = ref(false)
const poolForm = reactive({
  code: '',
  name: '',
  kind: 'rule',
  symbols: '',
  rule: '{"base": "all"}',
})

const poolMembersVisible = ref(false)
const poolMembers = ref([])
const poolMembersTitle = ref('')

const effectiveCount = computed(() => bindings.value.filter((row) => row.enabled).length)

function parseJson(text, label) {
  try {
    return JSON.parse(text)
  } catch (err) {
    throw new Error(`${label} 不是合法 JSON`)
  }
}

function paramsSummary(row) {
  const keys = Object.keys(row.params || {})
  if (!keys.length) return '继承策略默认'
  return keys.map((key) => `${key}=${row.params[key]}`).join(' · ')
}

function load() {
  if (loading.value) return Promise.resolve()
  loading.value = true
  error.value = ''
  return Promise.all([api.getBindings(), api.getPools()])
    .then(([bindingData, poolData]) => {
      bindings.value = bindingData.bindings || []
      pools.value = poolData.pools || []
    })
    .catch((err) => {
      error.value = err.message
    })
    .finally(() => {
      loading.value = false
    })
}

function openCreate() {
  bindingForm.strategy = props.strategies[0]?.code || 'absorption_reversal'
  bindingForm.pool = pools.value[0]?.code || ''
  bindingForm.interval = '15m'
  bindingForm.priority = 100
  bindingForm.enabled = true
  bindingForm.params = '{}'
  bindingVisible.value = true
}

async function submitCreate() {
  let params
  try {
    params = parseJson(bindingForm.params, '参数')
  } catch (err) {
    ElMessage.error(err.message)
    return
  }
  bindingSaving.value = true
  try {
    await api.createBinding({
      strategy: bindingForm.strategy,
      pool: bindingForm.pool,
      interval: bindingForm.interval,
      priority: Number(bindingForm.priority),
      enabled: bindingForm.enabled,
      params,
    })
    ElMessage.success('绑定已创建')
    bindingVisible.value = false
    await load()
    emit('change')
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    bindingSaving.value = false
  }
}

async function toggle(row) {
  saving[row.id] = true
  try {
    await api.updateBinding(row.id, { enabled: row.enabled })
    ElMessage.success(`${row.strategy} × ${row.pool} 已${row.enabled ? '启用' : '停用'}`)
    emit('change')
  } catch (err) {
    row.enabled = !row.enabled
    ElMessage.error(err.message)
  } finally {
    saving[row.id] = false
  }
}

function openParams(row) {
  paramsTarget.value = row
  paramsText.value = JSON.stringify(row.params || {}, null, 2)
  paramsVisible.value = true
}

async function saveParams() {
  let params
  try {
    params = parseJson(paramsText.value, '参数')
  } catch (err) {
    ElMessage.error(err.message)
    return
  }
  paramsSaving.value = true
  try {
    await api.updateBinding(paramsTarget.value.id, { params })
    ElMessage.success('绑定参数已保存')
    paramsVisible.value = false
    await load()
    emit('change')
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    paramsSaving.value = false
  }
}

async function removeBinding(row) {
  try {
    await ElMessageBox.confirm(
      `删除绑定「${row.strategy} × ${row.pool} × ${row.interval}」？该绑定下的参数配置会一并丢失。`,
      '删除绑定',
      { type: 'warning' },
    )
  } catch (err) {
    return
  }
  try {
    await api.deleteBinding(row.id)
    ElMessage.success('绑定已删除')
    await load()
    emit('change')
  } catch (err) {
    ElMessage.error(err.message)
  }
}

function openCreatePool() {
  poolForm.code = ''
  poolForm.name = ''
  poolForm.kind = 'rule'
  poolForm.symbols = ''
  poolForm.rule = '{"base": "all"}'
  poolVisible.value = true
}

async function submitPool() {
  let rule
  try {
    rule = parseJson(poolForm.rule || '{}', '池规则')
  } catch (err) {
    ElMessage.error(err.message)
    return
  }
  const symbols = poolForm.symbols.split(/[\s,;]+/).filter(Boolean)
  poolSaving.value = true
  try {
    await api.createPool({
      code: poolForm.code,
      name: poolForm.name || poolForm.code,
      kind: poolForm.kind,
      symbols,
      rule,
      enabled: true,
    })
    ElMessage.success('币种池已创建')
    poolVisible.value = false
    await load()
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    poolSaving.value = false
  }
}

async function showMembers(pool) {
  try {
    const data = await api.getPoolMembers(pool.code)
    poolMembers.value = data.symbols || []
    poolMembersTitle.value = `${pool.name}（${data.count} 个）`
    poolMembersVisible.value = true
  } catch (err) {
    ElMessage.error(err.message)
  }
}

async function removePool(pool) {
  try {
    await ElMessageBox.confirm(`删除币种池「${pool.name}」？`, '删除池', { type: 'warning' })
  } catch (err) {
    return
  }
  try {
    await api.deletePool(pool.code)
    ElMessage.success('币种池已删除')
    await load()
  } catch (err) {
    ElMessage.error(err.message)
  }
}

onMounted(load)
</script>

<template>
  <div class="panel" style="margin-bottom: 14px">
    <div class="panel-head">
      <div>
        <div class="panel-title">策略 × 币种池 绑定</div>
        <div class="panel-sub">
          同一策略可在不同池上跑不同周期与参数（全局设置 &lt; 策略默认 &lt; 本条绑定）。
          生效绑定 {{ effectiveCount }} 条；同币种落入多个池时按评分择优只开一仓。
        </div>
      </div>
      <div class="row">
        <el-button size="small" :loading="loading" @click="load">刷新</el-button>
        <el-button size="small" @click="openCreatePool">新建币种池</el-button>
        <el-button size="small" type="primary" :disabled="!pools.length" @click="openCreate">新增绑定</el-button>
      </div>
    </div>

    <div v-if="error" class="row" role="alert" style="margin-bottom: 12px">
      <span class="down">绑定加载失败：{{ error }}</span>
      <el-button size="small" @click="load">重试</el-button>
    </div>

    <div v-if="!bindings.length && !loading" class="empty-hint">
      还没有任何绑定。没有绑定就不会有币种被评估——先建一个「主流币 × 15m」。
    </div>

    <el-table v-else :data="bindings" class="table-compact">
      <el-table-column label="策略" min-width="150">
        <template #default="{ row }">
          <div>{{ row.strategy_name || row.strategy }}</div>
          <div class="muted mono">{{ row.strategy }}</div>
        </template>
      </el-table-column>
      <el-table-column label="币种池" min-width="150">
        <template #default="{ row }">
          <el-button link size="small" @click="showMembers({ code: row.pool, name: row.pool_name })">
            {{ row.pool_name || row.pool }}
          </el-button>
          <div class="muted">{{ row.member_count ?? 0 }} 个币种</div>
        </template>
      </el-table-column>
      <el-table-column prop="interval" label="周期" width="80" />
      <el-table-column label="启用" width="80">
        <template #default="{ row }">
          <el-switch v-model="row.enabled" :loading="saving[row.id]" @change="toggle(row)" />
        </template>
      </el-table-column>
      <el-table-column prop="priority" label="优先级" width="90" />
      <el-table-column label="参数覆盖" min-width="260">
        <template #default="{ row }">
          <span class="muted mono">{{ paramsSummary(row) }}</span>
          <div v-if="row.error" class="down">{{ row.error }}</div>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="150">
        <template #default="{ row }">
          <el-button size="small" link @click="openParams(row)">参数</el-button>
          <el-button size="small" link type="danger" @click="removeBinding(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="row" style="margin-top: 14px">
      <span class="muted">币种池</span>
      <el-tag
        v-for="pool in pools"
        :key="pool.code"
        size="small"
        effect="plain"
        closable
        @close="removePool(pool)"
        @click="showMembers(pool)"
        style="cursor: pointer"
      >
        {{ pool.name }} · {{ pool.member_count }} 币
      </el-tag>
      <span v-if="!pools.length" class="muted">暂无币种池</span>
    </div>

    <el-dialog v-model="bindingVisible" title="新增绑定" width="520px">
      <div class="form-grid">
        <div>
          <div class="muted" style="margin-bottom: 4px">策略</div>
          <el-select v-model="bindingForm.strategy" style="width: 100%">
            <el-option v-for="item in strategies" :key="item.code" :label="item.name" :value="item.code" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">币种池</div>
          <el-select v-model="bindingForm.pool" style="width: 100%">
            <el-option v-for="pool in pools" :key="pool.code" :label="`${pool.name}（${pool.member_count}）`" :value="pool.code" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">K线周期</div>
          <el-select v-model="bindingForm.interval" style="width: 100%">
            <el-option v-for="item in intervalOptions" :key="item" :label="item" :value="item" />
          </el-select>
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">优先级（仅排序）</div>
          <el-input-number v-model="bindingForm.priority" :min="0" :max="10000" style="width: 100%" />
        </div>
      </div>
      <div class="muted" style="margin: 12px 0 4px">参数覆盖（JSON，留空则全用策略默认）</div>
      <el-input v-model="bindingForm.params" type="textarea" :rows="7" class="mono" />
      <template #footer>
        <el-button @click="bindingVisible = false">取消</el-button>
        <el-button type="primary" :loading="bindingSaving" @click="submitCreate">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="paramsVisible" :title="`绑定参数 · ${paramsTarget?.strategy} × ${paramsTarget?.pool}`" width="560px">
      <div class="muted" style="margin-bottom: 8px">
        只写需要覆盖的字段；同名键会覆盖策略默认，删除某个键即恢复继承。
        保存时会与当前全局设置合并后完整校验，非法值直接拒绝。
      </div>
      <el-input v-model="paramsText" type="textarea" :rows="10" class="mono" />
      <template #footer>
        <el-button @click="paramsVisible = false">取消</el-button>
        <el-button type="primary" :loading="paramsSaving" @click="saveParams">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="poolVisible" title="新建币种池" width="520px">
      <div class="form-grid">
        <div>
          <div class="muted" style="margin-bottom: 4px">代码（小写字母/数字/下划线）</div>
          <el-input v-model="poolForm.code" placeholder="e.g. alt_top100" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">名称</div>
          <el-input v-model="poolForm.name" placeholder="e.g. 山寨前100" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 4px">类型</div>
          <el-select v-model="poolForm.kind" style="width: 100%">
            <el-option label="规则（动态）" value="rule" />
            <el-option label="静态列表" value="static" />
          </el-select>
        </div>
      </div>
      <div class="muted" style="margin: 12px 0 4px">静态成员（逗号/空格分隔，仅 static 用）</div>
      <el-input v-model="poolForm.symbols" type="textarea" :rows="3" class="mono" placeholder="BTCUSDT, ETHUSDT" />
      <div class="muted" style="margin: 12px 0 4px">
        规则（JSON）：base=all/major/static，可加 min_quote_volume_usd、exclude_major、max_symbols、exclude
      </div>
      <el-input v-model="poolForm.rule" type="textarea" :rows="4" class="mono" />
      <template #footer>
        <el-button @click="poolVisible = false">取消</el-button>
        <el-button type="primary" :loading="poolSaving" :disabled="!poolForm.code" @click="submitPool">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="poolMembersVisible" :title="`成员 · ${poolMembersTitle}`" width="520px">
      <div class="mono" style="line-height: 1.8; word-break: break-all">
        {{ poolMembers.join(', ') || '（空池）' }}
      </div>
    </el-dialog>
  </div>
</template>
