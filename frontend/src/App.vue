<script setup>
import { ref } from 'vue'

const example = `//@version=6
strategy("EMA cross")
fast = ta.ema(close, 9)
slow = ta.ema(close, 21)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.entry("Short", strategy.short)
`
const source = ref(example)
const symbol = ref('BTCUSDT')
const limit = ref(3000)
const dataSource = ref('binance')
const accountId = ref('demo')
const paper = ref(null)
const paperStatus = ref(null)
const quantity = ref(0.001)
const equity = ref(10000)
const fee = ref(0.0004)
const slippage = ref(0.0002)
const candles = ref(null)
const candleName = ref('')
const reference = ref(null)
const referenceName = ref('')
const numericColumns = ref('')
const generated = ref('')
const backtest = ref(null)
const comparison = ref(null)
const busy = ref('')
const error = ref('')

// Support quoted CSV fields, including embedded commas and newlines.
function parseCsv(text) {
  const rows = []; let row = []; let field = ''; let quoted = false
  const input = text.replace(/^\uFEFF/, '')
  for (let i = 0; i < input.length; i++) {
    const c = input[i]
    if (quoted) {
      if (c === '"' && input[i + 1] === '"') { field += '"'; i++ }
      else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = '' }
    else field += c
  }
  if (quoted) throw new Error('CSV 引号未闭合')
  if (field || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row) }
  if (rows.length < 2) throw new Error('CSV 没有数据行')
  const headers = rows.shift().map(x => x.trim())
  if (new Set(headers).size !== headers.length) throw new Error('CSV 存在重复列名')
  return rows.filter(r => r.some(x => x.trim())).map((r, index) => {
    if (r.length !== headers.length) throw new Error(`CSV 第 ${index + 2} 行列数不一致`)
    const item = {}
    headers.forEach((name, i) => { item[name] = r[i].trim() })
    if (!('open_time' in item) && 'time' in item) item.open_time = item.time
    if ('open_time' in item) {
      const raw = item.open_time
      const numericTime = /^\d+(\.\d+)?$/.test(raw)
      if (!numericTime && !/(?:Z|[+-]\d{2}:?\d{2})$/i.test(raw)) throw new Error(`CSV 第 ${index + 2} 行时间必须包含时区`)
      const time = numericTime ? Number(raw) : Date.parse(raw)
      if (!Number.isFinite(time)) throw new Error(`CSV 第 ${index + 2} 行时间无法识别`)
      item.open_time = time < 1e11 ? time * 1000 : time
    }
    for (const [key, value] of Object.entries(item)) {
      if (['long_entry', 'short_entry', 'long_close', 'short_close'].includes(key)) {
        if (!['true', 'false', '1', '0'].includes(String(value).toLowerCase())) throw new Error(`CSV 第 ${index + 2} 行 ${key} 必须是 true/false 或 1/0`)
        item[key] = ['true', '1'].includes(String(value).toLowerCase())
      } else if (key !== 'time' && key !== 'open_time' && value !== '' && Number.isFinite(Number(value))) item[key] = Number(value)
      else if (value === '') item[key] = null
    }
    return item
  })
}
async function loadCsv(event, type) {
  error.value = ''
  const file = event.target.files?.[0]
  if (!file) return
  if (type === 'candles') { candles.value = null; candleName.value = '' }
  else { reference.value = null; referenceName.value = '' }
  try {
    const rows = parseCsv(await file.text())
    if (type === 'candles') { candles.value = rows; candleName.value = file.name }
    else { reference.value = rows; referenceName.value = file.name }
  } catch (e) { error.value = e.message; event.target.value = '' }
}
async function request(path, payload) {
  const response = await fetch(`/api/pine/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
  return data
}
async function run(action) {
  error.value = ''; busy.value = action
  try {
    if (action === 'convert') {
      const data = await request('convert', { source: source.value })
      generated.value = data.python
    } else if (action === 'backtest') {
      backtest.value = await request('backtest', { source: source.value, ...(candles.value ? { candles: candles.value } : { symbol: symbol.value, interval: '1m', limit: Number(limit.value), data_source: dataSource.value }), initial_equity: Number(equity.value), quantity: Number(quantity.value), fee_rate: Number(fee.value), slippage: Number(slippage.value) })
    } else {
      if (!candles.value || !reference.value) throw new Error('对照验证需要同时选择 K 线 CSV 和 TradingView 参考 CSV')
      comparison.value = await request('compare', { source: source.value, candles: candles.value, reference: reference.value, numeric_columns: numericColumns.value.split(',').map(x => x.trim()).filter(Boolean) })
    }
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function paperAction(action) {
  error.value = ''; busy.value = action
  try {
    if (!/^[A-Za-z0-9_-]+$/.test(accountId.value)) throw new Error('账户 ID 只能包含字母、数字、下划线和连字符')
    const path = `paper/${encodeURIComponent(accountId.value)}`
    if (action === 'create') paper.value = await request(path, { source: source.value, settings: { initial_equity: Number(equity.value), quantity: Number(quantity.value), fee_rate: Number(fee.value), slippage: Number(slippage.value), interval_ms: 60000 } })
    else if (action === 'append') {
      if (!candles.value) throw new Error('请先选择已收盘 K 线 CSV')
      paper.value = await request(`${path}/candles`, { candles: candles.value })
    } else {
      const response = await fetch(`/api/pine/${path}`)
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
      paper.value = data
    }
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function runnerAction(action) {
  error.value = ''; busy.value = action
  try {
    if (!/^[A-Za-z0-9_-]+$/.test(accountId.value)) throw new Error('账户 ID 只能包含字母、数字、下划线和连字符')
    const path = `paper/${encodeURIComponent(accountId.value)}/${action}`
    if (action === 'status') {
      const response = await fetch(`/api/pine/${path}`)
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
      paperStatus.value = data
    } else paperStatus.value = await request(path, action === 'start' ? { symbol: symbol.value } : {})
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
function downloadPython() {
  const url = URL.createObjectURL(new Blob([generated.value], { type: 'text/x-python' }))
  const link = document.createElement('a'); link.href = url; link.download = 'converted_strategy.py'; link.click(); URL.revokeObjectURL(url)
}
function number(value) { return Number.isFinite(Number(value)) ? Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 4 }) : '—' }
</script>

<template>
  <div class="workbench">
    <header class="header"><div><span class="eyebrow">LOCAL STRATEGY LAB <span class="dot">●</span> RESEARCH</span><h1>策略转换工作台</h1><p>从 TradingView 源码到本地验证，一根 K 线一根 K 线地确认。</p></div><div class="status"><span class="status-dot"></span> 本地研究模式</div></header>
    <main>
      <section class="pipeline"><div><b>01</b><span>导入 Pine 源码</span></div><i></i><div><b>02</b><span>转换与对照</span></div><i></i><div><b>03</b><span>本地回测</span></div><i></i><div class="muted"><b>04</b><span>交易执行</span></div></section>
      <div class="layout">
        <section class="panel source-panel"><div class="panel-head"><div><span class="section-number">01 / SOURCE</span><h2>Pine 策略</h2><p>粘贴完整源码。无法可靠转换的语法会直接报错。</p></div><label class="file-link">导入 .pine<input type="file" accept=".pine,.txt" @change="async e => { const f = e.target.files?.[0]; if (f) source = await f.text() }"></label></div><textarea v-model="source" spellcheck="false" aria-label="Pine Script 源码"></textarea><div class="actions"><button class="primary" :disabled="!!busy" @click="run('convert')">{{ busy === 'convert' ? '正在检查…' : '检查并生成 Python' }} <span>↗</span></button><button v-if="generated" class="secondary" @click="downloadPython">下载 Python</button></div><details v-if="generated"><summary>查看生成的 Python</summary><pre>{{ generated }}</pre></details></section>
        <aside class="panel data-panel"><div class="panel-head"><div><span class="section-number">02 / DATA</span><h2>行情与参数</h2><p>使用本地数据库或上传同源 K 线。</p></div></div><div class="fields"><label>行情来源<select v-model="dataSource"><option value="binance">Binance 公共行情</option><option value="local">本地数据库</option></select></label><label>交易对<select v-model="symbol"><option>BTCUSDT</option><option>ETHUSDT</option></select></label><label>周期<input value="1m" disabled></label><label>最近 K 线数<input v-model.number="limit" type="number" min="2"></label><label>初始权益 / USDT<input v-model.number="equity" type="number" min="0" step="any"></label><label>数量 / 基础资产<input v-model.number="quantity" type="number" min="0" step="any"></label><label>单边手续费率<input v-model.number="fee" type="number" min="0" step="any"></label><label>单边滑点率<input v-model.number="slippage" type="number" min="0" step="any"></label></div><label class="upload">{{ candleName || '选择 K 线 CSV（可选）' }} <span>↗</span><input type="file" accept=".csv" @change="e => loadCsv(e, 'candles')"></label><button v-if="candles" class="clear" @click="candles = null; candleName = ''">清除 CSV，使用本地数据</button><p class="hint">CSV 列：open_time、open、high、low、close、volume；时间使用毫秒或秒。</p></aside>
      </div>
      <section class="panel compare-panel"><div><span class="section-number">03 / VALIDATION</span><h2>TradingView 逐 K 线对照</h2><p>使用同源行情和导出的信号列；通过对照不代表实盘表现相同。</p></div><div class="compare-controls"><label class="upload">{{ referenceName || '选择参考 CSV' }} <span>↗</span><input type="file" accept=".csv" @change="e => loadCsv(e, 'reference')"></label><input v-model="numericColumns" placeholder="指标列名，逗号分隔（可选）"><button class="secondary" :disabled="!!busy" @click="run('compare')">{{ busy === 'compare' ? '对照中…' : '运行对照' }}</button></div><div v-if="comparison" class="report"><strong :class="comparison.passed ? 'good' : 'bad'">{{ comparison.passed ? '对照通过' : '发现差异' }}</strong><pre>{{ JSON.stringify(comparison, null, 2) }}</pre></div></section>
      <section class="panel results"><div class="panel-head"><div><span class="section-number">04 / BACKTEST</span><h2>本地回测</h2><p>收盘生成信号，下一根开盘撮合；固定数量，计入手续费和滑点。</p></div><button class="primary" :disabled="!!busy" @click="run('backtest')">{{ busy === 'backtest' ? '回测中…' : '运行回测' }} <span>↗</span></button></div><div v-if="backtest" class="metrics"><div><span>初始权益</span><strong>{{ number(backtest.result.initial_equity) }}</strong></div><div><span>期末权益</span><strong>{{ number(backtest.result.final_equity) }}</strong></div><div><span>手续费</span><strong>{{ number(backtest.result.fees) }}</strong></div><div><span>已平仓交易</span><strong>{{ backtest.result.trades?.length ?? 0 }}</strong></div></div><p v-else class="empty">结果将在这里展示。请先确认策略语法与数据来源。</p><details v-if="backtest"><summary>查看成交和执行详情</summary><pre>{{ JSON.stringify(backtest.result, null, 2) }}</pre></details></section>
      <section class="panel results"><div class="panel-head"><div><span class="section-number">05 / PAPER ACCOUNT</span><h2>本地模拟账户</h2><p>创建后使用已收盘 CSV 增量更新并持久化；每次重放历史，按下一根开盘价模拟，并非实时可成交价格。</p></div></div><div class="compare-controls"><input v-model="accountId" placeholder="账户 ID"><button class="secondary" :disabled="!!busy" @click="paperAction('create')">创建账户</button><button class="secondary" :disabled="!!busy" @click="paperAction('read')">读取账户</button><button class="primary" :disabled="!!busy" @click="paperAction('append')">追加已收盘 K 线</button></div><p class="hint">创建时固定当前源码和参数；追加使用上方选择的 K 线 CSV。也可显式启动 Binance 公共行情轮询，服务重启后需重新启动。空账户从最新已收盘 K 线开始，指标需要预热；不发送真实订单。</p><div class="actions"><button class="secondary" :disabled="!!busy" @click="runnerAction('start')">启动行情轮询</button><button class="secondary" :disabled="!!busy" @click="runnerAction('stop')">停止轮询</button><button class="secondary" :disabled="!!busy" @click="runnerAction('status')">查询状态</button></div><p v-if="paperStatus" class="hint">运行状态：{{ paperStatus.status }} · {{ paperStatus.symbol || '—' }} <span v-if="paperStatus.error"> · {{ paperStatus.error }}</span></p><div v-if="paper" class="metrics paper-metrics"><div><span>账户权益</span><strong>{{ number(paper.result.final_equity) }}</strong></div><div><span>已处理 K 线</span><strong>{{ paper.candles.length }}</strong></div><div><span>已平仓交易</span><strong>{{ paper.result.trades.length }}</strong></div></div><details v-if="paper"><summary>查看账户结果</summary><pre>{{ JSON.stringify(paper.result, null, 2) }}</pre></details></section>
      <div v-if="error" class="error" role="alert">{{ error }}</div><footer>当前转换范围有限，必须使用 TradingView 数据逐项验证。交易执行能力以实际开放接口为准。</footer>
    </main>
  </div>
</template>

<style scoped>
.workbench{min-height:100vh;background:#0c1114;color:#e6eee9;font-family:'Segoe UI','Microsoft YaHei',sans-serif}.header,main{max-width:1440px;margin:auto}.header{padding:48px 48px 32px;display:flex;justify-content:space-between;gap:20px;align-items:center}.eyebrow,.section-number{font:600 11px/1.5 Consolas,monospace;letter-spacing:.14em;color:#80a89a}.dot{color:#58e4ae;margin:0 5px}h1{font-size:34px;letter-spacing:-.03em;margin:12px 0 8px}h2{font-size:19px;margin:7px 0}p{color:#95a8a0;font-size:13px;line-height:1.6;margin:0}.status{font-size:12px;color:#a8cabb;border:1px solid #31453e;border-radius:20px;padding:9px 13px;white-space:nowrap}.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#68dca8;margin-right:6px}main{padding:0 48px 48px}.pipeline{display:flex;align-items:center;gap:18px;border-block:1px solid #26332f;padding:20px 0;margin-bottom:24px}.pipeline div{display:flex;align-items:center;gap:9px;font-size:12px;white-space:nowrap}.pipeline b{color:#63dfa9;font:12px Consolas}.pipeline i{height:1px;background:#2d3c36;flex:1}.pipeline .muted{color:#6d7e77}.layout{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(300px,1fr);gap:18px}.panel{border:1px solid #293a33;background:#121a18;border-radius:8px;padding:24px}.panel-head{display:flex;justify-content:space-between;align-items:flex-start;gap:15px;margin-bottom:20px}.file-link{font-size:12px;color:#86dcb4;cursor:pointer;white-space:nowrap}.file-link input,.upload input{display:none}textarea,select,input{width:100%;color:#dcebe3;background:#0b1310;border:1px solid #304139;border-radius:5px;padding:10px 11px;font:13px 'Segoe UI','Microsoft YaHei',sans-serif}textarea{min-height:310px;resize:vertical;font:13px/1.65 Consolas,monospace;tab-size:4;color:#b5d8c6}textarea:focus,input:focus,select:focus{outline:2px solid #59c994;outline-offset:1px}.actions{display:flex;gap:10px;margin-top:16px}button{cursor:pointer;border-radius:5px;font-weight:600;padding:11px 15px;font-size:13px}button:disabled{opacity:.5;cursor:wait}.primary{background:#7ce4b1;color:#10241a;border:1px solid #7ce4b1}.primary span{margin-left:16px}.secondary{background:#202d27;color:#d8ece1;border:1px solid #3a5044}.fields{display:grid;grid-template-columns:1fr 1fr;gap:13px}.fields label{font-size:11px;color:#a4b8ad}.fields input,.fields select{margin-top:6px}.upload{margin-top:18px;display:flex;justify-content:space-between;border:1px dashed #4a6657;border-radius:5px;padding:12px;color:#b4d5c2;font-size:12px;cursor:pointer}.clear{border:0;padding:8px 0;background:none;color:#98c2aa;font-size:11px}.hint{margin-top:10px;font-size:11px}.compare-panel,.results{margin-top:18px}.compare-controls{display:flex;gap:10px;align-items:center;margin-top:15px}.compare-controls .upload{margin:0;min-width:210px}.compare-controls input{max-width:300px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #2d3c35;padding-top:20px;gap:15px}.paper-metrics{margin-top:18px}.metrics span,.metrics strong{display:block}.metrics span{color:#91a79a;font-size:12px}.metrics strong{font-size:23px;margin-top:9px;font-weight:500}.empty{padding:18px 0;border-top:1px solid #2d3c35}.report{margin-top:18px}.good{color:#80e4b0}.bad{color:#ffad94}details{margin-top:18px;color:#9db6a8;font-size:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow:auto;max-height:360px;background:#0b1310;padding:14px;color:#b7cebf;font:12px/1.5 Consolas,monospace}.error{margin-top:18px;padding:14px;border:1px solid #b26959;border-radius:5px;background:#39231d;color:#ffd0bc;font-size:13px}footer{padding:24px 0;color:#788c81;font-size:11px}@media(max-width:800px){.header{padding:30px 20px 24px}.header h1{font-size:27px}main{padding:0 20px 30px}.layout{grid-template-columns:1fr}.pipeline{gap:8px;overflow:auto}.pipeline i{min-width:20px}.status{display:none}.compare-controls{align-items:stretch;flex-direction:column}.compare-controls input{max-width:none}.metrics{grid-template-columns:1fr 1fr}}
</style>
