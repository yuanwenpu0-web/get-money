// K线公共工具：周期换算、按成交时间定窗口、标记落点与自动定位。
// 所有用到 K 线的页面共用，保证切换 1h / 5m 等周期时买卖点都能定位到同一笔成交。

export const INTERVAL_MS = {
  '1m': 60_000,
  '3m': 180_000,
  '5m': 300_000,
  '15m': 900_000,
  '30m': 1_800_000,
  '1h': 3_600_000,
  '2h': 7_200_000,
  '4h': 14_400_000,
  '6h': 21_600_000,
  '12h': 43_200_000,
  '1d': 86_400_000,
  '3d': 259_200_000,
  '1w': 604_800_000,
}

export const KLINE_MAX_LIMIT = 1500

export function intervalMs(interval) {
  return INTERVAL_MS[interval] || 0
}

// 只按「时间」定位，不要求时间正好等于某根开盘时间：
// 先找包含该时间的K线，找不到就退到不晚于该时间的最近一根（平仓晚于最后一根时也能标在末根）。
// 完全在数据范围外返回 null，由调用方决定是否继续加载更早数据。
export function locateBar(bars, timeValue, interval, options = {}) {
  const duration = intervalMs(interval)
  const time = timeValue ? Date.parse(timeValue) : NaN
  if (!Number.isFinite(time) || !duration || !Array.isArray(bars) || !bars.length) return null
  const first = Number(bars[0].time)
  if (!Number.isFinite(first)) return null
  const last = Number(bars[bars.length - 1].time)
  // 允许成交时间略晚于最后一根（未闭合桶不返回），只容忍两根。
  if (time < first) return null
  if (time > last + duration * 2) return null
  let low = 0
  let high = bars.length - 1
  let found = -1
  while (low <= high) {
    const mid = (low + high) >> 1
    const barTime = Number(bars[mid].time)
    if (barTime <= time) {
      found = mid
      low = mid + 1
    } else {
      high = mid - 1
    }
  }
  if (found < 0) return null
  const barTime = Number(bars[found].time)
  const inside = time < barTime + duration
  // 成交时间落在更晚的K线里时 binary search 已给出正确的那根；否则说明中间有缺口。
  return { index: found, exact: inside, offsetBars: Math.max(0, Math.round((time - barTime) / duration)) }
}

// 覆盖 anchors（成交时间）需要多少根K线；带一点前视余量，上限受后端 1500 根限制。
export function barsToCover(interval, anchors, nowMs = Date.now(), options = {}) {
  const duration = intervalMs(interval)
  const times = (anchors || []).map((value) => (value ? Date.parse(value) : NaN)).filter(Number.isFinite)
  if (!duration || !times.length) return options.min || 300
  const oldest = Math.min(...times)
  const newest = Math.max(nowMs, ...times)
  const span = Math.max(duration, newest - oldest)
  const padRatio = options.padRatio ?? 0.12
  const min = options.min ?? 240
  const max = Math.min(options.max ?? KLINE_MAX_LIMIT, KLINE_MAX_LIMIT)
  const needed = Math.ceil((span * (1 + padRatio)) / duration) + 20
  return Math.max(min, Math.min(max, needed))
}

// 让开仓/平仓点都落在视窗内：以标记区间为中心，至少保留 minSpanBars 根上下文。
export function zoomForIndexes(total, indexes, options = {}) {
  const valid = (indexes || []).filter((value) => Number.isInteger(value) && value >= 0 && value < total)
  const minSpan = Math.min(total, options.minSpanBars ?? 80)
  const pad = options.padBars ?? Math.max(6, Math.round(minSpan * 0.15))
  if (!total) return { start: 0, end: 100 }
  if (!valid.length) return { start: Math.max(0, 100 - (minSpan / total) * 100), end: 100 }
  let from = Math.min(...valid) - pad
  let to = Math.max(...valid) + pad
  if (to - from + 1 < minSpan) {
    const extra = minSpan - (to - from + 1)
    from -= Math.floor(extra / 2)
    to += Math.ceil(extra / 2)
  }
  from = Math.max(0, Math.min(from, total - 1))
  to = Math.min(total - 1, Math.max(to, from))
  const start = (from / total) * 100
  const end = ((to + 1) / total) * 100
  return { start: Math.max(0, start), end: Math.min(100, end) }
}

// 标记方向：买单用向上三角、卖单用向下三角，比同形状 pin 更容易一眼分辨。
export function tradeMarker(trade) {
  const isBuy = String(trade.side || '').toUpperCase() !== 'SHORT' && trade.action !== 'exit'
  return isBuy
    ? { symbol: 'triangle', symbolRotate: 0, color: '#f59e0b', label: '买' }
    : { symbol: 'triangle', symbolRotate: 180, color: '#38bdf8', label: '卖' }
}
