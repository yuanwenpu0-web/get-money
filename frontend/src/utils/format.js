export function money(value, digits = 2) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return `$${number.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`
}

export function number(value, digits = 2) {
  const num = Number(value)
  if (!Number.isFinite(num)) return '—'
  return num.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function pct(value, digits = 2) {
  const num = Number(value)
  if (!Number.isFinite(num)) return '—'
  return `${num > 0 ? '+' : ''}${num.toFixed(digits)}%`
}

export function price(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return '—'
  return num >= 1 ? num.toFixed(2) : num.toFixed(6)
}

export function tsLabel(value) {
  if (!value) return '—'
  const date = typeof value === 'number' ? new Date(value) : new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString('zh-CN', { hour12: false })
}

export function pnlClass(value) {
  const num = Number(value)
  if (num > 0) return 'up'
  if (num < 0) return 'down'
  return 'flat'
}

export const sideLabel = (side) => (side === 'LONG' ? '做多' : side === 'SHORT' ? '做空' : side)

export const statusLabel = (status) =>
  ({ OPEN: '持仓中', CLOSED: '已平仓', CANCELED: '已取消' })[status] || status

export const closeReasonLabel = (reason) =>
  ({
    manual: '手动平仓',
    reverse: '手动反手',
    stop_loss: '止损',
    take_profit: '止盈',
    time_stop: '超时平仓',
    signal_reverse: '信号反转',
    close_at_end: '回测结束',
  })[reason] || reason || '—'

export const strategyLabel = (code) =>
  ({ absorption_reversal: '吸收反转' })[code] || code

export function symbolsLabel(settings, count = 0) {
  if (settings?.symbol_scope === 'all') {
    return count > 0 ? `全部永续（${count} 个）` : '全部永续'
  }
  const list = settings?.symbols || []
  return list.length ? list.join(' / ') : '—'
}
