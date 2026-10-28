const reasons = {
  stale_data: '行情过期，暂停开仓',
  invalid_indicators: '指标未就绪',
  no_absorption_signal: '未出现吸收形态',
  stop_too_wide: '结构止损过宽（超过 max_sl_atr）',
  stop_too_tight: '结构止损过窄（低于 min_sl_atr）',
  stop_wrong_side: '结构止损落在价格错误一侧',
  long_disabled: '已关闭做多方向',
  short_disabled: '已关闭做空方向',
  trend_filter: 'EMA 趋势过滤未通过',
  session_filter: '不在允许的交易时段',
  below_threshold: '评分未达到开仓阈值',
  excess_volatility: '波动过大',
  funding_filter_long: '资金费正极端（多头拥挤），禁止做多',
  funding_filter_short: '资金费负极端（空头拥挤），禁止做空',
  cost_filter: '波动或净盈亏比不足以覆盖成本',
}

export const signalReason = (code) => reasons[code] || code || '符合开仓条件'
