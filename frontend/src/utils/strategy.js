const reasons = {
  stale_data: '行情过期，暂停开仓',
  invalid_indicators: '指标未就绪',
  below_threshold: '评分未达到开仓阈值',
  excess_volatility: '波动过大',
  model_required: '需要已训练模型',
  invalid_probability: '模型概率无效',
  model_edge: '模型方向优势不足',
  weak_trend: '趋势强度不足',
  trend_conflict: '快慢趋势方向不一致',
  trending_market: '强趋势中暂停均值回归',
  band_confirmation: '布林带尚未确认',
  cost_filter: '波动或净盈亏比不足以覆盖成本',
}

export const signalReason = (code) => reasons[code] || code || '符合开仓条件'
