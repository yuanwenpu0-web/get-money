<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  option: { type: Object, required: true },
  height: { type: String, default: '320px' },
})

// 把缩放/平移事件透给页面，页面据此在拖到左边缘时加载更早的K线。
const emit = defineEmits(['datazoom'])

const el = ref(null)
const loading = ref(true)
const error = ref('')
let chart = null
let observer = null
let disposed = false

function render() {
  if (!chart) return
  chart.setOption(props.option, true)
}

async function loadChart() {
  if (disposed || chart) return
  loading.value = true
  error.value = ''
  try {
    const echarts = await import('../utils/chartEngine')
    if (disposed || !el.value) return
    chart = echarts.init(el.value, 'dark', { renderer: 'canvas' })
    chart.getDom().style.backgroundColor = 'transparent'
    // 只用事件里的 start/end，避免 getOption() 在每次拖动时复制整份配置
    chart.on('datazoom', (params) => {
      const item = Array.isArray(params?.batch) && params.batch.length ? params.batch[0] : params
      const start = Number(item?.start)
      const end = Number(item?.end)
      if (!Number.isFinite(start) || !Number.isFinite(end)) return
      emit('datazoom', { start, end })
    })
    render()
    observer = new ResizeObserver(() => chart && chart.resize())
    observer.observe(el.value)
  } catch (failure) {
    if (observer) observer.disconnect()
    if (chart) chart.dispose()
    observer = null
    chart = null
    if (!disposed) error.value = '图表加载失败，请重试'
  } finally {
    if (!disposed) loading.value = false
  }
}

onMounted(loadChart)

watch(() => props.option, render, { deep: true })

onBeforeUnmount(() => {
  disposed = true
  if (observer) observer.disconnect()
  if (chart) chart.dispose()
  chart = null
})
</script>

<template>
  <div class="chart-container" :style="{ height }" :aria-busy="loading">
    <div ref="el" :style="{ width: '100%', height: '100%' }"></div>
    <div v-if="loading" class="chart-placeholder muted" role="status">图表加载中…</div>
    <div v-else-if="error" class="chart-placeholder" role="alert">
      <span class="muted">{{ error }}</span>
      <el-button size="small" link type="primary" @click="loadChart">重试</el-button>
    </div>
  </div>
</template>
