<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  option: { type: Object, required: true },
  height: { type: String, default: '320px' },
})

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
