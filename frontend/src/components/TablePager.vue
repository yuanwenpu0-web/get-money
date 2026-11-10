/**
 * 统一表格分页条。
 *
 * 全站列表单页最多 15 条（与后端 `app/pagination.py::MAX_PAGE_SIZE` 对齐），所以这里
 * 不提供 page-size 选择器——选了也会被后端 422 拒绝。所有列表都用同一个分页器，
 * 样式、文案、翻页行为保持一致。
 */
<script setup>
import { PAGE_SIZE } from '../utils/pagination'

const props = defineProps({
  page: { type: Number, default: 1 },
  total: { type: Number, default: 0 },
  pageSize: { type: Number, default: PAGE_SIZE },
  disabled: { type: Boolean, default: false },
  // 数据量大时（订单、日志）显示跳页输入，小列表只保留上下页
  jumper: { type: Boolean, default: false },
  size: { type: String, default: 'small' },
})
const emit = defineEmits(['change'])

function onChange(page) {
  if (props.disabled || page === props.page) return
  emit('change', page)
}
</script>

<template>
  <div v-if="pageSize && total > 0" class="table-pagination">
    <el-pagination
      :current-page="page"
      :page-size="pageSize"
      :total="total"
      :disabled="disabled"
      :size="size"
      :layout="jumper ? 'total, prev, pager, next, jumper' : 'total, prev, pager, next'"
      background
      @current-change="onChange"
    />
  </div>
</template>
