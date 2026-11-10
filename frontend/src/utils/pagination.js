/**
 * 列表分页口径。
 *
 * 与后端 `backend/app/pagination.py` 的 `MAX_PAGE_SIZE` 必须保持一致：后端对
 * `page_size > PAGE_SIZE` 直接返回 422，前端不要传更大的值（包括 `page_size: 100`
 * 这种「顺手多拿一点」的写法，会被拒绝）。
 */
export const PAGE_SIZE = 15

/** 兼容旧调用：把「想拿 N 条」收敛到合法分页区间。 */
export function clampPageSize(value = PAGE_SIZE) {
  const size = Number(value)
  if (!Number.isFinite(size) || size < 1) return PAGE_SIZE
  return Math.min(Math.floor(size), PAGE_SIZE)
}

/** 页码越界时回退到最后一页，避免删到空页后停在空白表格上。 */
export function lastPage(total, pageSize = PAGE_SIZE) {
  return Math.max(1, Math.ceil(Number(total || 0) / pageSize))
}
