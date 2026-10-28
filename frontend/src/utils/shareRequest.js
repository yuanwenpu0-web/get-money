// Coalesce simultaneous reads without retaining stale responses between refreshes.
export function shareRequest(request) {
  let pending = null
  function read() {
    if (!pending) {
      const current = Promise.resolve().then(request).finally(() => {
        if (pending === current) pending = null
      })
      pending = current
    }
    return pending
  }
  read.invalidate = () => { pending = null }
  return read
}
