const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export async function callApi(path, options = {}) {
  const headers = { ...options.headers }
  if (options.body instanceof FormData) delete headers['Content-Type']
  else headers['Content-Type'] ||= 'application/json'
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers,
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = body.detail
    throw new Error(typeof detail === 'string' ? detail : detail?.message || 'Something went wrong. Please try again.')
  }
  return body
}
