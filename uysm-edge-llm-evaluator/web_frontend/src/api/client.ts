import axios from 'axios'

/** Use Vite proxy to FastAPI (same origin) by default. */
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? '',
})
