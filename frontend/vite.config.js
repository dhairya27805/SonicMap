import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
// The frontend always calls the relative path `/api/...`; the dev server proxies it
// to the FastAPI backend. Override the target with VITE_BACKEND_URL in frontend/.env.local.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendUrl = env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    server: {
      host: env.HOST || '127.0.0.1',
      port: Number(env.PORT) || 5173,
      allowedHosts: true,
      proxy: {
        '/api': { target: backendUrl, changeOrigin: false },
      },
    },
  }
})
