import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

const backend = process.env.SCONE_DEV_API || 'http://127.0.0.1:7437';
const url = new URL(backend);
if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password) {
  throw new Error('SCONE_DEV_API must be an unauthenticated loopback HTTP address');
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    strictPort: true,
    fs: { strict: true, allow: [fileURLToPath(new URL('.', import.meta.url))] },
    proxy: {
      '/v1': { target: backend, changeOrigin: false },
      '/healthz': { target: backend, changeOrigin: false },
      // Local bootstrap only. /memory itself is owned by React Router.
      '/__native_console': { target: backend, changeOrigin: false, rewrite: () => '/memory' },
    },
  },
  build: {
    assetsInlineLimit: 1_000_000,
    cssCodeSplit: true,
    modulePreload: false,
    sourcemap: false,
  },
});
