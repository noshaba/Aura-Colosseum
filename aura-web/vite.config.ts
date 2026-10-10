import { defineConfig } from 'vite'
export default defineConfig({
  // @solana/web3.js (via bn.js) imports Node's 'buffer'. Point it at the npm 'buffer'
  // package (already a dependency) instead of Vite's empty browser stub.
  resolve: { alias: { buffer: 'buffer/' } },
  optimizeDeps: { include: ['buffer'] },
  server: { proxy: { '/aura-api': { target: 'http://127.0.0.1:8765', changeOrigin: true, rewrite: (path) => path.replace(/^\/aura-api/, '') } } },
})
