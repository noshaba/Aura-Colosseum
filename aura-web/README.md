# Aura web app

From the repository root:

- Live UI: `cd aura-web && npm install && npm run dev`
- GPU-free Judge Mode: `cd aura-web && npm install && npm run judge`
- Production build: `npm run build`

The live app proxies `/aura-api` to `http://127.0.0.1:8765` using `vite.config.ts`.
