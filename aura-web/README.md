# Aura web app

From the repository root:

- Live UI: `cd aura-web && npm install && npm run dev`
- GPU-free Judge Mode: `cd aura-web && npm install && npm run judge`
- Production build: `npm run build`

The live app proxies `/aura-api` to `http://127.0.0.1:8765` using `vite.config.ts`.

## Netlify

`netlify.toml` at the repository root sets `base = aura-web`, builds Judge Mode (`npm run build:judge`) and publishes `dist`. Judge Mode is static and does not need the local Aura API. For a live deployment, switch the command to `npm run build` and enable the `/aura-api/*` redirect to a hosted backend.
