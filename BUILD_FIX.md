# Frontend build fix

This patch fixes the TypeScript errors reported by `npm run build`:

- passes `onRefresh` into `WalletModal`;
- safely narrows `SkeletonHelper.material` before setting its color;
- uses the browser-compatible `buffer` package for Solana instruction data, matching `@solana/web3.js` typing/runtime expectations;
- adds `buffer@6.0.3` as a direct dependency.

After extracting, run:

```bash
cd aura-web
npm install
npm run build
npm run build:judge
```

Keep the generated `package-lock.json` in the submission repository.
