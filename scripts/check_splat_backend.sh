#!/usr/bin/env bash
set -u
ENV_NAME="${AURA_SPLAT_NERFSTUDIO_ENV:-aura-splat}"
if ! command -v conda >/dev/null 2>&1; then
  echo "FAIL: conda is not on PATH"
  exit 1
fi

echo "Checking Gaussian reconstruction backend in conda env: $ENV_NAME"
failed=0
for cmd in ns-process-data ns-train ns-export ffmpeg colmap; do
  printf '%-18s ' "$cmd"
  if conda run --no-capture-output -n "$ENV_NAME" "$cmd" --help >/dev/null 2>&1 || \
     { [[ "$cmd" == "ffmpeg" ]] && conda run --no-capture-output -n "$ENV_NAME" ffmpeg -version >/dev/null 2>&1; }; then
    echo "OK"
  else
    echo "MISSING"
    failed=1
  fi
done

if [[ $failed -eq 0 ]]; then
  echo
  echo "Backend ready. Start Aura with:"
  echo "  export AURA_SPLAT_NERFSTUDIO_ENV=$ENV_NAME"
  echo "  ./scripts/run_aura_live.sh"
else
  echo
  echo "Backend incomplete. See SPLAT_BACKEND_SETUP.md in this patch."
fi
exit $failed
