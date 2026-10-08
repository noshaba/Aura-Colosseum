#!/usr/bin/env bash
set -Eeuo pipefail

# Aura streamlined live launcher.
# Text2Motion Aura is used as a Python generation engine; the Viser/Text2Motion Aura demo UI is not started.
# Run after activating the Python environment where Aura dependencies are installed.

if [[ -n "${AURA_ROOT:-}" ]]; then
  ROOT="$(cd "$AURA_ROOT" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -d "$SCRIPT_DIR/../text2motion-aura" && -d "$SCRIPT_DIR/../aura-web" ]]; then
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  elif [[ -d "$PWD/text2motion-aura" && -d "$PWD/aura-web" ]]; then
    ROOT="$PWD"
  else
    echo "ERROR: Could not find project root (text2motion-aura/ + aura-web/)." >&2
    exit 1
  fi
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: Activate your Aura Python/Conda environment first." >&2
  exit 1
fi

for cmd in python npm; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: '$cmd' is not available in the active environment/PATH." >&2
    exit 1
  fi
done

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/text2motion-aura${PYTHONPATH:+:$PYTHONPATH}"
export TEXT_ENCODER_DEVICE=cpu
export TEXT_ENCODER_MODE=api
export AURA_G1_FINISH="${AURA_G1_FINISH:-all-gold}"
export AURA_MOTION_LIBRARY="${AURA_MOTION_LIBRARY:-$ROOT/aura-motion-library}"
export AURA_TEXT2MOTION_MODEL="${AURA_TEXT2MOTION_MODEL:-g1-rp}"
export AURA_TEXT2MOTION_STEPS="${AURA_TEXT2MOTION_STEPS:-30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset VITE_AURA_JUDGE_MODE || true

TEXT_PORT=9550
LIBRARY_PORT=8765
AURA_PORT=5173
LOG_DIR="$ROOT/.aura-logs"
mkdir -p "$LOG_DIR" "$AURA_MOTION_LIBRARY"

port_open() {
  python - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
s = socket.socket(); s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
else:
    s.close(); raise SystemExit(0)
PY
}

wait_port() {
  local name="$1" port="$2" timeout="$3"
  local start now
  start=$(date +%s)
  printf 'Waiting for %s on port %s' "$name" "$port"
  while ! port_open "$port"; do
    sleep 2; printf '.'; now=$(date +%s)
    if (( now - start >= timeout )); then
      printf '\nERROR: %s did not become ready within %ss.\n' "$name" "$timeout" >&2
      return 1
    fi
  done
  printf ' ready.\n'
}

for p in "$TEXT_PORT" "$LIBRARY_PORT" "$AURA_PORT"; do
  if port_open "$p"; then
    echo "ERROR: Port $p is already in use. Stop the old Aura process first." >&2
    exit 1
  fi
done

# Install frontend dependencies on a freshly extracted project.
if [[ ! -x "$ROOT/aura-web/node_modules/.bin/vite" ]]; then
  echo "Frontend dependencies are missing; installing them once..."
  (
    cd "$ROOT/aura-web"
    if [[ -f package-lock.json ]]; then npm ci; else npm install; fi
  )
fi

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Stopping Aura services..."
  for ((i=${#PIDS[@]}-1; i>=0; i--)); do kill "${PIDS[$i]}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT"
echo "Project: $ROOT"
echo "Conda env: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
echo "Text encoder: CPU"
echo "Text2Motion Aura G1 generator: loaded lazily on GPU by Aura API"
echo "Text2Motion Aura denoising steps: $AURA_TEXT2MOTION_STEPS"
echo

# 1) Heavy text encoder stays on CPU.
(
  cd "$ROOT/text2motion-aura"
  exec python -m text2motion_aura.scripts.run_text_encoder_server
) >"$LOG_DIR/textencoder.log" 2>&1 &
PIDS+=("$!")
if ! wait_port "Text2Motion Aura text encoder" "$TEXT_PORT" 600; then
  tail -n 100 "$LOG_DIR/textencoder.log" || true; exit 1
fi

# 2) Aura API owns the persistent Text2Motion Aura G1 generation model.
(
  cd "$ROOT"
  exec python aura_library_server.py
) >"$LOG_DIR/library.log" 2>&1 &
PIDS+=("$!")
if ! wait_port "Aura generation/library API" "$LIBRARY_PORT" 60; then
  tail -n 120 "$LOG_DIR/library.log" || true; exit 1
fi

# 3) Full Aura frontend.
(
  cd "$ROOT/aura-web"
  exec npm run dev -- --host 127.0.0.1 --port "$AURA_PORT" --strictPort
) >"$LOG_DIR/aura-web.log" 2>&1 &
PIDS+=("$!")
if ! wait_port "Aura web app" "$AURA_PORT" 120; then
  tail -n 120 "$LOG_DIR/aura-web.log" || true; exit 1
fi

echo
echo "Aura is ready:"
echo "  Full Aura UI: http://localhost:$AURA_PORT"
echo "  Aura API:     http://localhost:$LIBRARY_PORT"
echo
echo "The old Text2Motion Aura viewer is intentionally not started."
echo "The first Generate click may take longer because the G1 model is loaded lazily."
echo "Logs: $LOG_DIR"
echo "Press Ctrl+C to stop all services."
echo

wait -n "${PIDS[@]}"
status=$?
echo "A service exited (status $status). Check logs in $LOG_DIR." >&2
exit "$status"
