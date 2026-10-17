#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# train_rl_ppo_20260914.py 会为本次PPO会话强制使用10 Hz压力采样，并加载 pressure_drift_10hz.json。
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${1:-}" == "--foreground" ]]; then
  shift
  exec "$PYTHON_BIN" scripts/train_rl_ppo_20260914.py "$@"
fi
OUT_DIR="$ROOT/logs/rl_20260914"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/launcher_$(date +%Y%m%d_%H%M%S).out"
nohup "$PYTHON_BIN" scripts/train_rl_ppo_20260914.py "$@" >"$OUT" 2>&1 < /dev/null &
PID=$!
echo "PID=$PID"
echo "training stdout=$OUT"
echo "session directory will be printed by the training process"
