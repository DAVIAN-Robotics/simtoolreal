#!/usr/bin/env bash
# Run in an allocated GPU environment. This script never submits a job.
set -euo pipefail
TASK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TREE=$(cd -- "$TASK_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$TREE/.venv_isaacsim/bin/python"}
if [ "$#" -lt 1 ]; then
  echo "Usage: PYTHON_BIN=/path/to/python $0 /absolute/fresh-run-directory [train_screw.py arguments]" >&2
  exit 2
fi
WORK=$1
shift
export PYTHONPATH="$TREE/adapters:$TREE/rl_games:$TREE${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK=$("$PYTHON_BIN" "$TASK_DIR/prepare_run.py" --work "$WORK")
export WANDB_DIR="$WORK"
SSL_CERT_FILE=$("$PYTHON_BIN" -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
"$PYTHON_BIN" "$TREE/adapters/check_contact_loss.py"
"$PYTHON_BIN" "$TREE/adapters/check_kernels.py"
"$PYTHON_BIN" -u "$TREE/adapters/train_screw.py" --work "$WORK" --num-envs 96 --smoke --headless
# Isaac can return zero after a Python exception: require the fresh passing report.
"$PYTHON_BIN" - "$WORK" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / 'rl_smoke.json'
assert p.is_file(), 'Simulator did not produce a passing smoke report'
s = json.loads(p.read_text())
assert s['done_path_119_vs_120_steps'] and s['contact_loss_policy_steps'] == 120
assert s['actual_hand_object_contact_detected'] and s['immediate_reach_termination']
assert s['all_four_legs_and_eight_starting_stages'] and s['action_dim'] == 12
PY
exec "$PYTHON_BIN" -u "$TREE/adapters/train_screw.py" --work "$WORK" --num-envs 24576 --epochs 1000000 --headless "$@"
