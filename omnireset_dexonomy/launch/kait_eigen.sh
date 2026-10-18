#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --output=/home/nas_main/sibeenkim/draftrec/_055-eigen/slurm/%x_%j.out
# One grasped-start EigenDEXplore run on a kait worker, from scratch, in the unidexgrasp-reset-eigen tree whose own
# patched rl_games goes first on PYTHONPATH. CUR (global or cpo) and NOISE (joint, egosuite or dexonomy) come from the
# submitting environment, and the run name is built from them.
#   CUR=<global|cpo> NOISE=<joint|egosuite|dexonomy> sbatch -p compute -A sub -q sub-own -J <name> kait_eigen.sh
# The workers have no git, so check the tree hash (046ea5d) on the login node before submitting.
set -euo pipefail

ROOT=/home/nas_main/sibeenkim/draftrec
TREE=$ROOT/repos/simtoolreal-unidexgrasp-reset-eigen
PY=$ROOT/venv_isaacsim/bin/python
NAME=unidexgrasp-reset_dexonomy
TASK=""
if [ "$CUR" = cpo ]; then
  NAME=$NAME-cur_per_obj
  TASK="env.termination.tolerance_curriculum_per_object=True"
elif [ "$CUR" != global ]; then
  echo "CUR is global or cpo"; exit 1
fi
EIGEN=""
if [ "$NOISE" = egosuite ] || [ "$NOISE" = dexonomy ]; then
  EIGEN="--eigendexplore_basis $ROOT/_055-eigen/eigen_${NOISE}_xhand.npz"
  NAME=$NAME-eigen_$NOISE
elif [ "$NOISE" != joint ]; then
  echo "NOISE is joint, egosuite or dexonomy"; exit 1
fi
test "$NAME" = "$SLURM_JOB_NAME" || { echo "job name $SLURM_JOB_NAME is not the run name $NAME"; exit 1; }

export PYTHONPATH=$TREE:$TREE/rl_games
export LD_LIBRARY_PATH=$ROOT/syslibs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TMPDIR=/tmp/sibeenkim-isaaclab-${SLURM_JOB_ID}
export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"

# The worker image carries no CA store, so point TLS at certifi or wandb cannot connect.
SSL_CERT_FILE=$("$PY" -c "import certifi; print(certifi.where())")
export SSL_CERT_FILE
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export WANDB_DIR=$TREE/runs

cd "$TREE"
mkdir -p runs
test ! -e "runs/$NAME" || { echo "run dir runs/$NAME exists"; exit 1; }
echo "tree $TREE  run $NAME  rl_games $("$PY" -c 'import rl_games; print(rl_games.__file__)')  ${EIGEN:-joint noise}  $TASK"

# Slurm sets CUDA_VISIBLE_DEVICES for the allocated GPU. Never set it here.
"$PY" -u isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --wandb_activate --wandb_project simtoolreal --wandb_entity draftrec \
  --wandb_name "$NAME" $EIGEN \
  hydra.run.dir="runs/$NAME" \
  agent.params.config.save_frequency=1000 \
  env.robot=fr3-xhand-a4hs \
  env.assets.object_pool_dir=$ROOT/assets_040/objects_cap16 \
  env.assets.asset_cache_dir=$ROOT/assets_040/usd_cache_objects_cap16 \
  env.reset.reset_state_dir=$ROOT/_graspoc/cap16/reset_full/isaac/final \
  $TASK
