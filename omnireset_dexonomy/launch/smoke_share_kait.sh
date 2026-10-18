#!/usr/bin/env bash
# Smoke of the shared branch unidexgrasp_objects-omnireset_dexonomy-eigendexplore_dexonomy on a kait login-node B200.
# 16 objects, 2048 envs, no wandb, 40 epochs, the tolerance curriculum forced to tighten every 100 frames. Every
# variant has the settings of the run the branch comes from (grasped start, per-object curriculum, EigenDEXplore with
# the Dexonomy basis), and each variant but base adds one of the five merged features.
#   base      the run's settings only
#   gog       goals are reset states with their fingertips, plus the viewer capture
#   wrist     gog with the goal window on the palm pose
#   both      gog with the palm window and success on the object and the fingertips together
#   bps       the 32-number BPS shape code observation
#   adaptive  grasp-type adaptive sampling, updated every 200 frames
#   GPU=<gpu> VARIANT=<base|gog|wrist|both|bps|adaptive> bash smoke_share_kait.sh
set -euo pipefail
ROOT=/home/nas_main/sibeenkim/draftrec
TREE=$ROOT/repos/simtoolreal-unidexgrasp_objects-omnireset_dexonomy-eigendexplore_dexonomy
PY=$ROOT/venv_isaacsim/bin/python
GPU=${GPU:?set GPU}
VARIANT=${VARIANT:?set VARIANT}
NAME=smoke_share_${VARIANT}_$(date +%Y%m%d_%H%M%S)

OBS_GOAL="[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,fingertip_pos_rel_goal,object_scales]"
STATE_GOAL="[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,palm_vel,object_rot,object_vel,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,fingertip_pos_rel_goal,object_scales,closest_keypoint_max_dist,closest_fingertip_dist,closest_fingertip_max_dist,lifted_object,progress,successes,reward]"
OBS_BPS="[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,shape_code]"
STATE_BPS="[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,palm_vel,object_rot,object_vel,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,shape_code,closest_keypoint_max_dist,closest_fingertip_dist,lifted_object,progress,successes,reward]"
GOAL="env.reset.goal_sampling_type=reset_state env.reset.goal_fingertip_dir=$ROOT/_052/gog_check/ft16 env.obs.obs_list=$OBS_GOAL env.obs.state_list=$STATE_GOAL"
VIEW=""
case $VARIANT in
  base) EXTRA="" ;;
  gog) EXTRA="$GOAL env.reward.fingertip_goal_bonus=1000.0"
       VIEW="--capture_viewer --capture_viewer_interval 300 --capture_viewer_len 60" ;;
  wrist) EXTRA="$GOAL env.reward.fingertip_goal_bonus=1000.0 env.reset.goal_wrist_dir=$ROOT/_052/gog_check/palm16" ;;
  both) EXTRA="$GOAL env.reward.fingertip_goal_bonus=0.0 env.reset.goal_wrist_dir=$ROOT/_052/gog_check/palm16 env.termination.success_with_fingertips=True" ;;
  bps) EXTRA="env.assets.shape_code_file=$ROOT/_052/shape_compare/bps_32.npz env.obs.obs_list=$OBS_BPS env.obs.state_list=$STATE_BPS" ;;
  adaptive) EXTRA="env.reset.type_adaptive_sampling=True env.reset.grasp_type_file=$ROOT/_052/grasp_types.npz env.reset.type_adaptive_interval=200" ;;
  *) echo "VARIANT is base, gog, wrist, both, bps or adaptive"; exit 1 ;;
esac

export PYTHONPATH=$TREE:$TREE/rl_games
export LD_LIBRARY_PATH=$ROOT/syslibs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TMPDIR=/tmp/sibeenkim-share-$VARIANT
export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$TMPDIR"
cd "$TREE"
mkdir -p runs
echo "run $NAME tree $(git log --oneline -1 | cut -c1-60) rl_games $("$PY" -c 'import rl_games; print(rl_games.__file__)') extra $EXTRA $VIEW"
"$PY" -u isaacsimenvs/train.py --task Isaacsimenvs-SimToolReal-Direct-v0 --agent rl_games_sapg_cfg_entry_point \
  --headless $VIEW --eigendexplore_basis $ROOT/_055-eigen/eigen_dexonomy_xhand.npz \
  hydra.run.dir=runs/$NAME env.robot=fr3-xhand-a4hs \
  env.assets.object_pool_dir=$ROOT/_052/pool_smoke16 env.assets.asset_cache_dir=$ROOT/assets_040/usd_cache_objects_cap16 \
  env.reset.reset_state_dir=$ROOT/_graspoc/cap16/reset_full/isaac/final \
  env.termination.tolerance_curriculum_per_object=True \
  env.termination.tolerance_curriculum_success_threshold=0.0 env.termination.tolerance_curriculum_interval=100 \
  env.scene.num_envs=2048 agent.params.config.minibatch_size=8192 \
  agent.params.config.central_value_config.minibatch_size=8192 agent.params.config.expl_coef_block_size=512 \
  agent.params.config.save_frequency=20 agent.params.config.max_epochs=40 \
  $EXTRA
