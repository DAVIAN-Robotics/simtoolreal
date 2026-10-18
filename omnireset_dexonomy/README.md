# OmniReset + Dexonomy on SimToolReal, FR3 with XHand

This branch holds the code behind the SimToolReal runs that start each episode from a grasped object. The grasps come from Dexonomy, run for the XHand. An OmniReset-style pipeline turns them into reset states for the FR3 with the XHand. Training explores with EigenDEXplore.

The training code is the exact tree of the run `unidexgrasp-reset_dexonomy-cur_per_obj-eigen_dexonomy` ([wandb](https://forge.coreweave.com/wandb/draftrec/simtoolreal/runs/unidexgrasp-reset_dexonomy-cur_per_obj-eigen_dexonomy_20261009_214101)), tree hash 046ea5d at commit e698df9. Five features from sibling runs sit on top of it. Each is behind its own flag and off by default, so the run's own command behaves as it did.

The work has three stages.

| Stage | Folder | Output |
| --- | --- | --- |
| 1. Dexonomy grasps for the XHand on 3,200 objects | `omnireset_dexonomy/dexonomy/` | 12,899,466 grasps |
| 2. Reset states, the grasps placed in the scene with FR3 IK and tested in Isaac | `omnireset_dexonomy/reset/` | 10,000 states per object, 32,000,000 total |
| 3. Training from those states | `isaacsimenvs/` in this repo, plus `omnireset_dexonomy/data/` and `omnireset_dexonomy/launch/` | policies |

Every data path below is on our NAS. Nothing large is in git. The objects are the 3,200 cap16 objects, UniDexGrasp objects with CoACD collision capped at 16 pieces.

## Stage 1. Dexonomy grasps

**Upstream.** [JYChen18/Dexonomy](https://github.com/JYChen18/Dexonomy) at commit d561ff9, licensed CC BY-NC 4.0. This folder carries only our files on top of it, not a copy of their code.

**What is here.**
- `xhand_port/` mirrors paths inside the clone. Copy `xhand_port/assets/hand/xhand/` and `xhand_port/dexonomy/config/hand/xhand.yaml` into the clone. The port holds the XHand MJCF, hulls, meshes and astra's 31 rules-corrected grasp templates. Types 19 (Distal Type) and 21 (Tripod Variation) are left out on purpose.
- `dexonomy_d561ff9_clone.diff` is `git diff` of the clone exactly as the run used it. It is the 8 patch scripts in `src/agent4_cap16_patch_*.py` plus one edit made by hand. Apply it with `git apply` on d561ff9.
- `DISTILL-dexonomy-large-scale.md` is the full redo guide, written so an agent with no memory of the run can redo it. Read it before running anything here. Its paths are relative to this folder. `status/decisions/agent4.json` is the register of every decision it cites as A4-Dnn.
- `src/` and `scripts/` are the run machinery as it ran, the object prep, the multi-host scheduler, the shared workers and the delivery check. Host addresses are replaced by `<nas4 host>` and `<yj host>`.

**Result.** 40 epochs on 48 GPUs took 16h 54m. Refinement kept 16,820,193 of 39,680,000 starts, and Dexonomy's own MuJoCo test passed 12,899,466 of those. Every object has grasps, with a minimum of 79, a median of 3,983 and a maximum of 7,703.

**Format.** One `<stem>.npz` per object with `hand_pos` (N, 3), `hand_quat` (N, 4) and `q_pregrasp`, `q_grasp`, `q_squeeze` (N, 12) in the XHand joint order. `extra/<stem>.npz` has `tmpl`, the template name of each grasp, and `src`.

**Where.**
- dj `/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16_deliver/dexonomy/`, main files plus `extra/`.
- kait `/home/nas_main/sibeenkim/draftrec/_graspoc/cap16/reset_full/grasps/`, main files only, md5-equal to dj.

## Stage 2. Reset states

**What OmniReset-style means here.** Per object, the pipeline follows five steps.
1. It draws 100,000 object poses, uniform in SimToolReal's goal box with uniform orientation. A pose whose lowest point is below the table top is redrawn.
2. It pairs each pose with one random Dexonomy grasp, as OmniReset draws one grasp per attempt. Pairs whose hand dips below the table are dropped.
3. It solves FR3 IK with cuRobo, keeping the best of 32 seeds, inside Franka's suggested rectangular joint limits.
4. It tests the reached states in the SimToolReal scene in Isaac Sim, with OmniReset's check for a grasped object.
5. It keeps the first 10,000 states that pass.

**Result.** All 3,200 objects have 10,000 states. The Isaac pass rate per object runs from 33.2% to 96.8%, with a median of 87.9%.

**Format.** One `<stem>.npz` per object with `obj_pos` (10000, 3), `obj_quat_wxyz` (10000, 4), `arm_q` (10000, 7), `hand_q` (10000, 12), `hand_target` (10000, 12), `grasp` (the row of the Stage 1 file), `arm_joints`, `hand_joints`, `tested` and `passed`. `hand_q` is the grasp pose. `hand_target` is the squeeze target at Dexonomy's hand stiffness 3.0, which training rescales to the robot's own stiffness.

**Where.**
- yj `/home/nas5/minhopark/_053/reset_full_final/`
- kait `/home/nas_main/sibeenkim/draftrec/_graspoc/cap16/reset_full/isaac/final/`
- dj `/home/nas4_user/sibeenkim/work/_graspoc/reset/isaac_all/final/`

**Scripts.**
- **IK.** `src/reset_ik_all.py` is the full-set IK, one worker per GPU on dj. `reset_ik_omni.py` is the same scheme on one object. `reset_sample.py`, `reset_ik.py`, `reset_arm_object.py` and `reset_merge.py` are the earlier step-by-step version.
- **Isaac test.** `src/reset_isaac.py` runs it, `scripts/reset_isaac_all2.sh` is its worker loop, and `src/reset_isaac_check.py` checks the finished set. `reset_isaac_merge.py` pools one object's batches.
- **Shared code and figures.** `src/reset_common.py` holds the shared geometry. `reset_render.py`, `reset_html.py`, `reset_figs_*.py` and `reset_ik_status.py` make figures and pages.

**Environments.**
- The IK ran in a cuRobo 0.8.0 venv, kait `/home/nas_main/sibeenkim/draftrec/repos/curobo2/.venv`.
- The Isaac test ran with the `simtoolreal-050` tree on PYTHONPATH, kait `/home/nas_main/sibeenkim/draftrec/repos/simtoolreal-050`.
- `reset_common.py` checks its scene numbers against that tree's source, so another tree fails loud.

**Known gap.** The Isaac test used robot `fr3-xhand`, the hand before the real-XHand system identification, with stiffness 3.0 and damping 0.1. Training uses `fr3-xhand-a4hs`. No state was re-tested on the a4hs hand.

## Stage 3. Training

**PYTHONPATH must be `$TREE:$TREE/rl_games`.** The patched rl_games is inside this repo. Without it an installed rl_games is imported instead, and the EigenDEXplore and checkpoint patches are absent.

**The source run.** This is its command with yj paths. Use your own interpreter and your own TMPDIR, since Isaac Lab writes its logs under `gettempdir()/isaaclab`.

```bash
TREE=<this checkout>
export PYTHONPATH=$TREE:$TREE/rl_games OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=<your tmp dir>
cd $TREE
CUDA_VISIBLE_DEVICES=<gpu> python -u isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 --agent rl_games_sapg_cfg_entry_point --headless \
  --wandb_activate --wandb_project simtoolreal --wandb_entity draftrec \
  --wandb_name unidexgrasp-reset_dexonomy-cur_per_obj-eigen_dexonomy \
  --eigendexplore_basis /home/nas5/minhopark/_053/eigen/eigen_dexonomy_xhand.npz \
  hydra.run.dir=runs/unidexgrasp-reset_dexonomy-cur_per_obj-eigen_dexonomy \
  agent.params.config.save_frequency=1000 \
  env.robot=fr3-xhand-a4hs \
  env.assets.object_pool_dir=/home/nas5/minhopark/assets_039/objects_cap16 \
  env.assets.asset_cache_dir=/home/nas5/minhopark/assets_039/usd_cache_objects_cap16 \
  env.reset.reset_state_dir=/home/nas5/minhopark/_053/reset_full_final \
  env.termination.tolerance_curriculum_per_object=True
```

At the default 24,576 envs the 3,200-object scene takes about 32 to 41 minutes to spawn. On an RTX A6000 the run holds about 45 to 48 GB. `launch/launch_eigen_yj.sh` and `launch/kait_eigen.sh` are the launchers the runs used on yj and kait Slurm.

**Flags.** Everything here is off unless set.

| Flag | What it does | Data | Builder |
| --- | --- | --- | --- |
| `env.reset.reset_state_dir` | Episodes start from a random reset state of the object, lifted, with the start pose as the first goal. They end when no hand link touches the object. | Stage 2 dir | `reset/` |
| `env.reset.reset_state_count`, `env.reset.first_goal_random` | Use only the first N states, and draw the first goal like every later one | | |
| `env.termination.tolerance_curriculum_per_object` | The success tolerance tightens per object instead of globally. The curriculum is saved in the checkpoint and restored on resume. | | |
| `--eigendexplore_basis <npz>` | EigenDEXplore exploration noise with that hand basis, hgupt3/eigendexplore 2926933 | yj `/home/nas5/minhopark/_053/eigen/eigen_{dexonomy,egosuite}_xhand.npz` | `data/make_dexonomy_basis.py` for Dexonomy. EgoSuite is their release basis. |
| `env.reset.goal_sampling_type=reset_state` with `env.reset.goal_fingertip_dir` | Every goal is a reset state, its object pose and its 5 fingertip pad points. Add `fingertip_pos_rel_goal` to the obs lists and set `env.reward.fingertip_goal_bonus`. | yj `/home/nas5/minhopark/_053/goal_fingertips` | `data/goal_fingertips.py` |
| `env.reset.goal_wrist_dir` | With reset-state goals, the goal window is measured on the palm pose instead of the object pose | yj `/home/nas5/minhopark/_053/goal_palm` | `data/goal_palm.py` |
| `env.termination.success_with_fingertips` | With reset-state goals, success also needs every fingertip at the goal state's fingertips | | |
| `env.assets.shape_code_file` | A fixed 32-number BPS shape code per object. Add `shape_code` to the obs lists. | yj `/home/nas5/minhopark/_053/shape_code/bps_32.npz` | `data/shape_code_compare.py` then `data/make_bps32.py` |
| `env.reset.type_adaptive_sampling` with `env.reset.grasp_type_file` | Starts are drawn per grasp type with weight clip(mean / type mean) of successes per episode, updated every `type_adaptive_interval` frames | kait `/home/nas_main/sibeenkim/draftrec/_052/grasp_types.npz`, not on yj | `data/make_grasp_types.py`, from Stage 1 `extra/` |

`data/make_grasp_types.py` was written for this branch. Its output equals the kait file in all 3,201 arrays.

**Obs lists the sibling runs used.** Pass them as `env.obs.obs_list=` and `env.obs.state_list=`.

```text
reset-state goals
  obs   [joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,fingertip_pos_rel_goal,object_scales]
  state [joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,palm_vel,object_rot,object_vel,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,fingertip_pos_rel_goal,object_scales,closest_keypoint_max_dist,closest_fingertip_dist,closest_fingertip_max_dist,lifted_object,progress,successes,reward]
shape code
  obs   [joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,shape_code]
  state [joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,palm_vel,object_rot,object_vel,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,shape_code,closest_keypoint_max_dist,closest_fingertip_dist,lifted_object,progress,successes,reward]
```

**Sibling runs on wandb, project draftrec/simtoolreal.** They ran before EigenDEXplore was added.
- `unidexgrasp-coacd_16-tactile_none-reset_dexonomy-curriculum_per_object-goal_object_grasp`
- `unidexgrasp-coacd_16-tactile_none-reset_dexonomy-curriculum_per_object-goal_object_grasp-goal_sampling_wrist`
- `unidexgrasp-coacd_16-tactile_none-reset_dexonomy-cur_per_object-goal_object_grasp_both-goal_sampling_wrist`
- `unidexgrasp-coacd_16-reset_dexonomy-cur_per_object-bps_32`
- `unidexgrasp-coacd_16-tactile_none-reset_dexonomy_adaptive-curriculum_per_object`

## Smoke test of this branch

`launch/smoke_share_kait.sh` ran six variants on kait B200s on 2026-10-18. Each had 16 objects, 2,048 envs, 40 epochs, the source run's settings, and one feature turned on.

| Variant | Epochs | Tracebacks | Obs / state size | Feature signal |
| --- | --- | --- | --- | --- |
| base | 40 / 40 | 0 | 134 / 156 | none, the run's settings |
| reset-state goals | 40 / 40 | 0 | 149 / 172 | fingertip goal bonus paid, 956.5 per episode at the last log |
| plus palm window | 40 / 40 | 0 | 149 / 172 | fingertip goal bonus paid, 955.4 per episode at the last log |
| plus success with fingertips | 40 / 40 | 0 | 149 / 172 | bonus set to 0 as configured, no separate signal for the success rule in 40 epochs |
| shape code | 40 / 40 | 0 | 166 / 188 | 32 more inputs |
| grasp-type adaptive | 40 / 40 | 0 | 134 / 156 | 93 per-type tags logged, 16 weight updates |

EigenDEXplore loaded the Dexonomy basis in all six. The repeated `[pose_viewer] get_student_obs failed` line also fills the source run's log and is harmless.

## Files in this folder

| Path | What |
| --- | --- |
| `dexonomy/DISTILL-dexonomy-large-scale.md` | Stage 1 redo guide |
| `dexonomy/dexonomy_d561ff9_clone.diff` | the patched upstream clone as one diff |
| `dexonomy/xhand_port/` | XHand MJCF and the 31 templates, at their clone paths |
| `dexonomy/src/`, `dexonomy/scripts/` | Stage 1 machinery, patch scripts and template package |
| `dexonomy/status/` | decisions register and template checksums |
| `reset/src/`, `reset/scripts/` | Stage 2 |
| `data/` | builders for the training inputs above |
| `launch/` | the launchers and the smoke script |
