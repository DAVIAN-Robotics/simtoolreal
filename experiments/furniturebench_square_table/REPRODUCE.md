# FurnitureBench square-table reproduction map

This file identifies the training code, checkpoints, inputs, runtimes, upstream generation data, and later evaluation results for [furniturebench-square_table-contact_termination_2s_20261017_062847](https://forge.coreweave.com/wandb/draftrec/simtoolreal/runs/furniturebench-square_table-contact_termination_2s_20261017_062847). Locations were checked on **2026-10-19**. Large files stay on the lab servers; no trained checkpoint is uploaded to Git.

## 1. Recommended checkpoint: best at 2.58B global steps

**Use this preserved copy on KAIT:**

```text
/home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_reproduction_20261019/checkpoints/best-step-2584608768.pth
```

| Field | Value |
|---|---|
| Global environment steps | **2,584,608,768** |
| Size | 408,188,001 bytes |
| SHA-256 | `baa40c076ca5519912bcbbd376b76d973c44e2cf924be7d4b2a506d730c64c15` |
| Original save time | 2026-10-18 12:36:30 UTC |
| Copy verified | 2026-10-19 05:16:31 UTC |
| Selection | Best by training reward |

Training performance subsequently declined. The read-only investigation found that changing SGS sampling alone did not explain the decline; the cause was not established. We therefore recommend this best checkpoint over later `last/model.pth`. “Best” here is the trainer's reward criterion, not a separate evaluation ranking by full-sequence success.

The original source is:

```text
/home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_contact_termination_2s_20261017/training/0_furniturebench-square_table-contact_termination_2s/best/model.pth
```

That path is a symlink to a file in `nn/` and can change if training finds another best. The preserved copy above is the reproducible pointer. Its [metadata](provenance/best-checkpoint.json) is committed here and also saved beside the checkpoint as `../best-checkpoint.json`.

For comparison, the live `last/model.pth` read during this handoff was at **3,932,160,000** global steps. It is not the recommended checkpoint. The `last/`, `best/`, and W&B “latest” pointers are mutable.

Download using your KAIT SSH configuration, for example from your workstation:

```bash
scp kait:/home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_reproduction_20261019/checkpoints/best-step-2584608768.pth .
sha256sum best-step-2584608768.pth
```

The saved file is a multi-policy bundle. The policy state is under integer key `0`, and its `frame` field contains the global environment-step count. It also includes normalization, optimizer/critic state, and saved SGS state. To inspect it with the recorded PyTorch environment:

```python
import torch
bundle = torch.load("best-step-2584608768.pth", map_location="cpu", weights_only=False)
state = bundle[0]
assert int(state["frame"]) == 2_584_608_768
print("global steps:", state["frame"])
print("SGS state present:", "sgs" in state["env_state"])
```

## 2. Code and training inputs

Branch: `furniturebench_square_table-omnireset_dexonomy-eigendexplore_dexonomy-sgs` in `DAVIAN-Robotics/simtoolreal`.

The training implementation was published in commit `266ca1bc71a3294e36a5a7a5490e67e453bdeea7`. Subsequent handoff commits add documentation and validation evidence. The original parent was `c4d45799eccfa4d8ef6408e829f8a490c8bd4937`. [manifest.json](provenance/manifest.json) records every packaged source/input/asset hash; [portability.patch](provenance/portability.patch) identifies the two publication edits for paths and run identity. Training behavior was retained.

Everything needed to train this setup from scratch is in the branch except the Isaac/PyTorch runtime, GPU resources, and W&B authentication. Neither a pretrained policy nor upstream grasp regeneration is needed.

| Training input | Branch location | Role |
|---|---|---|
| 636,023 accepted reset rows | [inputs/screw_resets.npz](inputs/screw_resets.npz) | Four legs, eight screw-progress stages, accepted arm configurations and grasp states |
| Goal sequences and scene | [inputs/screw_goals.json](inputs/screw_goals.json) | Per-leg engagement through fully screwed poses; fixture and robot setup |
| Reset extraction provenance | [inputs/reset_preparation.json](inputs/reset_preparation.json) | Event files, row counts, selection, source hashes |
| Base environment settings | [inputs/september_config.yaml](inputs/september_config.yaml) | Base mechanics and physics; entry point applies recorded overrides |
| Actor/critic and trainer settings | [inputs/training_template.yaml](inputs/training_template.yaml) | Network, observation fields, PPO/SAPG configuration |
| Finger exploration PCA | [inputs/eigen_dexonomy_xhand.npz](inputs/eigen_dexonomy_xhand.npz) | EigenDEXplore basis; distinct from task-specific reset grasps |
| Prepared leg and open four-socket fixture | [task/inputs/assets/](task/inputs/assets/) | The exact `body3x` leg/thread geometry used in this run |
| Robot and registry assets | [assets/urdf/](../../assets/urdf/) | Exact packaged meshes and URDFs |

Reset-pool SHA-256: `34ae6703409f721c8de8e999b44689dc80c2384060e4296426bcf7f6e52a9dc0`.

The reset rows are **169,683 / 122,620 / 145,486 / 198,234** for legs 1/2/3/4. They are grasped engagement/intermediate states, not approach or pregrasp states. Saved poses use xyz + quaternion xyzw. Training starts from these states with one active leg, all four sockets open, and the arm held at the saved seven-joint configuration. The actor learns 12 finger actions. [README.md](README.md) gives the full setup, actual success semantics, SGS, termination, and validation.

## 3. Server locations and runtimes

`kait`, `kaist140`, and `yj-3-A6000` below are the SSH aliases used for verification. Use your own lab account/configuration. NAS paths are server-specific: do not replace `/home/nas_main`, `/home/nas4_user`, `/home/nas2`, or `/home/nas5` by assumption.

| Site | Purpose and verified location |
|---|---|
| KAIT | Original training root: `/home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_contact_termination_2s_20261017` |
| KAIT | Preserved recommended checkpoint: `/home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_reproduction_20261019` |
| KAIT | Isaac runtime: `/home/nas_main/sibeenkim/draftrec/venv_isaacsim/bin/python` |
| KAIT | Runtime library directory: `/home/nas_main/sibeenkim/draftrec/syslibs/usr/lib/x86_64-linux-gnu` |
| DJ, checked on `kaist140` | Isaac runtime: `/home/nas4_user/sibeenkim/work/_033_venv_isaacsim/bin/python` |
| DJ | CPU planning runtime: `/home/nas4_user/sibeenkim/work/_dexagent_stage1_003_e2e10k/cpu_env/bin/python` |
| DJ | Frozen evaluation code/config/checkpoint: `/home/nas4_user/sibeenkim/work/_dexagent_table_rl_eval_20261018` |
| DJ | Large evaluation recordings: `/home/nas2/sibeenkim/_dexagent_table_rl_eval_20261018` |
| DJ | Publication smoke-test checkout and outputs: `/home/nas4_user/sibeenkim/work/_furniturebench_square_table_publish_20261019` |
| YJ, checked on `yj-3-A6000` | Native grasp/event generation shards: `/home/nas5/sibeenkim/work/_dexagent_stage1_003_fresh3_20261015` |

The recorded Isaac runtime is Python **3.11.15**, Isaac Sim **5.1.0.0**, Isaac Lab **2.3.2.post1**, PyTorch **2.7.0+cu128**, NumPy **1.26.0**, Warp **1.14.0**, and W&B **0.27.2**. See [runtime-packages.txt](provenance/runtime-packages.txt). Use the branch's vendored `rl_games` and `eigendexplore`, not an unrelated installed version. [README.md](README.md#environment) explains installation.

On YJ, the KAIT and DJ paths above were not mounted at those names. Transfer files explicitly if needed. Existing generation and evaluation directories are historical records; create your own output directory for reproductions.

Under the original KAIT training root:

| Relative path | Contents |
|---|---|
| `code/` | Exact source used by the run |
| `screw_resets.npz`, `screw_goals.json`, `eigen_dexonomy_xhand.npz`, `september_config.yaml`, `training_template.yaml` | Original copies of the five training inputs |
| `reset_preparation.json` | Upstream reset extraction details |
| `resolved_environment_config.json`, `training_config.json` | Effective configurations after overrides |
| `rl_environment.json`, `rl_runtime_model.json` | Environment and actuator audits |
| `rl_smoke.json`, `checks/` | Original startup checks |
| `train-own.sh`, `logs/slurm-197338.log` | Original submission script and scheduler log; submitted as `sub-own` |
| `training/0_furniturebench-square_table-contact_termination_2s/` | `best/`, `last/`, `nn/`, scalar logs |
| `interactive_viewer/` | Native stochastic training HTML rollouts |
| `wandb/` | Local W&B run files |
| `source.patch`, `changes-from-parent.patch` | Original source provenance |

The original `code/assets` symlink resolves to `/home/nas_main/sibeenkim/draftrec/repos/simtoolreal-050/assets`. The published branch carries the required assets directly.

## 4. Train the same setup from scratch

Follow [the package checks and smoke test](README.md#check-the-package-and-run-a-simulation-smoke-test), then [the training commands](README.md#train). For KAIT, from a fresh branch checkout:

```bash
export PYTHON_BIN=/home/nas_main/sibeenkim/draftrec/venv_isaacsim/bin/python
export LD_LIBRARY_PATH=/home/nas_main/sibeenkim/draftrec/syslibs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export WANDB_BASE_URL=https://forge.coreweave.com/wandb
# Authenticate W&B with your own account before submitting.
sbatch experiments/furniturebench_square_table/train-sub-own.sh \
  /absolute/path/to/YOUR_NEW_training_directory \
  --wandb-entity YOUR_ENTITY
```

The wrapper requests **account `sub`, QoS `sub-own`**, one GPU, eight CPUs, and 200 GB RAM. It runs input verification, CPU checks, and a 96-environment simulator check before starting 24,576-environment training. Require the passing `rl_smoke.json`; process exit status alone is insufficient for Isaac startup failures. Do not reuse the original run directory or its W&B identity.

This is fresh training. The experiment-specific `adapters/train_screw.py` deliberately sets `load_checkpoint=False` and has no resume flag. The generic upstream README's checkpoint flag is not a tested resume command for this adapter. The checkpoint contains training/SGS state, but resuming this exact adapter would require explicitly wiring and validating restoration. Checkpoint evaluation is available as described next.

A rerun reproduces the setup, not an identical optimizer trajectory: the original trainer seed was 20261015, the environment seed was not explicitly set, and GPU simulation can vary.

## 5. Evaluate the policy from engagement to completion

### Exact checkpoint used for existing evaluation and assembly results

The earlier evaluation used a separately frozen **2,044,723,200-step** checkpoint. Keep it for reproducing those results; use the 2.58B best for new work.

```text
DJ checkpoint:
/home/nas4_user/sibeenkim/work/_dexagent_table_rl_eval_20261018/checkpoint.pth

SHA-256:
e54d8f3cb08e1d931c6ed694cb5da56bdc9184c726087929b3fefe433bc658e4

Size:
408,176,515 bytes
```

[Historical checkpoint metadata](provenance/historical-evaluation-checkpoint.json) preserves its identity. The evaluation root contains `evaluate.py`, frozen `code/`, `inputs/assets/`, reset pool, goals, resolved configs, robot audit, and `snapshot.json`. It also retains `payload.tar.gz`, `task-assets.tar.gz`, and `viewer-inputs.tar.gz` for transfer. `launch.py` and `evaluation-launch.json` preserve the original four-worker launch.

The evaluation visits **every progress-zero reset once**, with no SGS sampling, one leg per worker, deterministic mean actions, zero exploration conditioning, and eight goals required. It uses the configured 0.01 tolerance, held arm, 2-second contact termination, reach-box termination, and no retract. The 4,800-step limit was not reached by any of the completed evaluation episodes.

| Leg | Evaluated resets | Eight-goal completions |
|---|---:|---:|
| 1 | 18,882 | 13,498 |
| 2 | 13,588 | 8,961 |
| 3 | 15,102 | 11,529 |
| 4 | 20,752 | 14,955 |
| **Total** | **68,324** | **48,943 (71.63%)** |

These numbers belong to the **2.04B checkpoint**, not the recommended 2.58B best. They are engagement-to-finish results, not initial-scene acquisition or complete four-leg assembly success rates.

Existing results are under `/home/nas2/sibeenkim/_dexagent_table_rl_eval_20261018/leg1` through `leg4`. Each contains `results.jsonl`, `complete.json`, `planned_reset_rows.npy`, `episodes/row_*.npz`, `launch.json`, configs, and runtime checks. `analysis/` contains the aggregate analysis. Recordings include 60 Hz commanded 19-joint targets, measured joints/velocities, object poses, policy actions, goal poses, and progress.

To rerun one leg on an allocated/free DJ GPU with the historical checkpoint:

```bash
export TABLE_EVAL_ROOT=/home/nas4_user/sibeenkim/work/_dexagent_table_rl_eval_20261018
export PYTHONPATH="$TABLE_EVAL_ROOT/code/adapters:$TABLE_EVAL_ROOT/code/rl_games:$TABLE_EVAL_ROOT/code"
export OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/nas4_user/sibeenkim/work/_033_venv_isaacsim/bin/python \
  "$TABLE_EVAL_ROOT/evaluate.py" \
  --work "$TABLE_EVAL_ROOT" \
  --output /absolute/path/to/YOUR_NEW_eval/leg1 \
  --leg 1 --num-envs 3072 --seed 20261018 --maximum-steps 4800 --headless
```

Run legs 2–4 with their own output directories. Preserve the original GPU batch size and seed for the closest comparison. For a small functional check, use `--num-envs 96 --limit 96` and a separate output. The evaluator rejects an output containing `results.jsonl`.

### Use the recommended best with the evaluator

Create a private evaluation workspace on DJ. Copy the following from the frozen evaluation root, preserving directory structure:

```text
code/
inputs/
evaluate.py
screw_resets.npz
screw_goals.json
resolved_environment_config.json
training_config.json
rl_runtime_model.json
snapshot.json
```

Copy the recommended KAIT checkpoint into that workspace as `checkpoint.pth`. For example, from a workstation with both SSH aliases configured, after creating your workspace:

```bash
scp -3 \
  kait:/home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_reproduction_20261019/checkpoints/best-step-2584608768.pth \
  kaist140:/absolute/path/to/YOUR_PRIVATE_eval_workspace/checkpoint.pth
```

In the **private** `snapshot.json`, record the actual checkpoint's `global_steps=2584608768`, `checkpoint_sha256=baa40c076ca5519912bcbbd376b76d973c44e2cf924be7d4b2a506d730c64c15`, `source_size=408188001`, and the preserved KAIT `checkpoint_source` above. If retaining `checkpoint_epoch`, it is 6573. Remove stale source timestamps from the older snapshot. The evaluator reads this metadata for its reports; changing weights alone would leave misleading provenance. Verify the copied file's SHA-256 first.

Then run the same evaluation command with `TABLE_EVAL_ROOT` set to your private workspace and a fresh output. This handoff does not claim a new full evaluation of the 2.58B checkpoint; the included full evaluation results remain those of 2.04B.

## 6. Where the reset pool came from

The data lineage is:

```text
Task-specific Dexonomy grasp generation
  -> packed native grasp/squeeze states
  -> per-leg event configurations and accepted arm IK branches
  -> accepted engagement/intermediate grasp-state rows
  -> combined four-leg screw reset pool
  -> SGS sampling during RL
```

### Final event pools and exact reset reconstruction on KAIT

```text
Event/packed source:
/home/nas_main/sibeenkim/draftrec/_dexagent_stage1_003_fresh3_20261015

Original leg-4 reset pool used as extraction template:
/home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_leg4_20261017/screw_resets.npz
```

The source directory contains `inputs/event-specifications.json`, `inputs/scenes.json`, `inputs/run-spec.json`, `packed/`, `event-pools/table_assembly/`, `scripts/`, `hand/right.xml`, and event validation/completion reports. The reset builder uses **layout 0 only**. The other event layouts are not part of this training pool.

For each leg it selects native-passing `grasp` states at `thread_engaged` and `turn_1` through `turn_7`, expands all accepted arm branches, and retrieves finger squeeze targets from the matching `packed/<request_id>.npz`. Engagement rows additionally require a passing `lead_in`. `fully_screwed` is the final goal, not a reset progress stage. The extraction first rebuilds leg 4 and requires exact array equality with the original 198,234-row pool.

Rebuild on KAIT, using this branch checkout and a new output directory:

```bash
/home/nas_main/sibeenkim/draftrec/venv_isaacsim/bin/python \
  adapters/prepare_multi_resets.py \
  --source /home/nas_main/sibeenkim/draftrec/_dexagent_stage1_003_fresh3_20261015 \
  --leg4 /home/nas_main/sibeenkim/draftrec/_furniturebench_square_table_leg4_20261017/screw_resets.npz \
  --out /absolute/path/to/YOUR_NEW_reset_build
```

This is CPU data preparation; it does not launch Isaac or train a policy. Require 636,023 total rows and the per-leg counts above. Compare arrays to the bundled NPZ rather than assuming reserialization preserves its archive byte hash. Original leg-4 NPZ SHA-256: `d6d6b22d167298a81c065b2216d803bbe8f41ecdf988b1a71605ebd65f524187`.

### Native grasp generation and event construction records

| Site | Root and contents |
|---|---|
| DJ | `/home/nas4_user/sibeenkim/work/_dexagent_stage1_003_fresh3_20261015`: `Dexonomy/`, `run-spec.json`, `scripts/`, `task-assets/`, `prep/`, `generated_templates/`, `generation/`, `packed/`, `event-dist/`, `metrics/`, and `launch*.json` |
| YJ | `/home/nas5/sibeenkim/work/_dexagent_stage1_003_fresh3_20261015`: native generation and distributed event shards, `Dexonomy/`, `event-dist/`, `generated_templates/`, `generation/`, `packed/`, `metrics/`, and YJ launch records |
| KAIT | `/home/nas_main/sibeenkim/draftrec/_dexagent_stage1_003_fresh3_20261015`: final event pools, scene/event specifications, geometry environment, IK/event scripts, and validation reports |

These are complementary directories, not identical mirrors. The DJ native generator is `scripts/run_lane.py`; `collect.py` packs its outputs. KAIT event scripts include `event_coordinator.py`, `event_geometry.py`, `event_worker.py`, `ik_service.py`, `launch_ik.py`, and `validate_event_pools.py` under `scripts/`. The exact original commands and device assignments remain in their launch records. Historical scripts contain original roots and completion-marker behavior; copy and retarget them to fresh output locations before a new upstream generation run.

The task-specific leg generation requested 12 grasp types: **2 Small Diameter, 3 Medium Wrap, 6 Prismatic 4 Finger, 7 Prismatic 3 Finger, 8 Prismatic 2 Finger, 9 Palmar Pinch, 13 Precision Sphere, 14 Tripod, 20 Writing Tripod, 25 Lateral Tripod, 27 Quadpod, 29 Stick**. Request IDs begin with `square_table_leg__`. The recorded configuration used 40 epochs, 64 retained candidates per epoch, seed 20261015, a hybrid depth-2 template, and native tests at 60 Hz control / 240 Hz physics. Read `run-spec.json` and launch records for all generator settings. These candidate grasps are subsequently filtered into event states and arm branches, so their count is not the RL reset-row count.

### EigenDEXplore PCA provenance is separate

The exploration basis copied into training is also at:

```text
YJ:
/home/nas5/minhopark/_053/eigen/eigen_dexonomy_xhand.npz

SHA-256:
443432d4a3c14b8ffba9477970cca5b8de193fd83e715a6dfc4fa38d26da62de
```

It was fitted from the general Dexonomy bank at `/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16_deliver/dexonomy/` on DJ, including `extra/`. The main-only KAIT mirror is `/home/nas_main/sibeenkim/draftrec/_graspoc/cap16/reset_full/grasps/`. The donor set comprises 12,899,466 native-passing grasps and 38,698,398 first-pregrasp/grasp/squeeze postures; six components retain 90.4248% of variance. The generator is [`omnireset_dexonomy/data/make_dexonomy_basis.py`](../../omnireset_dexonomy/data/make_dexonomy_basis.py). This broad donor bank supplies exploration statistics, not the square-table reset states. The small bundled PCA file is all training needs.

## 7. Later generated data and assembly examples

These are **outputs using the 2.04B policy**, not additional inputs to the RL training above.

| Result | Code / plans | Recordings / results |
|---|---|---|
| 480,000 one-leg requests: 10,000 scenes × 4 first-leg choices × 12 grasp types | DJ `/home/nas4_user/sibeenkim/work/_table_oneleg_480k_20261018`; frozen `production-code/` and final planner `production-code-v2/` | DJ `/home/nas2/sibeenkim/_table_oneleg_480k_20261018/production` |
| Full four-leg example | DJ `/home/nas4_user/sibeenkim/work/_table_fourleg_example_20261019`; plans under `/home/nas2/sibeenkim/_table_fourleg_example_20261019/chains` | DJ `/home/nas2/sibeenkim/_table_fourleg_example_20261019/attempt-006-direct` |
| Later four-leg grasp-diversity search | DJ `/home/nas4_user/sibeenkim/work/_table_fourleg_diverse_20261019` | DJ `/home/nas2/sibeenkim/_table_fourleg_diverse_20261019` |

The complete four-leg recording shown in the original studio video is:

```text
/home/nas2/sibeenkim/_table_fourleg_example_20261019/attempt-006-direct/episodes/3170.npz
```

It completes legs in order **1 → 3 → 4 → 2**, reaches all 32 screw goals, and transitions directly without returning to the original reset arm pose. Its attempt directory includes `launch.json`, the exact `execute_source.py`, results, and runtime audits. For reproduction preserve the original 16-environment batch, seed 2026101904, and ID order `3141,3156,3165,3166,3170,3174,3176,3179,3185`; sample 3170 is environment index 4. The launch record contains the full command. Use a fresh output/runtime directory and the saved executor; do not overwrite `attempt-006-direct`. The four legs in this particular example use different instances of grasp type 8. It is not the claimed four-different-type example.

The one-leg and four-leg directories retain their own manifests, launch records, and source snapshots. The historical planner scripts are separate from the RL package; use their saved versions for reproducing those motions. Changing the checkpoint to the recommended best creates a new result, not an exact replay of the existing one.

## 8. Degradation evidence and verification status

The read-only degradation analysis and captured evidence are on KAIT:

```text
/home/nas_main/sibeenkim/draftrec/_rl_degradation_audit_20261019
```

Start with `REPORT.txt`; `evidence/` includes compact SGS snapshots, checkpoint metadata, source copies, and native scalar curves. The scripts compare completion curves and recent outcomes of matched reset rows. Nothing in that investigation changed the training runs. Its recorded training status is historical, not a live scheduler status check.

Publication verification matched 335 source files, 671 asset files, and the input hashes. CPU contact/SGS/reach checks passed. A fresh 96-environment simulation check on DJ passed with all four legs, all eight starting stages, fixed-arm control, runtime actuator checks, goal completion without retract, the 119/120-step contact boundary, and immediate reach termination. See [publication-checks.json](provenance/publication-checks.json) and [publication-rl_smoke.json](provenance/publication-rl_smoke.json).

The handoff verified checkpoint metadata and file hashes and created the preserved best copy. It did not launch another training run, rerun the complete dataset, or produce a new evaluation score for the best checkpoint.
