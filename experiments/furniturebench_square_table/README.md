# FurnitureBench square-table training

This branch packages the training code and inputs used by [furniturebench-square_table-contact_termination_2s_20261017_062847](https://forge.coreweave.com/wandb/draftrec/simtoolreal/runs/furniturebench-square_table-contact_termination_2s_20261017_062847).

**For checkpoint downloads, exact server paths, upstream data, and evaluation commands, start with [REPRODUCE.md](REPRODUCE.md).** The recommended checkpoint is the preserved best at **2,584,608,768 global steps**, rather than a later latest checkpoint after performance declined. The earlier 2.04B checkpoint used for existing evaluation and assembly results is listed separately.

It trains one shared finger policy for screwing a leg into any of the four sockets. The arm holds the pose from the sampled reset. Episodes start from saved grasped states at engagement or intermediate screw progress. Acquisition, arm transport, and chaining four completed legs are outside this RL training task.

**Start with [`adapters/train_screw.py`](../../adapters/train_screw.py), then [`adapters/reference_screw_env.py`](../../adapters/reference_screw_env.py).** The bundled inputs are sufficient to start training; generating new grasps or running Astra is unnecessary.

## What is included

| Component | Location |
|---|---|
| Training entry point and resolved configuration assembly | [`adapters/train_screw.py`](../../adapters/train_screw.py) |
| Saved resets, fixed-arm control, goal progression | [`adapters/reference_screw_env.py`](../../adapters/reference_screw_env.py) |
| Global success-guided reset sampling | [`adapters/sgs.py`](../../adapters/sgs.py) |
| SGS checkpoint state | [`adapters/sampler_checkpoint.py`](../../adapters/sampler_checkpoint.py) |
| Whole-hand/object contact-loss timer | [`adapters/contact_loss.py`](../../adapters/contact_loss.py) |
| Reach-box termination and bounds | [`adapters/reachability.py`](../../adapters/reachability.py), [`reach_geometry.json`](../../adapters/reach_geometry.json) |
| Actual robot actuator profile and runtime audit | [`adapters/isaac_sysid.py`](../../adapters/isaac_sysid.py), [`hand_sysid.json`](../../adapters/hand_sysid.json), [`audit/recent_training_robots.py`](../../adapters/audit/recent_training_robots.py) |
| Assembly environment, rewards, goal acceptance | [`isaacsimenvs/tasks/precise_assembly/`](../../isaacsimenvs/tasks/precise_assembly/) |
| Observation construction and inherited task mechanics | [`isaacsimenvs/tasks/simtoolreal/`](../../isaacsimenvs/tasks/simtoolreal/) |
| PPO/SAPG implementation | [`rl_games/`](../../rl_games/) |
| EigenDEXplore integration | [`eigendexplore/`](../../eigendexplore/) |
| Native W&B HTML rollout viewer | [`pose_viewer.py`](../../isaacsimenvs/tasks/simtoolreal/pose_viewer.py), [`index.template.html`](../../isaacsimenvs/utils/interactive_viewer/index.template.html) |
| Reset pool, goal sequences, configuration, exploration PCA | [`inputs/`](inputs/) |
| Prepared leg and four-socket fixture assets | [`task/inputs/assets/`](task/inputs/assets/) |
| Robot and supporting registry assets | [`assets/urdf/`](../../assets/urdf/) |
| Source/input provenance and original launch command | [`provenance/`](provenance/) |

The task uses the prepared `body3x` leg/thread geometry and the open four-socket fixture from the recorded run. Do not substitute differently scaled FurnitureBench assets. Some additional assets are required because the original problem registry validates its other registered tasks when imported.

## Exact training setup

| Setting | Value |
|---|---|
| Robot | Franka FR3 + XHand, `fr3-xhand-a4hs` actuator profile |
| Learned action space | 12 finger actions |
| Commanded robot joints | 7 held arm joints + 12 finger joints |
| Actor / privileged critic observations | 134 / 156 entries |
| Spatial coordinates | Inherited world-axis, environment-local construction |
| Policy / physics frequency | 60 / 120 Hz |
| Parallel environments | 24,576, with six SAPG blocks of 4,096 |
| Actor / critic learning rate | 0.0001 / 0.0001 |
| PPO horizon / minibatch / mini-epochs | 16 / 98,304 / 2 |
| Trainer seed | 20261015; environment seed was not explicitly set |
| Initial weights | Fresh training |
| Reset pool | 636,023 rows across four legs and eight progress stages |
| Reset sampling | One global SGS distribution over individual rows |
| Goal acceptance and goal-reaching reward | Fixed insertion tolerance 0.01 for every screw goal |
| Retract | Disabled |
| Board pose / support height | Fixed; support z = 0.38 m |
| Reset velocities | Zero |
| Hand–fixture contact penalty / force termination | Disabled |
| Added failure termination | Immediate reach-box separation; 2 s consecutive whole-hand/object contact loss |

Each environment has one active dynamic leg and a fixture with all four sockets open. A new reset can select a different leg/socket in the same environment. This is the behavior of the linked run, including its W&B viewer.

`inputs/september_config.yaml` supplies the base environment settings. `inputs/training_template.yaml` supplies actor/critic observation lists and trainer settings. The entry point explicitly applies the recorded overrides. Read the generated `resolved_environment_config.json`, `training_config.json`, and `rl_runtime_model.json` for the effective values rather than treating either input YAML as the final configuration.

Observation/action delays, object/goal pose noise, joint-velocity observation noise, object-scale observation noise, and random object forces/torques are disabled. The exact physics and actuator settings are retained and checked at startup.

### Reset pool and success meaning

| Leg | Reset rows |
|---|---:|
| Leg 1 | 169,683 |
| Leg 2 | 122,620 |
| Leg 3 | 145,486 |
| Leg 4 | 198,234 |
| **Total** | **636,023** |

The pool stores `arm_q`, `hand_q`, `hand_target`, `object_xyzw`, `progress`, and `leg_id`, alongside provenance fields. Pose arrays in these inputs use **xyz + quaternion xyzw**. The adapter converts to Isaac's wxyz representation.

A reset at progress `p` has `8 - p` remaining goals. SGS records success when those remaining goals are completed. Training success therefore includes intermediate starts; it is not a measurement of engagement-to-finish success from progress zero.

The inherited `current_success_tolerance` curriculum is still updated and logged, but this run's `goal_mode=preInsertAndFinal` and `random_goal_fraction=0` select the **fixed** insertion tolerance for all eight screw goals. The goal-reaching reward uses that same acceptance mask; it does not use a looser curriculum threshold. The configured 0.01 is multiplied by the inherited `keypoint_scale=1.5` when comparing scaled keypoint distances. See [`_keypoint_success_tolerance_m`](../../isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py) and [`obs_utils.py`](../../isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py).

SGS uses the most recent 100 outcomes per reset row, target success rate 0.5, temperature 2, concentration 1, and epsilon 0.0001. Rows without observations use success rate zero, so all rows initially receive equal weight. There is no leg/stage sampling hierarchy. [`sgs.py`](../../adapters/sgs.py) contains the exact weighting formula and duplicate-row update handling. This is the implementation used for the run, not a later upstream SGS implementation.

Contact-loss termination counts **120 consecutive policy steps** with no object contact on any of the 13 hand bodies, including the fixed-merged palm/wrist body. Contact on either physics substep clears the counter. Table/fixture contact does not clear it. The independent reach-box rule terminates immediately once the object's box and the hand's reachable box are disjoint.

## Environment

The recorded runtime used Python 3.11.15, Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1, PyTorch 2.7.0+cu128, NumPy 1.26.0, Warp 1.14.0, and W&B 0.27.2. See the full [`runtime-packages.txt`](provenance/runtime-packages.txt) snapshot.

Use the repository's [Isaac Sim installation guide](../../docs/isaacsim_installation.md) for the general installation procedure, with the versions above for this run. In particular, the recorded PyTorch build is **cu128**, while the general guide describes cu126. Install the root project with `--no-deps`: its old Isaac Gym dependency pins are unsuitable for this Isaac Sim environment. Use the vendored `rl_games` and `eigendexplore` through the `PYTHONPATH` below.

No Isaac Sim binaries, virtual environment, account credentials, or trained checkpoints are committed. Authenticate W&B with your own account. For the linked Forge server, configure `WANDB_BASE_URL` for your deployment before login; use your own W&B entity if you cannot write to `draftrec`.

## Check the package and run a simulation smoke test

From a fresh checkout of this branch:

```bash
git clone --branch furniturebench_square_table-omnireset_dexonomy-eigendexplore_dexonomy-sgs \
  https://github.com/DAVIAN-Robotics/simtoolreal.git
cd simtoolreal
export PYTHON_BIN=/absolute/path/to/isaacsim-python
export PYTHONPATH="$PWD/adapters:$PWD/rl_games:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" experiments/furniturebench_square_table/verify_package.py
"$PYTHON_BIN" adapters/check_contact_loss.py
"$PYTHON_BIN" adapters/check_kernels.py

# Use a fresh directory. This copies the bundled inputs and source provenance.
"$PYTHON_BIN" experiments/furniturebench_square_table/prepare_run.py \
  --work /absolute/path/to/table-smoke

# Run on an allocated GPU. This exits before PPO training or W&B initialization.
"$PYTHON_BIN" adapters/train_screw.py \
  --work /absolute/path/to/table-smoke --num-envs 96 --smoke --headless
cat /absolute/path/to/table-smoke/rl_smoke.json
```

The simulator check exercises all four legs and all eight starting stages, goal progression, completion without retract, fixed arm commands, finite observations/rewards, SGS state save/restore, actual hand/object contacts, the 119/120-step contact-loss boundary, and immediate reach termination. Require a fresh passing `rl_smoke.json`; the Isaac process can exit with status zero after a Python exception.

## Train

Run on an allocated GPU with the Isaac Sim environment and W&B authentication ready:

```bash
export PYTHON_BIN=/absolute/path/to/isaacsim-python
bash experiments/furniturebench_square_table/train.sh \
  /absolute/path/to/fresh-table-training \
  --wandb-entity YOUR_ENTITY \
  --wandb-project simtoolreal \
  --run-name furniturebench-square_table-contact_termination_2s
```

This launcher copies the inputs into a fresh directory, runs the CPU checks and 96-environment simulation check, and then starts fresh training with 24,576 environments. Additional arguments are passed to `train_screw.py`. Reducing environment count changes the experiment; keep a multiple of six for SAPG. The historical `--epochs` option counts PPO updates; the W&B x-axis reports global environment steps.

For KAIT, submit **from the repository root** using the included `sub-own` wrapper:

```bash
export PYTHON_BIN=/absolute/path/to/isaacsim-python
sbatch experiments/furniturebench_square_table/train-sub-own.sh \
  /absolute/path/to/fresh-table-training \
  --wandb-entity YOUR_ENTITY
```

It requests partition `compute`, account `sub`, QoS `sub-own`, one GPU, eight CPUs, 200 GB RAM, and seven days. It preserves Slurm's GPU assignment. `SIMTOOLREAL_ROOT` can specify the checkout if submitting from another directory. Change only site-specific allocation details when using a different cluster.

## Outputs and W&B

The run directory contains copied inputs, source provenance, resolved environment/trainer configuration, the runtime robot audit, `training/` checkpoints, W&B files, and `interactive_viewer/` HTML rollouts. The native viewer records the stochastic actions used during training: 600 frames per capture at interval 1,000 environment steps, environment zero. Its original presentation and logging machinery are preserved.

The default mesh URLs point at this published branch. If moving the code/assets to another repository, pass `--viewer-raw-base https://raw.githubusercontent.com/OWNER/REPO/BRANCH/`. Keep the trailing slash. URL checks remain enabled.

The vendored multi-policy training loop saves `last/model.pth` every 200 updates. With `save_frequency=1000`, numbered periodic snapshots are retained at updates 1,000, 4,000, 9,000, 16,000, and so on (square multiples of 1,000). Best is saved on an improvement in training reward from update 10 onward, with `best/model.pth` pointing to the named checkpoint in `nn/`. This loop does not use the configured `save_best_after=100` value. Final-limit saves are separate. See [`a2c_common.py`](../../rl_games/rl_games/common/a2c_common.py). SGS state is attached through the existing environment-state checkpoint interface. For this run, one update corresponds to 393,216 global environment steps.

## Provenance and changes for sharing

The frozen source was compared against the linked run's live code directory: all archived files matched. The five training input files also matched by SHA-256. [`manifest.json`](provenance/manifest.json) records original and published hashes, the parent Git commit, input hashes, and the run identity.

The training implementation is copied from that source. The two source edits for publication are visible in [`portability.patch`](provenance/portability.patch):

- `train_screw.py` resolves task/viewer assets from the checkout and accepts W&B identity and asset-location flags instead of machine-specific paths.
- `check_kernels.py` reads its bundled geometry fixture from this experiment directory.

The new preparation/launch/verification scripts and this guide package those files for a fresh checkout. No reward, observation, action, reset, SGS, goal, termination, or physics behavior was changed for publication. Historical absolute paths in the original inputs and provenance identify the source data; the launch path does not require those directories.

[`reset_preparation.json`](inputs/reset_preparation.json) records the original event-pool sources and selection. [`prepare_multi_resets.py`](../../adapters/prepare_multi_resets.py) is retained as the original preparation code, but rebuilding those upstream event pools is separate from training with the included reset pool.

### Validation of this publication

Source/input identity, package contents, required URDF mesh references, the contact-loss timer boundaries, global SGS sampling and history updates, and reach-box results on 28,958 recorded frames passed. Both float32 and float64 geometry checks passed. Preparation was checked with and without Git available.

A **fresh 96-environment GPU simulation check of the published code passed on October 19, 2026**, using an RTX 3090 and the recorded Python/Isaac Sim/Isaac Lab/PyTorch/NumPy versions. It exercised all four legs and eight starting stages, 120 policy steps, held arm targets, inherited observations, goal completion without retract, SGS state save/restore, real hand/object contact detection, the 119/120-step contact-loss boundary, and immediate reach-box termination. The actuator runtime audit also passed. See [`publication-rl_smoke.json`](provenance/publication-rl_smoke.json), [`publication-rl_runtime_model.json`](provenance/publication-rl_runtime_model.json), and [`publication-checks.json`](provenance/publication-checks.json).

[`original-rl_smoke.json`](provenance/original-rl_smoke.json) and [`original-rl_runtime_model.json`](provenance/original-rl_runtime_model.json) retain the **original training run's** reports separately. The new test validates the packaged environment and adapters; it does not rerun long training. The provided training launcher also requires a fresh passing simulator check before starting training on your machine.
