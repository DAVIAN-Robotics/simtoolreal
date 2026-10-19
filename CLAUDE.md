# Branch furniturebench_square_table-omnireset_dexonomy-eigendexplore_dexonomy-sgs

Read `experiments/furniturebench_square_table/README.md` first. This branch packages the exact four-leg training run `furniturebench-square_table-contact_termination_2s_20261017_062847`.

- Entry point: `adapters/train_screw.py`; task adapter: `adapters/reference_screw_env.py`.
- Required import order: `PYTHONPATH=$TREE/adapters:$TREE/rl_games:$TREE`. Use the vendored trainer and EigenDEXplore.
- 12 learned finger actions, seven reset arm joints held, 134 actor / 156 critic observations.
- 636,023 reset rows, global SGS, four open sockets, one active leg per episode; leg identity may change at reset.
- Fixed board/support poses, fixed 0.01 screw-goal acceptance, separate reward tolerance curriculum, no retract.
- Immediate reach-box termination and 2-second whole-hand/object contact-loss termination. Hand–fixture force penalty/termination remain disabled.
- Bundled prepared geometry, actuator profile, reset pool and goals must remain consistent. Do not silently substitute assets, move observation frames, or change reset semantics.
- Publication changes are recorded in `experiments/furniturebench_square_table/provenance/portability.patch`. New launchers use fresh output directories and retain the original training behavior.
- For KAIT use account `sub`, QoS `sub-own`; never `sub-extra`. Do not submit or change running training without user authorization.
- Run outputs, checkpoints, credentials, caches, and virtual environments do not belong in Git.

The parent branch's OmniReset/Dexonomy generation documentation is retained in `omnireset_dexonomy/README.md`. It is separate from the bundled table training entry point. Third-party Dexonomy files retain their own license.
