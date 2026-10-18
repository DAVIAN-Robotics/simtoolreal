# Dexonomy at large scale, the cap16 run and how to redo it

This document records the Dexonomy grasp synthesis run on the 3,200 cap16 objects with the reference XHand, done by agent 4 of session 051 on 2026-09-27 and 09-28. A fresh agent with no memory of the run should be able to redo it on another large object set from this file alone. Every path, command and number below comes from the run's code, logs or decisions register. Where something was never recorded, the text says so.

**How to read it.** Part 1 gives the result and the three changes a next run must make. Part 2 is the redo, step by step, with exact commands. Part 3 explains why the machinery looks the way it does, with every decision, trap and measured number.

**Short names used throughout.** The Daejeon cluster (dj) is the 8 GPU hosts kaist67, kaist121, kaist140, kaist143, kaist144, kaist145, kaist148 and kaist149. Its shared NAS (nas4) is a Synology at `<nas4 host>:/volume2/user`, mounted at `/home/nas4_user`. The work root on nas4 is `D=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy`. The Dexonomy clone is `$D/Dexonomy` at upstream commit d561ff9, the same as `.reference/126-dexonomy`. The run directory is `S=$D/cap16`, and the scheduler state is `$S/gsched`. The python is `PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python`. Local files are relative to `/home/sibisibi/Projects/research/dexmanip/.output/051-grasp-poc/`. The decisions register is `status/decisions/agent4.json`, cited by id as A4-Dnn. A pair is one (chunk, template) queue, and a job is one epoch range of one pair's init.

---

## Part 1. What the run produced, and what to change next time

### 1.1 Result and where things stand

**Output.** Over 40 epochs the run made 39,680,000 refinement starts, 10 per object, template and epoch. Refinement kept 16,820,193 of them, or 42.4%. Dexonomy's own MuJoCo test passed 12,899,466 of the kept grasps, or 76.7%. Every one of the 3,200 objects has passes. The minimum is 79 per object, the median 3,983 and the maximum 7,703. Every object has passes in at least 18 of the 31 grasp types, and the median object has all 31.

**Wall clock.** Generation took 16h 54m in two runs on 48 GPUs.

| Run | Epochs | UTC | Wall clock |
| --- | --- | --- | --- |
| First run | 0 to 4 | 09-27 21:52:31 to 09-28 00:40 | 2h 47m, delivered and verified |
| Continuation, phase 1 | 5 to 19 | 09-28 01:18:38 to about 08:35 | 7h 16m |
| Continuation, phase 2 | 20 to 39 | about 08:35 to 15:25:15 | 6h 50m. Init ended at 14:26, the last own test at 15:25 |
| Delivery | all 12,899,466 passes | 09-28 15:38 to 09-29 10:47:02 | Stopped after each nas4 freeze. The last 3h 7m ran by hand at about 16 readers in total |

**Where things stand.** Everything is delivered and verified. The first run's 1,583,000 passes were delivered and verified at 00:40 on 09-28. They are kept in `$S/delivered_5epoch/` and on kait at `draftrec/_graspoc/cap16/candidates/dexonomy/`. The 40-epoch delivery holds all 12,899,466 passes in `$D/cap16_deliver/dexonomy/`, as one `<stem>.npz` plus `extra/<stem>.npz` for each of the 3,200 objects, 14 GB in total. DONE was written at 10:47:02 on 09-29. At 10:50:15 `src/agent4_cap16_verify_delivery.py 32` on kaist121 reported 3,200 entries and 12,899,466 grasps on disk. That equals the 12,899,466 unique pass lines, with 0 missing and 0 not in the pass lists, and it printed "delivery verified". The delivery ran from 15:38 on 09-28 to 10:47 on 09-29, with stops after every nas4 freeze, and section 1.3 tells that story. The Isaac hold test was never run on this set. Sibeen ruled it runs after generation.

### 1.2 Passes per grasp type

The table comes from `src/cap16_pass_counts.py`, which wrote `$S/pass_counts/counts.csv`, one row per object with 31 type columns and a total. The figure is `vis/image/cap16_pass_counts.png`. The per-object maximum per type is 400, which is 10 starts times 40 epochs all passing.

| Type | Passes | Share % | Objects with 0 | Median per object | Max |
| --- | --- | --- | --- | --- | --- |
| 1_Large_Diameter | 477,889 | 3.7 | 109 | 146 | 375 |
| 2_Small_Diameter | 128,172 | 1.0 | 58 | 31 | 213 |
| 3_Medium_Wrap | 464,733 | 3.6 | 26 | 118 | 390 |
| 4_Adducted_Thumb | 208,599 | 1.6 | 14 | 52 | 342 |
| 5_Light_Tool | 186,441 | 1.4 | 19 | 50 | 280 |
| 6_Prismatic_4_Finger | 896,211 | 6.9 | 2 | 298 | 400 |
| 7_Prismatic_3_Finger | 904,834 | 7.0 | 2 | 302 | 400 |
| 8_Prismatic_2_Finger | 840,509 | 6.5 | 9 | 279 | 399 |
| 9_Palmar_Pinch | 610,799 | 4.7 | 14 | 187 | 376 |
| 10_Power_Disk | 314,413 | 2.4 | 65 | 76 | 349 |
| 11_Power_Sphere | 553,004 | 4.3 | 0 | 170 | 368 |
| 12_Precision_Disk | 116,259 | 0.9 | 112 | 26 | 185 |
| 13_Precision_Sphere | 745,082 | 5.8 | 1 | 241 | 397 |
| 14_Tripod | 816,155 | 6.3 | 15 | 275 | 390 |
| 15_Fixed_Hook | 162,644 | 1.3 | 90 | 35 | 276 |
| 16_Lateral | 473,079 | 3.7 | 30 | 135 | 356 |
| 17_Index_Finger_Extension | 404,930 | 3.1 | 0 | 100 | 370 |
| 18_Extension_Type | 221,547 | 1.7 | 118 | 53 | 264 |
| 20_Writing_Tripod | 248,266 | 1.9 | 41 | 74 | 271 |
| 22_Parallel_Extension | 201,715 | 1.6 | 11 | 43 | 237 |
| 23_Adduction_Grip | 79,614 | 0.6 | 457 | 14 | 152 |
| 24_Tip_Pinch | 400,892 | 3.1 | 84 | 112 | 382 |
| 25_Lateral_Tripod | 477,862 | 3.7 | 10 | 146 | 310 |
| 26_Sphere_4_Finger | 401,764 | 3.1 | 4 | 105 | 355 |
| 27_Quadpod | 702,860 | 5.4 | 3 | 237 | 388 |
| 28_Sphere_3_Finger | 455,914 | 3.5 | 0 | 132 | 367 |
| 29_Stick | 364,956 | 2.8 | 6 | 78 | 378 |
| 30_Palmar | 318,747 | 2.5 | 6 | 90 | 301 |
| 31_Ring | 317,379 | 2.5 | 4 | 83 | 316 |
| 32_Ventral | 105,288 | 0.8 | 89 | 20 | 207 |
| 33_Inferior_Pincer | 298,909 | 2.3 | 7 | 87 | 248 |
| **Total** | **12,899,466** | 100 | 0 objects with no pass at all | 3,983 | 7,703 |

These are own-test passes in MuJoCo. On the 16-object proof of concept, types 12 to 32 passed the Isaac hold test at only 0 to 13%, even where the own test passed 65 to 69%. That was measured on the older 25 templates, so it does not carry over directly, but the hold test decides what is usable.

### 1.3 The three changes a next run must make

**Settled cause of the 09-28 nas4 freezes.** Sibeen ruled that our delivery's per-file read storm caused them. The evidence is in three steps.
1. nas4 froze fleet-wide twice within 36 minutes of the delivery start, for about 10 minutes and then 6 to 9 minutes. The second freeze also hung kaist139, which ran nothing of ours, and hung other users' threads on our hosts.
2. The delivery was stopped at 16:21:14 on 09-28, and our load reached zero at 16:21:54.
3. A probe on kaist143 then looked up a fresh file name every 30 s for 31 minutes. All 62 probes answered in 6 to 12 ms, with no hang and no slow phase. No freeze has been seen since.

The delivery read one small npy file per pass, 12.9M files, from 8 hosts with 64 processes each. Each file cost about 4.5 NFS calls, which were OPEN, READ, CLOSE, a third of a LOOKUP, and one DELEGRETURN that returned the read delegation nas4 granted on the OPEN. Every client sat at its limit of about 31 calls in flight, with 1,100 to 2,800 DELEGRETURN calls queued behind them. The fleet read 2,507 files per second at about 11,000 calls per second.

**09-29 showed the same at every scale.** Three restarts froze the loaded clients, or all of nas4, again.
1. Two hosts, then a third, at 64 processes froze their clients within about 20 minutes.
2. One host at 24 processes read 2,268 files/s and froze its client after 12.5 minutes.
3. Eight hosts at 24 processes froze all of nas4 within 3 minutes.

A probe on a lightly used client never saw the per-client freezes, so a health probe has to run on every reading host. The orchestrator then finished the delivery by hand at about 16 readers in total, 12 on kaist140 and 4 on kaist148. That moved about 10 to 20 objects a minute, and nas4 stalled every host whenever the total went past about 16 to 20. So for per-file reads the sustainable load on nas4 is about 16 readers across the whole fleet, not per host.

**Change 1. Write outputs in batches from the start, never one file per grasp.** The run already does this for init, whose output is one npz per batch since A4-D92, proven byte-identical. Refinement and the own test still write per grasp through upstream code. That means 16.8M `grasp_data` npy files, one per kept grasp, then 12.9M `succ_grasp` symlinks, one per pass, and a `new_tmpl_pool` symlink for every pass that may seed template evolution. The next run should close that gap in the shared worker, which already runs the per-item upstream functions and already serves npz members through an `np.load` shim.
1. **Fuse refinement and own test in one lease.** The worker refines an init, keeps the refined dict in memory, and own-tests it at once through the same shim. Upstream's `np.save` of the grasp file is captured in memory, the same way init's saver was. The own test then needs no file at all.
2. **Record verdicts in memory, not as symlinks.** Upstream `_single_eval` makes the pass and pool links with `os.system("ln -s ...")`. A small patch to `eval_data.py` hands those two events to the worker instead. The hand loader already reads the pool from index files since A4-D74, so a pool entry becomes an index line `path<TAB>npz` served by the shim in init.
3. **Write each lease once, as one npz.** A lease writes one npz with every kept refinement for the record, one with its passes already in delivery form, and one with its evolution templates. The delivery form is `hand_pos`, `hand_quat`, `q_grasp`, `q_squeeze` and `q_pregrasp`, with `stem` and `tmpl` columns added. It writes to a dot-temp name and renames. A lease of about 4,000 items gives about 1,300 passes, so 12.9M passes become roughly 10,000 files instead of 12.9M.

Per-object output files must not be appended by several hosts. On nas4 that left NUL holes in 8 files during this run, A4-D77. Per-lease write-once files avoid it. None of this is built yet. Before launch, the worker change needs the same proof the run used for every other change. That means the same starts in, and byte-equal grasp dicts and equal verdicts out, against the per-file path on a sample of pairs.

**Change 2. Deliver from the batches, never file by file.** With change 1, delivery reads the per-lease pass npz, about 10,000 files, groups rows by `stem`, and writes one `<stem>.npz` plus `extra/<stem>.npz` per object. The pass npz could even be the delivery itself, with the per-object regroup done on kait. Any read of millions of small files on nas4 must be treated as a load on a shared NAS. Sibeen has to approve its host and process count first. If a per-file read cannot be avoided, run it at about 16 readers in total across all hosts, as the 09-29 finish did.

**Change 3. Start from the final machinery, not the path this run took.** This run began with per-host drivers, then moved to shared workers, then to init jobs, fixing traps as they appeared. A redo should launch the final shape directly. That is init as epoch-range jobs on every GPU, one shared worker per host, one indexer, and every patch applied before the first job. Part 2 gives that order. Three smaller lessons go with it.
- Size worker pools by `min(CPU budget, (RAM - 30 GB) / 1.4 GB)` from the first minute. kaist140 ran out of memory at 184 workers.
- Sample per host, from minute one, how many processes run, wait on storage and sit idle. Push if use falls below about 80%.
- Never list a big directory in a hot loop. A template pool listing took 132 s, and one template dir took 8 s.

---

## Part 2. How to redo it, step by step

### 2.1 Inputs, the objects, the hand and the templates

#### 2.1.1 Objects

**What cap16 is.** 3,200 URDFs with `pool.json`, from `/home/nas5/minhopark/assets_039/objects_cap16` on yj. The kait copy for the hold test is `/home/nas_main/sibeenkim/draftrec/assets_040/objects_cap16`. Each `pool.json` entry has `urdf, key, scale, extent_m, mass_kg, parts`. Each URDF has one link `object_root`, one visual mesh, one to 16 convex collision pieces with origin and scale, and a full inertial. The 3,200 entries reference 18,058 mesh files, 770,848,160 bytes in total. 780 entries have 1 collision piece and 641 have 16, the most. URDF masses run from 13 g to 902 g, with a median of 109 g. They come from piece volume times 500 kg/m3 at pool staging, in `.output/039-dex4d-object-pool/src/stage_dex4d_pool.py`.

**Stems must be unique.** The stem is the URDF basename without `.urdf`, for example `core__bottle-109d55a137c042f5760315ac3bf2c13e_s008`. Dexonomy's Warp mesh cache is keyed by name, so two objects with one stem would share a mesh.

**Step 1. Pull the pool to dj and prove it equals the source.** yj cannot reach dj, so dj pulls from yj through `ssh -p 10004 sibeenkim@<yj host>`. The scripts are `src/agent4_cap16_manifest_side.py`, which writes per-side sha256 manifests, and `src/agent4_cap16_compare.py`, which compares two of them.
```
# on yj-4, in a dir holding the manifest scripts
$E/bin/python agent4_cap16_manifest_side.py <pool dir> yj_manifest.json --files files_all.txt --workers 48
# on kaist140
rsync -a -e "ssh -p 10004 -o StrictHostKeyChecking=no" sibeenkim@<yj host>:<pool dir>/ $S/pool/
rsync the files_all.txt and yj_manifest.json into $S/xfer/, then split -n l/12 -d files_all.txt part_
# 12 parallel streams, two per host on 6 hosts
cd $S/xfer && nohup rsync -a --files-from=part_NN -e 'ssh -p 10004 -o StrictHostKeyChecking=no' sibeenkim@<yj host>:/ $S/mesh_root/ > part_NN.log 2>&1 &
# prove equality, then write the prep manifest
$PY agent4_cap16_manifest_side.py $S/pool dj_manifest.json --remap /=$S/mesh_root/ --workers 48
$PY agent4_cap16_compare.py yj_manifest.json dj_manifest.json          # must print difference_counts {} and exit 0
$PY agent4_cap16_prep_manifest.py dj_manifest.json $S $S/manifest.json
```
The meshes land at their original absolute yj path below `$S/mesh_root/`. `$S/manifest.json` is a list of 3,200 dicts with `stem, key, mass_kg, pool_mass_kg, urdf, collision, visual`. Each collision and visual element carries `mesh, xyz, rpy, scale`. The same compare against the kait manifest proved the hold-test copy identical too.

**Step 2. Prep each object with exact hulls, not MeshProcess.** `src/agent4_cap16_prep.py` follows the exact-hull path agent 3 used for m05. It runs no CoACD, no ACVD and no normalization. Per object it does three things.
1. It loads every collision element and applies its URDF origin and scale, with `euler_matrix(*rpy, axes="sxyz")`. It then takes the convex hull with `manifold3d.Manifold.hull_points` and writes `urdf/meshes/convex_piece_NNN.obj`.
2. It unions the hulls with `Manifold.batch_boolean(..., OpType.Add)`, asserts the union is watertight with a volume at most the hull sum, and writes it as `mesh/simplified.obj`. The name is what Dexonomy reads. The mesh is not simplified.
3. It exports `urdf/coacd.urdf`, `urdf/coacd.xml` and `info/simplified.json` with MeshProcess's python exporters only, and saves the scene cfg.

The scene cfg is saved as `scene_cfg/<stem>/floating/real.npy`.
```python
scene_cfg = {"scene": {stem: {"type": "rigid_object",
                              "file_path": f"{d}/mesh/simplified.obj", "xml_path": f"{d}/urdf/coacd.xml",
                              "urdf_path": f"{d}/urdf/coacd.urdf", "info_path": f"{d}/info/simplified.json",
                              "scale": np.array([1.0, 1.0, 1.0]), "pose": np.array([0.0, 0, 0, 1, 0, 0, 0]),
                              "mass_urdf_kg": mass, "mass_kg": float(np.clip(mass, 0.05, 0.5))}},
             "scene_id": f"{stem}/floating/real",
             "task": {"type": "force_closure", "obj_name": stem}}
```
Scale 1 and identity pose mean the run's world frame is the URDF `object_root` frame, the same frame the hold test uses. Paths are absolute, so chunk dirs can symlink the cfgs. The MeshProcess python comes from the yj-4 clone at commit 638d777, rsynced without `third_party/` to `$S/MeshProcess/`. Its import works under the dexonomy env. Launch one shard per host.
```
rsync -a -e "ssh -p 10004 -o StrictHostKeyChecking=no" --exclude third_party/ --exclude __pycache__/ sibeenkim@<yj host>:/home/nas5/sibeenkim/work/_graspoc/dexonomy/MeshProcess/ $S/MeshProcess/
# i = 0..7 over kaist67:48 kaist121:100 kaist140:200 kaist143:32 kaist144:32 kaist145:40 kaist148:100 kaist149:100 (host:processes)
ssh $h "cd $D && nohup $PY agent4_cap16_prep.py cap16/manifest.json $S/MeshProcess/src $S/prep $i 8 $n > cap16/prep_shard_$i.log 2>&1 < /dev/null &"
```
It took 5 to 12 s per 400-entry shard, and 42 s on one loaded host.

**Why not MeshProcess.** On the 16-object proof of concept, MeshProcess bridged concave pockets. Its CoACD pieces sat up to 8.7 mm outside the true hulls, and `simplified.obj` reached 12.9 mm. The exact-hull path keeps the object the hold test sees. The CoACD and ACVD builds are still recorded in section 3.3 in case a future object set has only raw meshes.

**Step 3. Apply the mass cap exactly.** Sibeen ruled "Mass cap clip(URDF mass, 0.05, 0.5) kg for every test". Init does not use mass for `force_closure`, whose external wrench is zero. So the cap acts only in Dexonomy's own test, through the object density in `dexonomy/op/eval_data.py`. MuJoCo integrates every convex piece separately, so overlapping pieces count twice. The density must therefore divide by the summed piece volume, not the union volume. First add that volume to every info file.
```
cd $D && OMP_NUM_THREADS=1 $PY agent4_cap16_fix_mass.py $S/prep 0 1 48
# prints: 3200 info files updated, MuJoCo mass = target on all; pieces/union volume max 1.1226, over 1.01: 259
```
`src/agent4_cap16_fix_mass.py` writes `info["collision_volume"]` by temp and rename. It rebuilds each object's eval env and asserts the MuJoCo body mass equals `mass_kg` to 1e-6. The cap moved 505 entries up to 50 g and 42 down to 500 g. The `eval_data.py` edit is in section 2.1.4.

**Step 4. Check the prep.** `src/agent4_cap16_prep_check.py <prep dir> <shard> 8 <n> <out json>` loads each cfg with Dexonomy's loader, checks paths and the clipped mass, and builds the eval env with the hand. All 3,200 passed.

#### 2.1.2 The hand, the XHand MJCF port

The port lives in the clone at `$D/Dexonomy/assets/hand/xhand/`, with `dexonomy/config/hand/xhand.yaml` pointing at `${asset_root}/hand/xhand/right.xml`. It is built by `src/agent4_build_xhand_mjcf.py <reference xhand dir> <out dir>` from agent 1's reference hand at `kait:/home/nas_main/sibeenkim/draftrec/_graspoc/reference/xhand/`. That dir holds `xhand_right.urdf`, `collision_hulls/`, `meshes/` and `joint_order.json`. The rules are these.
- **Fixed links merge into their movable parent.** Every link reached through fixed joints joins its movable ancestor body, with the accumulated transform and a fused inertial. The palm merges `right_hand_link`, `ee_link` and `back_link`. Finger link1 bodies merge their back link1, and link2 bodies merge back link2 and the tip.
- **Collision geoms come first, the body's own hull first.** Each URDF collision becomes a mesh geom from the reference hull, `group=3`. The body's own link comes before merged links, because GraspADMM projects onto a body's first collision geom. Visual STL geoms follow with `contype=0 conaffinity=0 density=0`.
- **Joints and actuators copy the training drive.** Each revolute joint becomes a hinge with the URDF range, `damping=0.1`, `armature=0` and `frictionloss=0`. Each actuated joint gets a `general` actuator with `gainprm=[3.0]`, `biasprm=[0, -3.0, 0]`, ctrlrange equal to the joint range, and forcerange equal to the URDF effort, 1.1 or 0.4 N m. Dexonomy's squeeze is torque divided by kp, so kp must equal the training stiffness of 3.0.

There are no contact excludes. The MJCF root is `palm` with no joint, and Dexonomy adds the freejoint at load. With the freejoint, MuJoCo filters only parent-child pairs, the same as PhysX in training. The palm hull against `index_rota_link1` is a live grandparent pair past `index_joint1 = 1.419 rad`, so deep index flexion self-collides, in training too. The model has nq 12, and 19 with the freejoint. It has 60 geoms. `right.xml` is 21,360 bytes, sha256 `3934a16ceee4efca6ab9ff7a3b7ea50765bc7a5cb8c12f4efca0d5a5eeb0bc4d`.

`src/agent4_check_xhand_mjcf.py` proves the port. It checks that joints, ranges, gains, damping and effort equal `joint_order.json`. It checks that every one of the 60 geoms' vertices match the reference URDF pose over 50 random joint vectors, with a max error of 1.25e-7 m. It also reports self-contacts, 0 at q = 0 and 0 at mid-range. The support files are `body_group.yaml`, `keypoint.yaml` and `skeleton.yaml`, all in `src/agent4_xhand_port/`. Do not reuse session 012's port. It dropped the fixed-joint offsets, put tips 4.2 to 6.5 cm off, used kp 5 and added 16 excludes.

#### 2.1.3 The templates, astra's 31

**Which set.** Sibeen approved astra's rules-corrected package, types 1 to 33 without 19 Distal Type and 21 Tripod Variation. Those two are excluded by his instruction and must never be restored. The package is `src/astra_rules_corrected_xhand_port/`. The tarball is `src/xhand_templates_rules_corrected.tar.gz`, 3,545,119 bytes, sha256 `2d8710e403131c1c88221e02002d6154dd9370c52bdddedd10e42e3734bc5359`. Checksums for all 138 files are in `status/rules_corrected_artifacts.json`. The package's hand model files are byte-identical to agent 4's port. Only `raw_anno/` and `init_tmpl/` differ. The earlier disk-corrected package is superseded.

**Install.** `src/agent4_cap16_install_templates.py` checks every package file against the json and refuses to overwrite if any hand model file differs. It backs up the old templates, installs `raw_anno/` and `init_tmpl/`, and asserts 31 of each with no `19_` or `21_`.
```
scp src/xhand_templates_rules_corrected.tar.gz status/rules_corrected_artifacts.json src/agent4_cap16_install_templates.py kaist143:$D/
ssh kaist143 'bash -s' <<'EOF'
D=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy
mkdir -p $D/templates_rules_corrected && tar xzf $D/xhand_templates_rules_corrected.tar.gz -C $D/templates_rules_corrected
python3 $D/agent4_cap16_install_templates.py $D/rules_corrected_artifacts.json $D/templates_rules_corrected/astra_rules_corrected_xhand_port $D/Dexonomy/assets/hand/xhand $D/Dexonomy/assets/hand/xhand_backup_25templates_20260928
EOF
```
`op=tmpl` is not rerun. The approved converted `init_tmpl/*.npy` are used as delivered.

#### 2.1.4 The clone patches, in the order to apply them

Every patch script replaces exact upstream strings that must occur once and fails loud otherwise. Apply them from the clone root with `python3 <script> .`, or `$PY <script> $D/Dexonomy` for the first. They build on each other textually, so the order matters. Section 3.1 classifies each as result-changing or proven identical.

| Order | Script | File changed | What it does |
| --- | --- | --- | --- |
| 1 | `src/agent4_cap16_patch_dexonomy.py` | `op/eval_data.py`, `data/hand_loader.py` | Per-object own-test mass from `mass_kg`. Hand loader refresh from 1 s to 60 s |
| 2 | inline edit, below | `op/eval_data.py` | Density divides by `collision_volume` |
| 3 | `src/agent4_cap16_patch_40epoch.py` | `op/gen_init.py`, `data/hand_loader.py` | `+op.epoch_start` with reseed 12 plus start. Pool read from `gsched/pool/*__<tmpl>.txt` index files, not a recursive glob |
| 4 | `src/agent4_cap16_patch_forkserver.py` | `data/obj_loader.py` | DataLoader `multiprocessing_context="forkserver"` |
| 5 | `src/agent4_cap16_patch_poolnul.py` | `data/hand_loader.py` | Skip pool index lines with a NUL byte |
| 6 | `src/agent4_cap16_patch_asyncsave.py` | `op/gen_init.py` | Init saves on one saver thread, logging `Saved N` |
| 7 | `src/agent4_cap16_patch_initcuts.py` | `op/gen_init.py` | Dirs made once per batch. skip_done globs only when the pair log already has this job's epochs |
| 8 | `src/agent4_cap16_patch_npzbatch.py` | `op/gen_init.py` | One npz per init batch, `Saved N npz <path>` |

The second edit was applied inline during the run and exists in no script. Apply it by hand after patch 1. It replaces the line patch 1 wrote in `_single_eval`.
```python
            # 051 cap16: per-object mass clip(URDF mass, 0.05, 0.5) kg from the scene cfg, not cfg.obj_mass.
            # Divided by the summed piece volume MuJoCo integrates, so overlapping pieces do not add mass.
            o_cfg["density"] = o_cfg["mass_kg"] / (o_info["collision_volume"] * np.prod(o_cfg["scale"]))
```
Afterwards, `git -C $D/Dexonomy diff` should show exactly these changes.

### 2.2 Environment and hosts

#### 2.2.1 The dj conda env

**Target.** `/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy`, python 3.10.20, torch 2.7.1+cu128, triton 3.3.1, warp-lang 1.8.0, mujoco 3.3.3, trimesh 4.5.3, numpy 2.2.6, hydra-core 1.3.2 and manifold3d 3.5.1. It uses an editable install of the dj clone, so `import dexonomy` resolves to `$D/Dexonomy` from any directory. The yj-4 env `bk_dexonomy` does not, because its editable install points at a grasp-bakeoff copy. There, always run from the clone root.

**Build.** `scripts/agent4_build_dj_env.sh` does four things, run on kaist140 in tmux.
1. It rsyncs the clone and port from yj-4.
2. It runs `conda create -p $ENV --file bk_dexonomy_conda_explicit.txt`.
3. It runs `pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu128 -r <yj-4 pip freeze without -e lines>`.
4. It runs `pip install --no-deps -e $D/Dexonomy`.

Capture the spec on yj-4 first with `pip freeze`, `conda list` and `conda list --explicit` of bk_dexonomy. Then run three more steps by hand.
```
git -C $D/Dexonomy remote set-url origin https://github.com/JYChen18/Dexonomy.git
$PY -m pip install --no-deps setuptools==81.0.0      # plain pip freeze drops it, and triton needs it
$PY -m pip check                                    # must print No broken requirements found
```
Parity was proven against yj-4 with `pip freeze --all`, 106 lines, and with the name, version and channel of `conda list`, 134 packages. Both diffed clean.

**CUDA 13 wheels.** The yj freeze carried 18 unused CUDA 13 packages. Uninstalling them afterwards deleted `libnccl.so.2`, which the cu12 wheels share, and broke torch for 3 minutes until the 14 cu12 wheels were force-reinstalled with `pip install --no-deps --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128 <nvidia-*-cu12==... pins>`. In a redo, drop the 18 lines from the requirements before installing. That is an inference, not a tested recipe.

**Host check.** `scripts/agent4_dj_host_check.sh` runs `src/agent4_dj_host_check.py` on each host on a free GPU. It proves `dexonomy` imports from the clone, runs a torch matmul and a warp mesh query, and builds the XHand env with nq 19. kaist67, 121, 140, 145, 148 and 149 are RTX 3090. kaist143 and 144 are TITAN RTX. kaist143 and 144 are Intel Xeon Silver 4210R, and the other six are AMD EPYC.

#### 2.2.2 Hosts, GPUs and the chunk plan

| Host | CPU threads | GPUs used | RAM | Objects | Batches per epoch per template |
| --- | --- | --- | --- | --- | --- |
| kaist67 | 64 | 0 to 6 | 503 GB | 283 | 29 |
| kaist121 | 128 | 0 to 7 | 503 GB | 627 | 63 |
| kaist140 | 256 | 0 to 3 | 251 GB | 313 | 32 |
| kaist143 | 40 | 0 to 3 | 251 GB | 184 | 19 |
| kaist144 | 40 | 0 to 3 | 251 GB | 184 | 19 |
| kaist145 | 96 | 2 to 7 | 377 GB | 434 | 44 |
| kaist148 | 128 | 0 to 6 | 314 GB | 548 | 55 |
| kaist149 | 128 | 0 to 7 | 377 GB | 627 | 63 |

kaist145 GPUs 0 and 1 belong to junyeongahn. Run `gpustat -i` on each host at launch and take only GPUs nobody else holds. `src/agent4_cap16_plan.py <S> cap16_hosts.json plan.json` shuffles the stems with `random.seed(0)` and sizes chunks by `min(CPU, GPU)` capacity. That uses 620 CPU-s and 52 GPU-s per object, with a TITAN RTX at 0.6 of a 3090. It writes symlinks `$S/chunks/<host>/scene_cfg/<stem>/floating/real.npy` to the prep cfgs, and the glob `$S/chunks/<host>/scene_cfg/**/floating/*.npy`. Each chunk has its own exp dir `$D/Dexonomy/output/cap16_<host>_xhand/`, and all share one template pool `$S/new_tmpl_pool`. RAM is MemTotal from `/proc/meminfo` on 09-28. Read it again before sizing pools, since other users share these hosts.

### 2.3 Generation and delivery

#### 2.3.1 How data flows

One pair, a chunk and a template, goes through four stages. There are 8 chunks times 31 templates, so 248 pairs per stage kind.
1. **Init, on GPU.** Upstream `gen_init` fits the template to 10 objects at a time, with 1,024 surface points times 8 in-plane spins. It keeps up to 10 valid starts per object per epoch. It logs `Get N valid` per batch and writes one npz per batch at `init_data/<tmpl>/_batch/<epoch>/<j>.npz`. The members are named `<stem>/floating/real/<epoch>_<k>_grasp` and hold the exact bytes upstream would have saved per grasp.
2. **The indexer.** It reads every pair's init log from a saved byte offset. It reproduces upstream's object order, `np.random.seed(12)` then `np.random.permutation(sorted(glob))`, batches of 10. It appends one line per init to `gsched/todo/grasp__<chunk>__<tmpl>.txt`. The line is `path<TAB>npz`, where the path is the per-grasp name upstream would have used.
3. **Refinement, on CPU.** Each host's shared worker leases pairs and runs upstream `_single_grasp` per line. The npz member is served through an `np.load` shim. Kept grasps are written by upstream to `grasp_data/.../<epoch>_<k>_grasp.npy` and appended to `gsched/todo/eval__<chunk>__<tmpl>.txt`.
4. **Own test, on CPU.** The worker runs upstream `_single_eval`. A pass creates the `succ_grasp` symlink. Passes are appended to `gsched/pass/<chunk>__<tmpl>.txt`. Passes that may seed evolution also get a `new_tmpl_pool` link, indexed in `gsched/pool/<chunk>__w<host>__<tmpl>.txt`, which the init hand loaders read.

Every queue, pass and index line is relative to the clone root, and element 4 of it is the object stem. Every python process that calls upstream first does `os.chdir($D/Dexonomy)`.

#### 2.3.2 The scheduler state, `$S/gsched/`

| Path | Written by | Read by | Content |
| --- | --- | --- | --- |
| `todo/grasp__<c>__<t>.txt` | indexer only | workers | append-only queue, `path` or `path<TAB>npz` |
| `todo/eval__<c>__<t>.txt` | the grasp lease holder | workers | append-only queue of `grasp_data` paths |
| `done/<kind>__<c>__<t>.txt` | the lease holder, after outputs | workers | every processed line, as queued |
| `seen/<pair>@<q_off>_<d_off>` | the lock holder | workers | empty file whose name holds the queue offset done so far and the done-list size then |
| `lock/<pair>/owner` | worker, by `mkdir` | workers | `<host> <pid>`, removed when the lease finishes |
| `aff/<pair>@<host>` | indexer for grasp pairs, grasp lease holder for eval pairs | workers | host affinity, local pairs are leased first |
| `indexed/grasp__<c>__<t>` | indexer | workers | init of the pair finished epoch 39 and every batch is queued |
| `final/<pair>` | worker | workers, delivery trigger | pair has nothing left. 496 names means generation is done |
| `pass/<c>__<t>.txt` | eval lease holder | gate, delivery, verify, counts | one `grasp_data` path per own-test pass |
| `pool/<c>__w<host>__<t>.txt` | eval lease holder on that host | init hand loaders | one pool link per line, one writing host per file |
| `count/<host>.txt` | worker | progress | `<UTC> <kind> <chunk> <tmpl> <items> <outputs>` per lease |
| `stop/<host>` | restart script | worker | lease nothing new, finish in-flight leases, exit |
| `init_jobs.json` | `init_jobs.py` | runners, progress | `{id, chunk, tmpl, start, end, batches, prev}` per job |
| `ijob_lock/<job>/owner`, `ijob_done/<job>`, `ijob_failed/<job>` | runners | runners, indexer, progress | job claim, done and failed markers |
| `index_state.json`, `index_summary.json` | indexer, temp and rename | indexer, progress | per-pair log offset, epoch, batch, open batches, finished |
| `log/` | everyone | poll, gate | `<host>.log` workers, `init_<host>.log` runners, `indexer.log`, `watchdog_<host>.log`, `restart_<host>.log` |

Outside gsched, `$S/progress.txt` gets one line per 5 minutes from `src/agent4_cap16_progress.py`, and `$S/plan.json` holds the chunk plan.

#### 2.3.3 Launch order for a redo

This is the order the final machinery needs. It is not the order this run took, which Part 3 explains. Run everything detached with `nohup ... < /dev/null &` from `/tmp`, so a VPN drop cannot kill it. Never use Slurm on dj.

1. **Env, inputs and patches.** Sections 2.1 and 2.2.1, then `cap16_hosts.json` and `plan.py`.
2. **Init jobs.** `cd /tmp && python3 $S/agent4_cap16_init_jobs.py` writes `gsched/init_jobs.json`. Phases are `[(5, 20), (20, 40)]` in this run. A fresh run starts at epoch 0. Epochs per job are set per host, 1 for kaist121 and kaist149, 2 for kaist145 and kaist148, 3 for kaist67 and kaist140, and 5 for kaist143 and kaist144. That gives 60 to 110 batches per job. Each job names its predecessor, and jobs are sorted by start epoch so all pairs advance together. This run had 4,464 jobs, 144 per template. Retune `STEP`, `PHASES` and the epoch count for a new object set.
3. **Smoke test.** One job on 3 objects must exit 0 and log `Finish initialization.`, as below.
```
cd $D/Dexonomy && $PY -m dexonomy.main op=init hand=xhand exp_name=cap16_smoke40 new_tmpl_dir=$S/new_tmpl_pool tmpl_name=14_Tripod log_id=14 init_gpu=[0] op.epoch=6 +op.epoch_start=5 "op.object.cfg_path=\"<3-object glob>\"" op.object.n_cfg=null
```
4. **Init runners, 2 processes per GPU.** On each host, after checking that no other user holds the GPUs, run the command below. Each runner claims jobs by `mkdir`, runs them, and counts a job done only on exit 0 plus a new `Finish initialization.` line. It retries after 30 s, up to 30 tries.
```
nohup $PY $S/agent4_cap16_init_runner.py $h <gpu csv> 2 > $S/gsched/log/init_$h.out 2>&1 < /dev/null &
```
Each slot runs the command below with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`.
```
$PY -m dexonomy.main op=init hand=xhand exp_name=cap16_<c> new_tmpl_dir=$S/new_tmpl_pool tmpl_name=<t> log_id=<t number> init_gpu=[<g>] n_worker=4 op.epoch=<end> +op.epoch_start=<start> "op.object.cfg_path=\"<chunk glob>\"" op.object.n_cfg=null
```
Quote the glob inside the value like this, because a Hydra value with `**` or `[..]` parses as a list.
5. **The indexer, once, on a host with memory headroom.** It ran on kaist143 after it was moved off kaist140.
```
ssh kaist143 "cd /tmp; OMP_NUM_THREADS=1 nohup $PY $S/agent4_cap16_indexer.py 32 20 40 >> $S/gsched/log/indexer.out 2>&1 < /dev/null &"
```
The arguments are 32 listing threads, a 20 s cycle and 40 target epochs. It exits by itself when every pair's init is final.
6. **Shared workers, one per host, below init in priority.** The pool size `n` per host is `min(CPU budget, (RAM - 30 GB) / 1.4 GB)`. The CPU budget is threads minus init's measured CPU, minus 10% of threads, minus 1. The lease is 40 times n items. The pools this run held while init ran were kaist67 44, kaist121 99, kaist140 120, kaist143 27, kaist144 27, kaist145 74, kaist148 80 and kaist149 101. After init ended they were 56, 114, 160, 35, 35, 85, 114 and 114.
```
cd /tmp; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= nohup nice -n 10 $PY $S/agent4_cap16_gworker.py $h $n $((40*n)) reuse >> $S/gsched/log/$h.v2.out 2>&1 < /dev/null &
```
7. **Watchdog, progress and poll.** Start one watchdog loop per host, `nohup bash $S/agent4_cap16_watchdog_loop.sh $h`. It kills an init whose last log line is older than 600 s, and the runner retries it. Start one progress loop on one host. It runs `python3 $S/agent4_cap16_progress.py` every 300 s until `$D/cap16_deliver/dexonomy/DONE` exists. Run the workstation poll `scripts/agent4_cap16_poll7.sh` every 240 s. It alerts on a dead worker, a dead watchdog, a failed init, a swallowed item error, or less than 20 GB of available memory.
8. **Pool raise at init end.** `scripts/agent4_cap16_endinit_pools.sh` waits per host until every init job has started and the host runs no init process. It then restarts that host's worker at the post-init size through `scripts/agent4_cap16_restart_workers.sh "<host>=<n>"`.
9. **End.** `scripts/agent4_cap16_final.sh` waits for 496 `final/` names and `driver/EVAL_DONE` on all 8 chunks. It then runs the gate `src/agent4_cap16_predeliver.py` detached. With change 1 of section 1.3 the delivery then reads per-lease npz. Without it, the existing `src/agent4_cap16_deliver_final.py` reads one file per pass. It must then run at about 16 readers in total across all hosts, as on 09-29. Never run it at 8 hosts times 64, or even 8 hosts times 24.

**Restarting a worker.** Always go through `scripts/agent4_cap16_restart_workers.sh "<host>=<n> ..."`. For each host it launches `scripts/agent4_cap16_restart_one.sh` detached on that host, under a `mkdir` lock. That script touches `gsched/stop/<host>`, waits until the old worker has closed its in-flight leases and exited, starts the new one, and logs a token the caller waits for. Every restart flushes that host's model and mesh caches, so never restart all hosts at once.

#### 2.3.4 The shared worker in detail

`src/agent4_cap16_gworker.py <host> <n proc> <lease items> reuse` runs one `multiprocessing.Pool`.
- **Leasing.** Each scan reads five directories, `final/`, `indexed/`, `seen/`, `todo/` sizes and `aff/`, and nothing per pair. Local pairs come first, eval before grasp, shuffled by a host-seeded rng. A lease takes up to 40 times n pending lines from one pair by `mkdir lock/<pair>`. Pending lines are queued lines past the `seen` offset that are not in the done list. The worker keeps leasing while fewer than 2 groups per process are outstanding, up to 8 new leases per scan.
- **Running.** A lease splits into one task per object. `run_group` lists each input dir once, which defeats stale negative lookups from files another host just wrote. It opens each npz once and asserts every member exists outside upstream's error swallowing. It then calls upstream `_single_grasp` or `_single_eval` per line and asserts the return equals the input.
- **Finishing, in this order.** The worker appends kept grasps to the eval queue and marks the eval pair with its own host. It then appends passes to the pass list and pool links to its host's pool file. Only after all outputs does it append every processed line to the done list. Then it writes `seen` if the lease emptied the queue, removes the lock and writes a counter line. A pair becomes final when its init is indexed, or its grasp pair is final, and nothing is left.

**The model cache, `src/agent4_cap16_envcache.py`.** It keeps one pristine MuJoCo env per kind per process and hands out copies with a fresh `MjData`. It memoizes scene cfg and json loads, clearing at 64 entries. It serves assets from memory. Each XML is read once and parsed with `MjSpec.from_string`, the 60 hand meshes are held for good, and object mesh pieces sit in a 256 MB LRU, all handed to `MjSpec.compile` through `spec.assets`. Models are byte-identical to fresh builds. It cut nas4 syscalls per item from 24.4 to 6.4, and raised per-process throughput by 36%.

#### 2.3.5 Delivery as it exists now

- **Gate, `src/agent4_cap16_predeliver.py`.** It re-runs every item whose traceback upstream swallowed, since `safe_wrapper` logs it as a WARNING and returns as if done. Each such traceback must be a missing-file error. It then checks that no pass line repeats within a file, and that 20,000 randomly sampled pass lines each have their grasp file and succ link. It writes `$S/predeliver.json`.
- **Shards, `src/agent4_cap16_deliver_final.py <shard> <n shards> <n proc>`.** The final delivery used 16 shards. Each shard takes the stems at its index modulo the shard count, reads every pass list, loads each pass's grasp npy, and writes `<stem>.npz` with `hand_pos (n,3)`, `hand_quat (n,4)` wxyz normalized, `q_grasp`, `q_squeeze` and `q_pregrasp`, each `(n,12)` float32. It also writes `extra/<stem>.npz` with `tmpl` and `src` per row. Extra is renamed into place first, then main. The last shard to finish writes `DONE` with the totals. A relaunched shard keeps entries whose main file already exists, after asserting the row count, A4-D104. Only in-flight entries are redone. `scripts/agent4_cap16_relaunch_shard.sh` matches the 8-shard split only. For 16 shards, launch and stop by hand as below, and never run one shard on two hosts at once.
```
# launch shard <i> at <N> processes on a host, which returns at once
ssh <host> 'cd /tmp; nohup bash -c "OMP_NUM_THREADS=1 $PY $S/agent4_cap16_deliver_final.py <i> 16 <N> >> $S/deliver_final/shard_<i>.out 2>&1; echo \$? > $S/deliver_final/shard_<i>.rc" > /dev/null 2>&1 < /dev/null &'
# stop it by PID, wrapper first so it writes no exit status
w=$(pgrep -f "^bash -c OMP_NUM_THREADS=1 $PY $S/agent4_cap16_deliver_final.py <i> 16 "); p=$(pgrep -f "^$PY $S/agent4_cap16_deliver_final.py <i> 16 "); kill -9 $w; kill -9 $p
```
- **Verify, `src/agent4_cap16_verify_delivery.py <n proc>`.** It checks fields, dtypes, shapes and unit quaternions per entry. It checks that main and extra agree, and that the delivered sources equal the set of all pass lines, with nothing missing, nothing extra and nothing twice. It checks the DONE totals against disk. The first run's delivery verified clean.

#### 2.3.6 What this run actually launched, for the record

**First run, epochs 0 to 4, 09-27 21:52.** Per-host drivers, not the machinery above. The plan and launch were as follows.
```
cd $S && rm -rf chunks && python3 agent4_cap16_plan.py $S cap16_hosts.json plan.json
for h in kaist67 kaist121 kaist140 kaist143 kaist144 kaist145 kaist148 kaist149; do ssh $h "bash $S/agent4_cap16_launch.sh $h"; done
```
`scripts/agent4_cap16_launch.sh` refuses any planned GPU with a compute process or 500 MiB in use, and asserts 31 templates. It then starts `scripts/agent4_cap16_run.sh cap16_$H 5 $GPUS $KPAR $NWG $NWE "$GLOB" $S/new_tmpl_pool $TMPLS`. That driver runs one init process per template, 3 per GPU, retrying up to 30 times until `Finish initialization.` appears. It also runs upstream `op=grasp` and `op=eval` loops with the plan's worker counts. At 22:31 the grasp and eval loops were stopped by PID and replaced by shared workers v1, then by v2 at 22:58 through `scripts/agent4_cap16_switch_v2.sh` and `src/agent4_cap16_bootstrap.py`. Init stayed in the drivers. Two templates needed `scripts/agent4_cap16_rerun_init.sh` after 30 instant out-of-memory retries. The first run's delivery used `src/agent4_cap16_deliver_final.py` and verified clean.

**Continuation, epochs 5 to 39, 09-28 01:18:38.** The first run's delivery, `final/`, `indexed/` and chunk markers were moved to `$S/delivered_5epoch/` and `$S/gsched/prev_5epoch/`. `index_state.json` was reopened, and queues, done lists and pass lists were kept. The pool index was seeded by `src/agent4_cap16_pool_index.py <host> 48` on every host, 370,124 paths, and checked equal to the glob. Runners, workers and the indexer then started as in section 2.3.3. Every later clone patch went live for new jobs only, with running jobs finishing on the code they started with.

---

## Part 3. Why it looks like this

### 3.1 Every change to upstream behaviour, and its proof

Result-changing means the outputs differ from what upstream Dexonomy would produce with the same inputs. Proven identical means a test compared outputs with and without the change.

| Change | Register | Effect on results | Proof or reason |
| --- | --- | --- | --- |
| Own-test mass from `mass_kg`, divided by `collision_volume` | A4-D54 | Changes results on purpose. Sibeen's mass cap replaces upstream's 0.1 kg for every object | `fix_mass.py` asserts MuJoCo mass equals the target to 1e-6 on all 3,200. The first 4 minutes of the first run used the union volume, up to 12% heavier, on 259 entries |
| Hand loader refresh 60 s instead of 1 s | A4-D55 | Changes timing, so which evolved templates are buffered at a batch differs | Upstream is timing-dependent here too, so neither setting is reproducible. About 150 inits globbing nas4 every second would load a shared NAS |
| Reseed 12 plus epoch_start for a resumed job | A4-D72, A4-D73 | Changes the random draw stream against one uninterrupted process. Object order and batch composition are unchanged | Needed, since `rng_test.py` shows a fresh seed-12 process repeats epoch 0's start poses, hash `3549959bb6e398f5` twice, and 17 gives `8ec1f79a21714f82` |
| Pool from index files instead of a recursive glob | A4-D74 | Same set of paths at the check point | `pool_index.py --check` found index equal to glob on 31 of 31 templates. A listing took 132 s for one template's 34,673 files |
| DataLoader forkserver | A4-D76 | None | `rng_test.py` through the patched copy gave the same hashes |
| Skip pool lines with NUL | A4-D77 | None on valid lines | Lines with NUL were never loadable. Whether a real path was lost inside the 8 NUL runs is unknown |
| Init saves on a saver thread | A4-D79 | None | `asyncsave_test.py`, 300 files byte-equal to the bytes handed over, `Saved N` equal to `Get N` |
| Dirs once per batch, skip_done only when resuming | A4-D82 | None | `initcuts_test.py`, 300 files byte-equal, and a second run skipped 3 of 3 batches with files unchanged |
| One npz per init batch | A4-D92 | None downstream. `init_data` changes format and is not delivered | `npzbatch_test.py`, 300 members byte-equal and the indexer's queue lines exact. `npzworker_test.py`, 66 refined grasps byte-identical and 334 failing in both |
| Model reuse, loader memo, memory-served assets | A4-D60, D64, D65, D88, D97 | None on the same CPU model | `envcache_test.py`, up to 305 refined grasps equal in every key and 305 own-test trajectories equal. `memassets_test.py`, 40 objects with byte-equal `mj_saveModel` |
| Shared worker replaces upstream's grasp and eval loops | A4-D58, D59 | Item order only, and the own test sees only finished refinements | `gworker_test.py`, cfg equal to the CLI's, 24 re-refined grasps equal, 24 verdicts equal. `gworker_v2_test.py`, 60 inits, 49 kept as in the live run, 37 passed |

**One caveat on bit identity.** Refinement is bit-identical only on the same CPU model. 7 of 158 recomputed grasps differed in `ho_c.wrench` by 3e-9 to 8.7e-8 between the Intel and AMD hosts, and one in `squeeze_qpos` by 2.3e-10, A4-D67. Any host refines any chunk, so output low bits depend on the host.

### 3.2 Decisions register

Every entry of `status/decisions/agent4.json`, A4-D1 to A4-D114, appears below, one line each, in id order within three groups. The class says how the entry bears on results. **R** changes generation results against upstream. **B** touches upstream code paths with a named identity proof. **I** is infrastructure, scheduling, monitoring or delivery, with results unaffected. **S** is a ruling by Sibeen or an order relayed by the orchestrator. The json holds the full what, why and evidence fields.

**Register anomalies.** A4-D45 to D50 are stamped 09-28 12:20 to 12:45 but happened on 09-27 at about 14:55 to 15:21, before the launch. The 40-epoch GO has no entry of its own and is cited in A4-D72. A4-D95 lists 8 watchdog kills while A4-D96 counts 13 in an overlapping window. A4-D100 and A4-D101 disagree on kaist140's post-init target, and A4-D101 is the later word. The A4-D99 gate evidence is a mid-run trial, and the final gate passed at 15:37:52.

#### Group 1. The XHand port and the 16-object proof of concept on yj-4, A4-D1 to A4-D44

- **A4-D1, I.** Both methods ran on yj-4, which had 255 idle CPUs and 8 idle A6000s.
- **A4-D2, I.** A fresh clone at d561ff9 ran from its own dir, because bk_dexonomy's editable install points elsewhere.
- **A4-D3, R.** The hand MJCF comes from agent 1's URDF with fixed links merged and each body's own hull first. All 60 geoms match the reference within 1.25e-7 m over 50 random joint vectors.
- **A4-D4, R.** kp is 3.0, damping 0.1 and forcerange the URDF effort of 1.1 or 0.4 N m, because squeeze is torque over kp and must press with the training drive.
- **A4-D5, R.** There are no contact excludes, matching training. Index link1 enters the palm hull past 1.419 rad, by up to 5.2 mm.
- **A4-D6, R.** Six body groups, one per finger plus the palm, as upstream does.
- **A4-D7, R.** Keypoints come from ray casts at the hulls, tutorial option B. op=tmpl accepted all of them.
- **A4-D8, R.** The skeleton has 17 segments inside the hulls, distal links included.
- **A4-D9, R.** Templates were posed by a least-squares script around a canonical object per type, since no GUI annotator was available.
- **A4-D10, R.** Each template's contact links copy the authors' Shadow template of the same type.
- **A4-D11, R.** Posing caps index_joint1 at 1.40 rad, below the self-collision onset.
- **A4-D12, R.** Five multi-start posing passes, keeping each template's best result.
- **A4-D13, R.** PASS means contacts within 2 mm and 30 degrees, and NEAR-MISS within 4 mm and 45 degrees. Both ran.
- **A4-D14, R.** The proof of concept ran 25 of 31 types, 14 PASS and 11 NEAR-MISS. Six types had no acceptable fit. cap16 superseded this with all 31 approved templates.
- **A4-D15, R.** Types 19 and 21 are excluded, as in the upstream dataset. This still holds.
- **A4-D16, S.** Sibeen said to run end to end with the 25 templates.
- **A4-D17, R.** Proof-of-concept objects were the union of each object's collision hulls, run through unchanged MeshProcess. cap16 replaced this with A4-D53.
- **A4-D18, I.** CoACD 2e4842c and ACVD b65c74a were built from source, CoACD with cmake below 4.
- **A4-D19, R.** One real-size scene per object, undoing MeshProcess's normalization, instead of its four random scales.
- **A4-D20, I.** The proof of concept kept upstream's 0.1 kg and friction 0.6 and 0.02. cap16 replaced the mass by A4-D54.
- **A4-D21, I.** The driver ran upstream's op sequence with per-op worker counts, and the grasp and eval loops waited for every init.
- **A4-D22, R.** 40 epochs instead of upstream's 5, because hold-test yields were 2 to 10%.
- **A4-D23, I.** GraspADMM kept its upstream config.
- **A4-D24, I.** GraspADMM's silent fallback to Dexonomy refinement was kept and measured.
- **A4-D25, I.** The common format takes hand pose from `grasp_qpos[:7]`, `q_squeeze` from `squeeze_qpos`, and `q_pregrasp` from the first pregrasp waypoint.
- **A4-D26, I.** Dexonomy and GraspADMM ran as separate experiments with their own template libraries.
- **A4-D27, I.** No split onto yj-3, since yj-4 alone finished in under an hour.
- **A4-D28, I.** Session 012's port was not reused. Its tips sat 42 to 65 mm off.
- **A4-D29, I.** The ACVD build needed VTK 9.2.6, its ffmpeg, GL and Boost dev packages, and a sysroot shim, over nine tries.
- **A4-D30, I.** MeshProcess ran from a venv overlay with CPU torch, since it imports torch at startup.
- **A4-D31, I.** CoACD's seed is random in the official task, so prep is not reproducible.
- **A4-D32, I.** MeshProcess output was kept although it bridges pockets by up to 8.7 mm.
- **A4-D33, I.** GraspADMM's QP fallback fired 0 times in 200 inits.
- **A4-D34, I.** Own-test mass came out 0.097 to 0.099 kg from the union-versus-pieces volume mismatch.
- **A4-D35, I.** Five inits per GPU with up to 10 retries on CUDA out-of-memory errors.
- **A4-D36, I.** A second delivery round went only to objects under 128 hold-test passes.
- **A4-D37, R.** Four objects got 80 epochs.
- **A4-D38, R.** m05 used exact hull pieces, the precursor of cap16's prep.
- **A4-D39, R.** m07 used a hulls-only mesh.
- **A4-D40, R.** GraspADMM m05 got 160 epochs.
- **A4-D41, I.** No more m07 generation. The zero-contact signature later proved to be a validator bug.
- **A4-D42, S.** Keep only Dexonomy and GraspADMM.
- **A4-D43, S.** Drop m07, 15 objects in the final table.
- **A4-D44, S.** Stop GraspADMM. Dexonomy is the focus, and cap16 is under consideration.

#### Group 2. The dj env and the first cap16 run, epochs 0 to 4, A4-D45 to A4-D71

- **A4-D45, S.** Build a dj env for Dexonomy only, with no Isaac Sim, no MeshProcess C++ and no runs yet.
- **A4-D46, I.** The env copies bk_dexonomy's conda spec and pip freeze with `--no-deps`. pip freeze --all and conda list diffed clean, at 106 and 134 lines.
- **A4-D47, I.** setuptools 81.0.0 was installed by hand, since pip freeze omits it and triton needs it.
- **A4-D48, I.** The clone and port were rsynced from yj-4, the origin was set to GitHub, and the clone installed editable.
- **A4-D49, I.** The smoke profile used all 16 proof-of-concept objects and 400 re-homed init files.
- **A4-D50, I.** Drivers 575 and 570 both run torch cu128 on the 3090 and TITAN RTX hosts. 8 of 8 live hosts passed.
- **A4-D51, S.** Sibeen's GO. All 3,200 entries, 5 epochs, the 31 approved templates, dj, maximum parallelism, and the mass cap `clip(URDF mass, 0.05, 0.5)` kg for every test.
- **A4-D52, I.** The pool was pulled to dj by 12 parallel rsyncs and matched yj and kait hash for hash on every piece.
- **A4-D53, R.** Exact-hull prep with no CoACD or ACVD, one scene cfg per stem at scale 1 and identity pose. All 3,200 compile.
- **A4-D54, R.** Own-test density comes from the capped mass divided by the summed piece volume, exact on all 3,200. For about 4 minutes, 259 overlapping entries tested up to 12% heavier.
- **A4-D55, R.** The template producer refreshes every 60 s instead of every 1 s, to spare the shared NAS.
- **A4-D56, R.** One chunk per host, sized to `min(CPU, GPU)` capacity, entries shuffled with seed 0, 3 inits per GPU. Each chunk has its own seed-12 object order, a stochastic difference only.
- **A4-D57, I.** Incremental delivery in rounds of 256 new passes. Superseded by A4-D66.
- **A4-D58, B.** Shared workers v1 lease pairs by `mkdir` locks and run upstream's per-item functions. 24 of 24 grasps and verdicts reproduce.
- **A4-D59, B.** Workers v2 add the indexer, queues and pass lists, and no worker lists a directory. A 60-init sandbox kept the same 49 grasps as the live run.
- **A4-D60, B.** Refinement reuses one compiled model per object per process. 150 grasps equal in every key, and 167 ms per start fell to 90 ms.
- **A4-D61, I.** Two templates' inits were rerun after 30 immediate out-of-memory retries on a full GPU.
- **A4-D62, I.** Workers keep 2 groups per process queued. 71,757 own tests from killed grasp passes were accounted separately.
- **A4-D63, I.** Delivery read pass lists by offset instead of globbing `succ_grasp`, which took 16 to 34 minutes. Superseded by A4-D66.
- **A4-D64, B.** The own test also reuses compiled models, under the A4-D60 proof, because fresh builds kept 141 of 220 kaist140 processes in NFS waits.
- **A4-D65, B and I.** Lease scans read 4 listings instead of about 2,500 per-pair metadata calls. Scene cfg and info loads are memoized, with 162 grasps and 162 own tests exact. Pools were resized.
- **A4-D66, S.** Sibeen dropped the Isaac test from the loop and ruled one delivery at the end. 1,532 round-1 files were moved aside.
- **A4-D67, R.** Refinement is bit-identical only on the same CPU model. 7 of 158 grasps differed in low bits between Intel and AMD hosts.
- **A4-D68, I.** Worker pools run at nice 10, so init wins the CPU.
- **A4-D69, I.** The slowest single-template inits were not split. They were GPU-bound at 8 to 11 s per batch, and a split needed an epoch-start mechanism.
- **A4-D70, I.** A sandbox test wrote real chunk markers. One premature marker was removed, and no worker reads markers.
- **A4-D71, I.** The first run was delivered. 1,583,000 passes in 3,200 files, verified against the pass lists with 0 missing and 0 extra. 4,960,000 starts, 41.8% kept, 76.4% passed.

#### Group 3. The 40-epoch continuation and its delivery, A4-D72 to A4-D114

- **A4-D72, R.** Epochs 5 to 39 ran as 4,464 epoch-range jobs with `+op.epoch_start`, 2 processes per GPU on 96 slots. The smoke job exited 0 in 68 s.
- **A4-D73, R.** A resumed job reseeds with 12 plus its start epoch after the object order is fixed. Without it every job redraws epoch 0's poses, proven by hash.
- **A4-D74, B.** The hand loader reads the pool from index files. The index equals the glob on 31 of 31 templates, and one glob took 132 s.
- **A4-D75, I.** Workers read queues from saved offsets, and the first run's delivery and markers were moved aside.
- **A4-D76, B.** The DataLoader uses forkserver after 13 of 96 inits deadlocked at fork. Start-pose hashes are unchanged.
- **A4-D77, ambiguous between I and R.** One pool index file per writing host, and the loader skips NUL lines. 8 of 248 files had NUL runs.
- **A4-D78, I.** The watchdog judges a hang by the last main-flow log line, after a deadlocked init hid behind a 60 s error line.
- **A4-D79, B.** Init saves on a saver thread. 300 files byte-equal. Saves had cost 1.04 s per batch.
- **A4-D80, I.** Workers list each input dir first, after a stale negative lookup swallowed one own test.
- **A4-D81, I.** Host affinity, so a host reads the files it wrote. DELEGRETURN fell from up to 1,285 ms to 1 to 40 ms.
- **A4-D82, B.** Init makes dirs once per batch and skips skip_done globs on a fresh job. 300 files byte-equal, and a resume skipped 3 of 3.
- **A4-D83, I.** Multi-lease scans were built and held back, then deployed by A4-D91.
- **A4-D84, I.** Workers were drained for 12 minutes during the 03:40 freezes. The freezes continued, so our workers did not cause them. Sibeen ruled to keep going.
- **A4-D85, I.** The indexer moved to kaist143, whose NFS client was not saturated.
- **A4-D86, I.** Big pools were cut on two hosts, since each client caps at about 32 calls in flight and big pools starved init.
- **A4-D87, I.** strace of a live worker counted 30 file syscalls per item, half of them hand assets.
- **A4-D88, B.** Hand XML and meshes are served from memory. Models are byte-identical, and nas4 syscalls per item fell from 24.4 to 6.4.
- **A4-D89, I.** The pool cut was extended to three more hosts at the orchestrator's request.
- **A4-D90, I.** Pools are sized by a CPU budget, threads minus init's CPU minus 10% minus 1, in steps with a GPU check.
- **A4-D91, I.** Multi-lease scans were deployed with the budget steps.
- **A4-D92, B.** One npz per init batch. About 500 NFS calls per batch fell to about 5, byte-equal on both the init and the worker side.
- **A4-D93, I.** After an indexer crash, affinity marking moved before the queue appends, and a one-time trim restart dropped 10,700 repeated lines.
- **A4-D94, I.** Pools were backed off on three hosts in a slow phase, when init GPUs fell to 1 to 29%.
- **A4-D95, I.** Eight watchdog kills in slow phases were logged, likely NFS starvation. Each costs one redone batch.
- **A4-D96, I.** The watchdog was at 1,800 s from 06:58 to 07:45 while npz batches rolled out, then back to 600 s.
- **A4-D97, B.** Object mesh pieces go in a 256 MB per-process LRU in place of routing. 305 grasps and 305 own tests equal.
- **A4-D98, I.** Pools were re-stepped toward the budget once npz batches were live. Two steps were kept and two backed off by GPU.
- **A4-D99, I.** The pre-delivery gate re-runs swallowed items and checks pass lists for repeats and missing files.
- **A4-D100, I.** Pools rise per host at init end, once every job has started and the host runs no init.
- **A4-D101, I.** Pools are bounded by memory. kaist140 ran out at 184 workers, and a worker is about 1.3 GB PSS.
- **A4-D102, I.** The indexer tolerates the owner-file race, a named exception to fail-loud, after two crashes.
- **A4-D103, I.** The gate, the delivery shards and worker restarts run detached on dj, because the VPN dropped twice.
- **A4-D104, I.** One shard was relaunched at 16 processes as a pool-size test, with a skip for entries already on disk. It was overtaken by the stop.
- **A4-D105, S.** Sibeen stopped the delivery. Our load was zero from 16:21:54, and a fresh-name probe ran on kaist143. 62 probes over 31 minutes answered in 6 to 12 ms with no freeze.
- **A4-D106, I.** The ramped resume on 09-29 started 2 hosts at 64 processes, with the probe on kaist143 and the controller `src/agent4_cap16_ramp.py`.
- **A4-D107, I.** The controller added a third host on the probe alone, before the backlog comparison existed, which was a gap against the rule. Adding then stopped on a backlog rise.
- **A4-D108, I.** The loaded clients froze at 07:00. The probe missed it, and every shard was stopped by PID.
- **A4-D109, I.** The delivery resumed at 1 host with 24 processes and a per-client freeze trigger.
- **A4-D110, I.** At 24 processes one host read 2,268 files/s with its client queue near empty, against 632 to 925 at 64.
- **A4-D111, I.** That host froze its client after 12.5 minutes, and the run stopped at level 1.
- **A4-D112, S.** Sibeen's GO put 8 hosts at 24 under per-host control. nas4 froze server-wide within 3 minutes, and every host's readers were killed by PID.
- **A4-D113, S.** Sibeen ordered AIMD control of the total, then no scripted controller at all. The AIMD controller was stopped having launched nothing.
- **A4-D114, I.** Delivered and verified. The orchestrator finished by hand at about 16 readers in total. DONE was written at 10:47:02 on 09-29, and the verify at 10:50:15 found 3,200 entries and 12,899,466 grasps, 0 missing and 0 extra.

### 3.3 Traps and numbers

#### 3.3.1 nas4 and NFS

- **Per-client limit.** Each dj host gets about 31 NFS calls in flight to nas4, the session slot count. A host's throughput is about 31 over the server's round trip. At a 4 to 6 ms round trip the fleet did 28,000 calls per second. In slow phases, with 27 to 33 ms, each host sat at 960 to 1,170. Everything on a host shares those slots. A big worker pool therefore makes that host's init wait 0.5 to 2.9 s per call and idles its GPUs.
- **Readers, about 16 in total, never sized per host.** On 09-29 one host at 24 processes read 2,268 files/s, with 6.9 calls queued, a 3 ms DELEGRETURN wait and an 8 ms OPEN round trip. At 64 it read 632 to 925 files/s behind a DELEGRETURN backlog of 4,000 to 7,700 calls. 64 readers fill every client slot and starve the delegation returns. But the 24-process host still froze its client after 12.5 minutes, and 8 hosts at 24 froze all of nas4 in 3 minutes. The finish held at about 16 readers in total, and nas4 stalled every host whenever the total went past about 16 to 20. Size a per-file reader fleet by its total load on nas4, never by CPU. The numbers are in A4-D110, A4-D111 and A4-D114.
- **Delegation recalls.** nas4 grants a read delegation on OPEN. When another host holds the file, it must recall it first, and DELEGRETURN then tracks OPEN one for one. At 1,450 to 4,400 per second, stalls showed DELEGRETURN at 36 to 1,285 ms, and nas4 fell to 11,552 calls per second. Host affinity fixed it in A4-D81, so a host reads the files it wrote. After that, DELEGRETURN took 1 to 40 ms and throughput rose 22%.
- **Freezes.** nas4 froze fleet-wide on 09-28 from 03:40 to 03:52, then from 03:55 to about 04:07 with our workers drained but init running. It froze again during the delivery, at about 15:45 to 15:55 and 16:09 to 16:18. A frozen nas4 hangs every dj ssh login while TCP still answers. The freeze also hangs other users. Only a DSM admin can see why. The delivery freezes are settled as ours, see section 1.3. The morning freezes came with our init still running, so their cause stays open.
- **How to measure.** Read `/proc/self/mountstats` per op, with no NFS traffic. In awk the columns are `$2` ops, `$7` cumulative queue ms, `$8` cumulative round trip ms and `$9` execute ms. `$6` is bytes received, which I once misread as queue time. `scripts/agent4_cap16_nfsrates.sh` prints files/s, calls/s, calls per file, calls on the wire and calls queued per host. A process-level D count hides NFS waits in threads, so use `ps -eLo stat`.
- **Probes.** A stat of a known file can be answered from the attribute cache for up to 60 s. A health probe must look up a fresh name, which always reaches the server.
- **Directory listings.** One template dir took 8 s to list, and one template's pool took 132 s. Listing the delivery's `extra/` dir under active writes took 74 s. Keep listings out of every loop.
- **Stale negative lookups.** A file another host just wrote can raise FileNotFoundError. Listing its dir first fixes it, A4-D80.
- **Appends.** Several hosts appending to one NFS file that many hosts re-read left NUL runs of 630 to 2,361 bytes in 8 of 248 pool files. Keep one writing host per file, A4-D77. Anything one host reads while another writes needs temp then rename.

#### 3.3.2 Processes and memory

- **Fork deadlock.** Upstream init forks its DataLoader workers each epoch while hand-loader threads run. Once the pool index made those threads busy at startup, 13 of 96 inits hung with one worker stuck in a futex since the fork. forkserver fixed it, A4-D76.
- **Watchdog by log line, not file age.** A deadlocked init kept writing a NUL-path error every 60 s, so a file-age test never flagged it. The watchdog now reads the last main-flow line of the pair log. Its threshold is 600 s. It was raised to 1,800 s between 06:58 and 07:45 on 09-28, because nas4 slow phases stalled saves, A4-D96. A4-D95 logs 8 kills that were likely false, and A4-D96 counts 13 in an overlapping window. Killed inits leave orphaned forkserver processes, which must be killed by PID.
- **Memory.** Measure PSS from `smaps_rollup`, not RSS. A worker child is about 1.0 GB PSS at 14 min and 1.3 GB at 3.6 h, and the parent 0.13 GB. kaist140, with 251 GB, ran out of memory at 184 workers, with swap full and init re-reading evicted pages from NFS. Recycling workers would save only about 0.3 GB each and was declined, A4-D101.
- **Upstream swallows errors.** `safe_wrapper` logs a WARNING and returns the input as if done. `operate_init` catches everything and exits 0. Only a `Finish initialization.` line proves an init finished.
- **Logs.** A process that opens a log with `>` overwrites lines another process appended. Judge init completion from each pair's own init log.
- **Indexer crash window.** A crash between a cycle's queue appends and its state save makes a restart queue that cycle again. Restart once with the `trim` argument, which drops lines already at each queue's tail. It dropped 10,700 on 09-28. Its owner-file race with the runners is tolerated on purpose by A4-D102. That is the one named exception to fail-loud.
- **Code-reading risk, unmeasured.** An npz batch killed after its rename but before its `Saved` log line is closed as crashed with 0 lines queued. The retry then skips it, since its npz exists, so those inits would never be refined. `index_state.json` counts `crash_kept`. How often this happened is unknown.
- **Shell traps.** pgrep can match its own shell's command line, so use a `[.]` in the pattern or anchor it with `^`. Never pkill. Kill by PID only. GNU `timeout` signals the whole process group. strace with `-o FILE PROG` ignores SIGINT. Bash reads scripts incrementally, so never edit a running script. The dj login shell is `sh`, so use `ssh host 'bash -s'` for bash syntax. The VPN drops, so run everything detached on the host.

#### 3.3.3 Measured costs

| Stage | Cost | Where measured |
| --- | --- | --- |
| Init, 10 objects per batch | 5.1 s warm on one 3090 core. At buffer 1024 an evolved template is GPU-bound at 6 to 8 s. Filter 4.65 s, matcher 3.38 s, NFS saves 1.04 s before async saves | kaist140 profile, A4-D69, A4-D79 |
| Refinement | 0.12 CPU-s per start, 0.07 to 0.09 of it rebuilding the model before the cache | yj-4 and kaist140 profiles |
| Own test | 0.56 to 0.64 CPU-s per grasp | same |
| Init throughput | 2.88 batches/s before npz batches, 10.45 after, by finished jobs | 07:46 on 09-28 |
| Delivery read | 295 to 328 files/s per host at 64 processes, 2,507 fleet-wide, about 4.5 calls per file | window A on 09-28 |

**Sizing a new object set.** Starts are objects times templates times epochs times 10. Init batches are about that number divided by 100, since a batch holds 10 objects with 10 starts each. Refinement keeps about 42% of starts, and the own test passes about 77% of those. The continuation ran 351,540 init batches in 13h 7m on 48 GPUs. That is about 560 batches per GPU-hour over the whole init, and about 780 at the post-npz rate of 10.45 batches per second. After init, about 710 worker processes refined about 880 starts per second while also running own tests. Scale the wall clock from the phases in section 1.1.

#### 3.3.4 Builds kept for raw-mesh object sets

cap16 did not need these, but a set with only raw meshes would. MeshProcess is `https://github.com/JYChen18/MeshProcess` at 638d777. It needs CoACD from the JYChen18 fork at `2e4842cec9aed2e25d5766e654eb02cc62c1769c` and ACVD at `b65c74ab3f4d73e0df1936c1df00bb249e66f14d`. `scripts/agent4_build_meshprocess.sh` builds both on yj-4.
- **CoACD.** It builds in conda env `graspoc_meshbuild` with `cmake>=3.24,<4`, since cmake 4 rejects the zlib dependency. The command is `make main`.
- **ACVD.** It builds in `graspoc_acvdbuild` with `vtk-base=9.2.6`, `vtk-io-ffmpeg=9.2.6`, `libgl-devel` and `libboost-devel=1.85.0`. It also needs a sysroot shim, which links `libGL.so` and the include dir under a fake `CONDA_BUILD_SYSROOT`.
- **Run.** MeshProcess imports torch, so it runs from a `--system-site-packages` venv with CPU torch. Its official task normalizes each mesh. A scene cfg must undo that, with scale equal to half the bbox diagonal and pose at the bbox centre, as `src/agent4_scene_cfg.py` does. Its CoACD seed is random, so outputs are not reproducible.

### 3.4 Where every artifact is

| What | Where |
| --- | --- |
| All run code | `src/agent4_cap16_*.py`, `scripts/agent4_cap16_*.sh`, deployed flat in `$S/` |
| Decisions register | `status/decisions/agent4.json`, A4-D1 to A4-D114 |
| Progress log | `$S/progress.txt`, one line per 5 minutes from 09-27 22:55 |
| Pass lists | `$S/gsched/pass/<chunk>__<tmpl>.txt`, 248 files, 1.7 GB |
| Grasp files | `$D/Dexonomy/output/cap16_<chunk>_xhand/grasp_data/`, and `succ_grasp/` links for passes |
| Per-type counts | `$S/pass_counts/counts.csv`, `vis/image/cap16_pass_counts.png` |
| First-run delivery | `$S/delivered_5epoch/` and kait `draftrec/_graspoc/cap16/candidates/dexonomy/` |
| 40-epoch delivery | `$D/cap16_deliver/dexonomy/`, 3,200 entries plus `extra/`, 14 GB, DONE. Shard records and `verify.out` are in `$S/deliver_final/` |
| Template package | `src/astra_rules_corrected_xhand_port/`, tarball and checksums as in section 2.1.3 |
| Hand port | `src/agent4_xhand_port/`, installed at `$D/Dexonomy/assets/hand/xhand/` |
| nas4 probe | `/tmp/a4_nas4probe.sh` and `/tmp/a4_nas4probe.log` on kaist143, stopped by PID at the end |
| Delivery controllers, not used at the end | `src/agent4_cap16_ramp.py`, `src/agent4_cap16_hostctl.py`, `src/agent4_cap16_aimd.py` |
