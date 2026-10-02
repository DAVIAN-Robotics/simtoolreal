"""State allocation and reset helpers for SimToolReal."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import math

import torch

from isaaclab.utils.math import quat_apply, random_orientation

from .action_utils import sample_log_uniform
from .goal_sampling import sample_absolute_goal_pose, sample_delta_goal_pose
from .obs_utils import KEYPOINT_CORNERS, NUM_FINGERTIPS

# Dexonomy's squeeze pose is the grasp plus the contact torques over this gain, the gainprm of every
# XHand actuator in its assets/hand/xhand/right.xml.
DEXONOMY_HAND_KP = 3.0


def _load_reset_states(env) -> None:
    """Every pool object's grasped states, stacked (objects, states, ...) on the device. The squeeze
    target is rescaled from Dexonomy's gain to this hand's stiffness and clamped to the joint limits."""
    import numpy as np

    n = env.cfg.reset.reset_state_count or None
    columns = {k: [] for k in ("obj_pos", "obj_quat_wxyz", "arm_q", "hand_q", "hand_target")}
    for urdf in env._object_urdf_paths:
        with np.load(Path(env.cfg.reset.reset_state_dir) / f"{Path(urdf).stem}.npz") as f:
            for key, rows in columns.items():
                rows.append(f[key][:n])
            arm_joints, hand_joints = list(f["arm_joints"]), list(f["hand_joints"])
    names = list(env.robot.data.joint_names)
    env._rs_arm_ids = torch.tensor([names.index(j) for j in arm_joints], device=env.device)
    env._rs_hand_ids = torch.tensor([names.index(j) for j in hand_joints], device=env.device)

    def stack(key):
        return torch.tensor(np.stack(columns[key]), dtype=torch.float32, device=env.device)

    env._rs_obj_pos = stack("obj_pos")
    env._rs_obj_quat = stack("obj_quat_wxyz")
    env._rs_arm_q = stack("arm_q")
    env._rs_hand_q = stack("hand_q")
    kp = torch.tensor([env.cfg.robot.hand_stiffness[j] for j in hand_joints], device=env.device)
    lo, hi = env.robot.data.joint_pos_limits[0, env._rs_hand_ids].unbind(-1)
    squeeze = env._rs_hand_q + (stack("hand_target") - env._rs_hand_q) * DEXONOMY_HAND_KP / kp
    env._rs_hand_target = torch.maximum(torch.minimum(squeeze, hi), lo)

    if env.cfg.reset.goal_fingertip_dir:
        names = list(env._fingertip_body_names)
        ft = []
        for urdf in env._object_urdf_paths:
            with np.load(Path(env.cfg.reset.goal_fingertip_dir) / f"{Path(urdf).stem}.npz") as f:
                bodies = [str(b) for b in f["bodies"]]
                ft.append(f["ft_obj"][:n][:, [bodies.index(b) for b in names]])
        env._rs_ft_obj = torch.tensor(np.stack(ft), dtype=torch.float32, device=env.device)
        assert env._rs_ft_obj.shape[:2] == env._rs_obj_pos.shape[:2], (env._rs_ft_obj.shape, env._rs_obj_pos.shape)
        env._goal_row = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._goal_ft = torch.zeros(env.num_envs, NUM_FINGERTIPS, 3, device=env.device)
        env._closest_fingertip_max_dist = torch.full((env.num_envs,), -1.0, device=env.device)


def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer + index cache used by the hooks.

    Called once from ``__init__`` after ``super().__init__`` (which runs
    ``_setup_scene`` and makes ``env.robot``/``env.object``/``env.goal_viz``
    available).
    """
    dr = env.cfg.domain_randomization
    rew = env.cfg.reward

    # --- Joint/body id caches ---
    env._arm_joint_ids = env.robot.find_joints(env.cfg.robot.arm_joint_regex)[0]      # 7
    env._hand_joint_ids = env.robot.find_joints(env.cfg.robot.hand_joint_regex)[0]     # 22
    env._palm_body_id = env.robot.find_bodies(env.cfg.robot.palm_body)[0][0]
    _ft_ids, _ft_names = env.robot.find_bodies(env.cfg.robot.fingertip_body_regex)
    env._fingertip_body_ids = _ft_ids  # 5
    env._fingertip_body_names = list(_ft_names)  # 009: same order as ids

    # Convert between Lab parser order and canonical policy order.
    lab_names = list(env.robot.data.joint_names)
    env._perm_canon_to_lab = torch.tensor(
        [env.cfg.robot.joint_order.index(n) for n in lab_names],
        device=env.device, dtype=torch.long,
    )
    env._perm_lab_to_canon = torch.tensor(
        [lab_names.index(n) for n in env.cfg.robot.joint_order],
        device=env.device, dtype=torch.long,
    )

    limits = env.robot.data.joint_pos_limits  # (N, num_joints, 2), Lab order

    # Canonical-order limits for normalizing joint_pos observations.
    env._joint_lower_canon = limits[0, :, 0][env._perm_lab_to_canon]  # (29,)
    env._joint_upper_canon = limits[0, :, 1][env._perm_lab_to_canon]  # (29,)

    # Lab-order limits for action target clamping.
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    # --- Action target buffers  ---
    action_space = env.cfg.action_space
    env._cur_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._prev_targets = torch.zeros(env.num_envs, action_space, device=env.device)

    # --- Keypoint offsets in object local frame ---
    corners = torch.tensor(
        KEYPOINT_CORNERS, device=env.device, dtype=torch.float32
    )  # (8, 3)

    env._keypoint_offsets = (
        corners.unsqueeze(0)
        * (env._object_scale_per_env * rew.object_base_size * rew.keypoint_scale * 0.5).unsqueeze(1)
    )  # (N, 8, 3)

    # --- Per-env DR priors (re-sampled on reset; seeded once here) ---
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(
        env.num_envs, 3, device=env.device
    ).uniform_(lo, hi)

    # --- Reward / termination trackers  ---
    # whether the object is lifted
    env._lifted_object = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    # the maximum distance between the object and the goal since the last goal reset
    env._closest_keypoint_max_dist = torch.full(
        (env.num_envs,), -1.0, device=env.device
    )
    # the minimum distance between a fingertip and the object since the last goal reset
    env._closest_fingertip_dist = torch.full(
        (env.num_envs, NUM_FINGERTIPS), -1.0, device=env.device
    )
    # the number of succeesses in the current episode
    env._successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    # the number of consecutive steps that the object is near the goal
    env._near_goal_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )

    # --- Tolerance curriculum state ---
    env._current_success_tolerance: float = env.cfg.termination.success_tolerance
    env._prev_episode_successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._frame_counter: int = 0
    env._last_curriculum_update: int = 0
    if env.cfg.termination.tolerance_curriculum_per_object:
        n_obj = len(env._object_urdf_paths)
        env._object_success_tolerance = torch.full(
            (n_obj,), env.cfg.termination.success_tolerance, device=env.device
        )
        env._object_last_curriculum_update = torch.zeros(n_obj, dtype=torch.long, device=env.device)

    # --- Object lifted-reward reference z (updated on each _reset_object_pose) ---
    init_z = env.cfg.reset.table_reset_z + env.cfg.reset.table_object_z_offset
    env._object_init_z = torch.full(
        (env.num_envs,), init_z, device=env.device
    )

    # --- Table half height, from the urdf the scene spawns ---
    env._table_half_height_m = float(
        ElementTree.parse(Path(env.cfg.assets.table_urdf))
        .getroot()
        .find(".//collision/geometry/box")
        .get("size")
        .split()[2]
    ) * 0.5

    if env.cfg.reset.reset_state_dir:
        _load_reset_states(env)

    # --- Per-env table surface z (randomized in _reset_table_pose) ---
    env._table_z_per_env = torch.full(
        (env.num_envs,), env.cfg.reset.table_reset_z, device=env.device
    )

    # --- DR rolling buffers ---
    env._object_state_queue = torch.zeros(
        env.num_envs,
        max(1, dr.object_state_delay_max),
        13,  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        device=env.device,
    )
    env._obs_queue = torch.zeros(
        env.num_envs,
        max(1, dr.obs_delay_max),
        env.cfg.observation_space,
        device=env.device,
    )
    env._action_queue = torch.zeros(
        env.num_envs,
        max(1, dr.action_delay_max),
        action_space,
        device=env.device,
    )

    # --- Wrench DR state (Phase C/D) ---
    env._random_force_prob = sample_log_uniform(
        dr.force_prob_range, env.num_envs
    ).to(env.device)
    env._random_torque_prob = sample_log_uniform(
        dr.torque_prob_range, env.num_envs
    ).to(env.device)
    env._object_forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_torques = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_mass = env.object.data.default_mass[:, 0:1].to(env.device)  # (N, 1)

    # --- Step-shared caches populated by compute_intermediate_values (Phase F) ---
    env._keypoints_max_dist = torch.zeros(env.num_envs, device=env.device)
    env._curr_fingertip_distances = torch.zeros(
        env.num_envs, NUM_FINGERTIPS, device=env.device
    )
    env._near_goal = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._is_success = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # --- Fixed-trajectory pool (trajectory_count ablation) ---
    # Loaded once at init from the JSON file. Different runs share the same
    # file and pick the first N via cfg.reset.fixed_trajectory_count
    # (0 = take all). Empty filename = ablation disabled (baseline path).
    if env.cfg.reset.fixed_trajectory_file:
        path = Path(env.cfg.reset.fixed_trajectory_file)
        with open(path) as f:
            payload = json.load(f)
        all_pos = torch.tensor(payload["pos"], device=env.device, dtype=torch.float32)
        all_quat = torch.tensor(payload["quat_wxyz"], device=env.device, dtype=torch.float32)
        n_total = all_pos.shape[0]
        n_take = env.cfg.reset.fixed_trajectory_count or n_total
        if n_take > n_total:
            raise ValueError(
                f"fixed_trajectory_count={n_take} exceeds pool size {n_total} in {path}"
            )
        env._fixed_traj_pos = all_pos[:n_take].contiguous()    # (N, K, 3)
        env._fixed_traj_quat = all_quat[:n_take].contiguous()  # (N, K, 4)
        env._traj_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._traj_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # set the reward buffer to 0
    env.reward_buf = torch.zeros(env.num_envs, device=env.device)


def _randomize_robot_dof_state(env, env_ids: torch.Tensor) -> None:
    """Reset DOF state and seed previous targets from the reset pose."""
    cfg = env.cfg.reset
    default_pos = env.robot.data.default_joint_pos[env_ids]  # (n, num_dofs)
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]

    reset_scale = torch.zeros_like(default_pos)
    reset_scale[:, env._arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
    reset_scale[:, env._hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

    sampled_pos = lower + (upper - lower) * torch.rand_like(default_pos)
    joint_pos = torch.lerp(default_pos, sampled_pos, reset_scale).clamp(lower, upper)
    joint_vel = torch.empty_like(default_pos).uniform_(
        -cfg.reset_dof_vel_random_interval,
        cfg.reset_dof_vel_random_interval,
    )

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos


def _reset_table_pose(env, env_ids: torch.Tensor) -> None:
    """Randomize the table's pose per env and write the new transform.

    Z position is required (the policy uses `_table_z_per_env` to set the
    object init height). XY position and yaw are gated on
    `table_reset_xy_range_m` / `table_reset_yaw_range_deg` — default ranges
    are zero so this is a no-op for runs that don't opt in.
    """
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    dz = torch.empty(n, device=env.device).uniform_(
        -cfg.table_reset_z_range, cfg.table_reset_z_range
    )
    table_z = cfg.table_reset_z + dz
    env._table_z_per_env[env_ids] = table_z

    pos_local = torch.zeros(n, 3, device=env.device)
    pos_local[:, 2] = table_z

    # XY position noise — per-env independent uniform half-widths.
    xy_range = tuple(float(v) for v in cfg.table_reset_xy_range_m)
    if xy_range[0] > 0.0 or xy_range[1] > 0.0:
        rx = torch.empty(n, device=env.device).uniform_(-xy_range[0], xy_range[0])
        ry = torch.empty(n, device=env.device).uniform_(-xy_range[1], xy_range[1])
        pos_local[:, 0] = rx
        pos_local[:, 1] = ry

    # Yaw noise — sample uniform [-r, r] degrees, build a z-axis rotation quat.
    yaw_range_deg = float(cfg.table_reset_yaw_range_deg)
    if yaw_range_deg > 0.0:
        yaw_rad = (
            torch.empty(n, device=env.device).uniform_(-1.0, 1.0)
            * yaw_range_deg * (torch.pi / 180.0)
        )
        half = yaw_rad * 0.5
        w = torch.cos(half)
        z = torch.sin(half)
        quat = torch.stack([w, torch.zeros_like(w), torch.zeros_like(w), z], dim=-1)
    else:
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.table.write_root_pose_to_sim(pose, env_ids=env_ids)

def box_gap(centre, rot, half, lo, hi):
    """Separation between oriented boxes at `centre`/`rot`/`half` and an axis-aligned box.

    Separating axis over the three world axes and the three object axes. Positive means the two are
    apart, negative means they overlap by that much on the tightest axis.
    """
    box_c = (lo + hi) * 0.5
    box_h = (hi - lo) * 0.5
    d = box_c - centre
    obj_w = (rot.abs() * half.unsqueeze(-2)).sum(dim=-1)
    gap_w = d.abs() - obj_w - box_h
    d_o = (rot * d.unsqueeze(-1)).sum(dim=-2)
    box_o = (rot.abs() * box_h.unsqueeze(-1)).sum(dim=-2)
    gap_o = d_o.abs() - half - box_o
    return torch.maximum(gap_w.amax(-1), gap_o.amax(-1))


def _hand_box(env):
    """The hand's box at the reset pose, env-local, grown by the margin.

    A constant. The base pose and the reset joint angles are both fixed, so the hand sits in the
    same place on every reset in every env. The numbers are in robots.py.
    """
    if not hasattr(env, "_hand_box_local"):
        lo, hi = env.cfg.robot.hand_box_local
        m = float(env.cfg.reset.hand_clearance_margin)
        env._hand_box_local = (
            torch.tensor(lo, dtype=torch.float32, device=env.device) - m,
            torch.tensor(hi, dtype=torch.float32, device=env.device) + m,
        )
    return env._hand_box_local


def _sample_clear_start(env, env_ids: torch.Tensor):
    """Draw start poses that clear the table and the hand.

    Each candidate carries its own position and its own orientation. Holding either fixed leaves
    large objects with no admissible pose, measured on the ManipTrans pool.
    """
    from isaaclab.utils.math import matrix_from_quat

    cfg = env.cfg.reset
    n = env_ids.numel()
    k = cfg.start_pose_candidates

    noise = torch.empty(n, k, 2, device=env.device).uniform_(-1.0, 1.0)
    xy = torch.stack(
        (noise[..., 0] * cfg.reset_position_noise_x, noise[..., 1] * cfg.reset_position_noise_y),
        dim=-1,
    )
    quat = random_orientation(n * k, device=env.device).view(n, k, 4)
    rot = matrix_from_quat(quat.reshape(-1, 4)).view(n, k, 3, 3)

    # The lowest point of the object's box below its centre, in the drawn orientation.
    half = env._object_scale_per_env[env_ids] * (env.cfg.reward.object_base_size * 0.5)
    support = (rot[..., 2, :].abs() * half.unsqueeze(1)).sum(dim=-1)
    surface = (env._table_z_per_env[env_ids] + env._table_half_height_m).unsqueeze(1)
    land_z = surface + support
    z = land_z + cfg.object_drop_height
    centre = torch.cat([xy, z.unsqueeze(-1)], dim=-1)

    lo, hi = _hand_box(env)
    clearance = box_gap(centre, rot, half.view(n, 1, 3).expand(n, k, 3), lo, hi)

    # First survivor, and the least-bad candidate where nothing survives. Candidates are drawn
    # identically, so taking the first keeps the start distribution unbiased. A least-bad pose
    # gets ejected and falls off the table, so the episode resets and the draw costs nothing.
    survives = clearance > 0.0
    first = torch.argmax(survives.float(), dim=1)
    pick = torch.where(survives.any(dim=1), first, clearance.argmax(dim=1))

    rows = torch.arange(n, device=env.device)
    return centre[rows, pick], quat[rows, pick], land_z[rows, pick]


def _reset_object_pose(env, env_ids: torch.Tensor) -> None:
    """Reset object pose and lifted-reward reference height."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_start_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_start_pose, device=env.device, dtype=torch.float32)
        pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        quat = fixed[3:].unsqueeze(0).expand(n, -1)
        land_z = pos_local[:, 2]
    else:
        pos_local, quat, land_z = _sample_clear_start(env, env_ids)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )

    env._object_init_z[env_ids] = pos_local[:, 2]


def _reset_goal_pose(env, env_ids: torch.Tensor, mode: str) -> None:
    """Resample the goal pose and write it to GoalViz."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_goal_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_goal_pose, device=env.device, dtype=torch.float32)
        new_pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        new_quat = fixed[3:].unsqueeze(0).expand(n, -1)
        pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        return

    # Fixed-trajectory ablation: ignore goal_sampling_type, draw from the
    # pre-loaded pool. mode acts as the "hard reset vs intra-episode" signal:
    #   "absolute" → fresh episode → pick new traj_id, reset step to 0
    #   anything else → intra-episode goal-hit → advance step
    # The K-th success briefly calls this with step=K (out of bounds) before
    # max_consecutive_successes terminates the episode; the clamp keeps the
    # lookup safe — the goal we write is then immediately overwritten by the
    # full reset.
    if cfg.fixed_trajectory_file:
        K = env._fixed_traj_pos.shape[1]
        if mode == "absolute":
            n_traj = env._fixed_traj_pos.shape[0]
            env._traj_id[env_ids] = torch.randint(
                0, n_traj, (n,), device=env.device, dtype=torch.long,
            )
            env._traj_step[env_ids] = 0
        else:
            env._traj_step[env_ids] = env._traj_step[env_ids] + 1

        traj_id = env._traj_id[env_ids]
        step_clamped = env._traj_step[env_ids].clamp(max=K - 1)
        new_pos_local = env._fixed_traj_pos[traj_id, step_clamped]
        new_quat = env._fixed_traj_quat[traj_id, step_clamped]
        pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        return

    if mode == "reset_state":
        _sample_goal_state(env, env_ids)
        return

    if mode == "delta":
        prev_pos_local = env.goal_viz.data.root_pos_w[env_ids] - env_origins
        prev_quat = env.goal_viz.data.root_quat_w[env_ids]
        new_pos_local, new_quat = sample_delta_goal_pose(
            prev_pos=prev_pos_local,
            prev_quat_wxyz=prev_quat,
            delta_distance=cfg.delta_goal_distance,
            delta_rotation_degrees=cfg.delta_rotation_degrees,
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
        )
    elif mode == "absolute":
        new_pos_local, new_quat = sample_absolute_goal_pose(
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
            n_envs=n,
            device=env.device,
        )
    else:
        raise ValueError(f"unknown goal sampling mode: {mode}")

    new_pos_local = _clear_table(env, env_ids, new_pos_local, new_quat)

    pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)


def _clear_table(env, env_ids, pos_local, quat_wxyz):
    """Raise a sampled goal until the object clears the table at that orientation.

    The sampler draws position uniformly and orientation uniformly over SO(3) with no geometry in
    the loop, so 2.4 percent of goals put part of the object below the table surface, measured over
    the cap16 pool. Those goals are unreachable and only add noise.

    ``_table_z_per_env`` holds the table's centre, not its surface. The URDF box is 0.3 m tall, so
    the surface is 0.15 m above it. The cfg comment claiming table_reset_z is the surface is wrong.

    The lowest point of a box below its centre is the support function along -z, sum over axes of
    |R[2,i]| times the half extent. object_scale_per_env is extent_m / object_base_size, so the
    metric half extent is that times object_base_size / 2.
    """
    from isaaclab.utils.math import matrix_from_quat

    half = env._object_scale_per_env[env_ids] * (env.cfg.reward.object_base_size * 0.5)
    r_z = matrix_from_quat(quat_wxyz)[:, 2, :]
    support = (r_z.abs() * half).sum(dim=-1)
    table_top = env._table_z_per_env[env_ids] + env._table_half_height_m

    out = pos_local.clone()
    out[:, 2] = torch.maximum(out[:, 2], table_top + support)
    return out


def _clear_goal_trackers(env, env_ids: torch.Tensor) -> None:
    env._closest_keypoint_max_dist[env_ids] = -1.0
    env._closest_fingertip_dist[env_ids] = -1.0
    if env.cfg.reset.goal_fingertip_dir:
        env._closest_fingertip_max_dist[env_ids] = -1.0
    env._near_goal_steps[env_ids] = 0


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """Clear per-goal trackers and sample the next goal."""
    _clear_goal_trackers(env, env_ids)
    _reset_goal_pose(env, env_ids, mode=env.cfg.reset.goal_sampling_type)


def _set_goal_state(env, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
    """Make reset state `rows` of each env's object its goal: the pose to GoalViz, its fingertip points placed there."""
    asset = env._object_asset_index_per_env[env_ids]
    pos_local = env._rs_obj_pos[asset, rows]
    quat = env._rs_obj_quat[asset, rows]
    env.goal_viz.write_root_pose_to_sim(
        torch.cat([pos_local + env.scene.env_origins[env_ids], quat], dim=-1), env_ids=env_ids
    )
    env._goal_row[env_ids] = rows
    env._goal_ft[env_ids] = quat_apply(
        quat.unsqueeze(1).expand(-1, NUM_FINGERTIPS, -1), env._rs_ft_obj[asset, rows]
    ) + pos_local.unsqueeze(1)


def _sample_goal_state(env, env_ids: torch.Tensor) -> None:
    """Next goal: uniform among the object's reset states inside the delta window around the current goal state."""
    cfg = env.cfg.reset
    asset = env._object_asset_index_per_env[env_ids]
    cur = env._goal_row[env_ids]
    p, q = env._rs_obj_pos[asset], env._rs_obj_quat[asset]  # (n, states, 3), (n, states, 4)
    in_pos = ((p - env._rs_obj_pos[asset, cur].unsqueeze(1)).abs() <= cfg.delta_goal_distance).all(dim=-1)
    cos_half = math.cos(math.radians(cfg.delta_rotation_degrees) / 2.0)
    in_rot = (q * env._rs_obj_quat[asset, cur].unsqueeze(1)).sum(dim=-1).abs() >= cos_half
    rows = torch.multinomial((in_pos & in_rot).float(), 1).squeeze(-1)
    _set_goal_state(env, env_ids, rows)


def _reset_grasped(env, env_ids: torch.Tensor) -> None:
    """Start from a random one of the object's grasped states, with the start pose as the first goal."""
    n = env_ids.numel()
    asset = env._object_asset_index_per_env[env_ids]
    k = torch.randint(0, env._rs_obj_pos.shape[1], (n,), device=env.device)

    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, env._rs_arm_ids] = env._rs_arm_q[asset, k]
    joint_pos[:, env._rs_hand_ids] = env._rs_hand_q[asset, k]
    env.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    targets = joint_pos.clone()
    targets[:, env._rs_hand_ids] = env._rs_hand_target[asset, k]
    env._prev_targets[env_ids] = targets
    env._cur_targets[env_ids] = targets

    pos_local = env._rs_obj_pos[asset, k]
    pose = torch.cat([pos_local + env.scene.env_origins[env_ids], env._rs_obj_quat[asset, k]], dim=-1)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
    if env.cfg.reset.goal_fingertip_dir:
        _set_goal_state(env, env_ids, k)
    else:
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
    if env.cfg.reset.first_goal_random:
        _reset_goal_pose(env, env_ids, mode="delta")
    env._object_init_z[env_ids] = pos_local[:, 2]


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full per-env reset after ``super()._reset_idx``."""
    n = env_ids.numel()

    if env.cfg.reset.reset_state_dir:
        _reset_table_pose(env, env_ids)
        _reset_grasped(env, env_ids)
    else:
        _randomize_robot_dof_state(env, env_ids)
        _reset_table_pose(env, env_ids)
        _reset_object_pose(env, env_ids)
        _reset_goal_pose(env, env_ids, mode="absolute")  # full reset → always absolute

    env._prev_episode_successes[env_ids] = env._successes[env_ids]

    _clear_goal_trackers(env, env_ids)
    env._lifted_object[env_ids] = bool(env.cfg.reset.reset_state_dir)
    env._successes[env_ids] = 0

    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    for queue_name in ("_student_camera_queue", "_student_obs_queue"):
        if hasattr(env, queue_name):
            getattr(env, queue_name)[env_ids] = 0.0
    env._object_forces[env_ids] = 0.0
    env._object_torques[env_ids] = 0.0

    dr = env.cfg.domain_randomization
    env._random_force_prob[env_ids] = sample_log_uniform(
        dr.force_prob_range, n
    ).to(env.device)
    env._random_torque_prob[env_ids] = sample_log_uniform(
        dr.torque_prob_range, n
    ).to(env.device)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(
        n, 3, device=env.device
    ).uniform_(lo, hi)

    # Camera pose randomization (no-op when cfg.student_obs.use_camera_pose_rand is False).
    from .scene_utils import _apply_camera_pose_rand_at_reset
    _apply_camera_pose_rand_at_reset(env, env_ids)


__all__ = [
    "allocate_state_buffers",
    "reset_env_state",
    "reset_goal_trackers",
]
