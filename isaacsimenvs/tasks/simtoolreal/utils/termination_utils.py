"""Termination and success-tolerance curriculum helpers."""

from __future__ import annotations

import torch

from .reset_utils import reset_goal_trackers


def update_tolerance_curriculum(env) -> None:
    """Shrink success tolerance when completed episodes average enough goals."""
    env._frame_counter += 1
    term = env.cfg.termination
    if term.tolerance_curriculum_per_object:
        _update_tolerance_per_object(env)
    elif env._frame_counter - env._last_curriculum_update >= term.tolerance_curriculum_interval:
        successes = env._prev_episode_successes.float()
        eligible_mask = None
        if hasattr(env, "_curriculum_eligible_mask"):
            eligible_mask = env._curriculum_eligible_mask()
        if eligible_mask is not None:
            successes = successes[eligible_mask]

        threshold = term.tolerance_curriculum_success_threshold
        if hasattr(env, "_curriculum_success_threshold"):
            custom_threshold = env._curriculum_success_threshold()
            if custom_threshold is not None:
                threshold = float(custom_threshold)

        if successes.numel() > 0 and successes.mean().item() >= threshold:
            new_tol = env._current_success_tolerance * term.tolerance_curriculum_increment
            new_tol = max(min(new_tol, term.success_tolerance), term.target_success_tolerance)
            env._current_success_tolerance = new_tol
            env._last_curriculum_update = env._frame_counter

    # Eval pins the success criterion.
    if term.eval_success_tolerance is not None:
        env._current_success_tolerance = float(term.eval_success_tolerance)
        if term.tolerance_curriculum_per_object:
            env._object_success_tolerance.fill_(float(term.eval_success_tolerance))


def _update_tolerance_per_object(env) -> None:
    """Tighten each object whose own envs' last episodes average enough goals, at most once per interval."""
    term = env.cfg.termination
    obj = env._object_asset_index_per_env
    n_obj = env._object_success_tolerance.numel()
    count = torch.bincount(obj, minlength=n_obj)
    mean = torch.bincount(obj, weights=env._prev_episode_successes.float(), minlength=n_obj) / count
    due = env._frame_counter - env._object_last_curriculum_update >= term.tolerance_curriculum_interval
    tighten = due & (count > 0) & (mean >= term.tolerance_curriculum_success_threshold)
    tol = env._object_success_tolerance
    new_tol = (tol * term.tolerance_curriculum_increment).clamp(term.target_success_tolerance, term.success_tolerance)
    env._object_success_tolerance = torch.where(tighten, new_tol, tol)
    env._object_last_curriculum_update = torch.where(
        tighten, torch.full_like(env._object_last_curriculum_update, env._frame_counter),
        env._object_last_curriculum_update,
    )
    env._current_success_tolerance = float(env._object_success_tolerance.mean())


def get_curriculum_state(env) -> dict:
    state = {
        "frame_counter": env._frame_counter,
        "last_curriculum_update": env._last_curriculum_update,
        "current_success_tolerance": env._current_success_tolerance,
    }
    if env.cfg.termination.tolerance_curriculum_per_object:
        state["object_success_tolerance"] = env._object_success_tolerance.cpu()
        state["object_last_curriculum_update"] = env._object_last_curriculum_update.cpu()
    return state


def set_curriculum_state(env, state: dict) -> None:
    assert state is not None, "the checkpoint has no env_state, it was saved before the curriculum was"
    per_object = env.cfg.termination.tolerance_curriculum_per_object
    assert ("object_success_tolerance" in state) == per_object, (sorted(state), per_object)
    env._frame_counter = state["frame_counter"]
    env._last_curriculum_update = state["last_curriculum_update"]
    env._current_success_tolerance = state["current_success_tolerance"]
    if per_object:
        tol = state["object_success_tolerance"]
        assert tol.shape == env._object_success_tolerance.shape, (tol.shape, env._object_success_tolerance.shape)
        env._object_success_tolerance = tol.to(env.device)
        env._object_last_curriculum_update = state["object_last_curriculum_update"].to(env.device)
        print(f"[curriculum] restored per object, frame {env._frame_counter}, tolerance min "
              f"{tol.min().item():.5f} median {tol.median().item():.5f} max {tol.max().item():.5f}", flush=True)
    else:
        print(f"[curriculum] restored, frame {env._frame_counter}, tolerance "
              f"{env._current_success_tolerance:.5f}", flush=True)


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Update goal-hit state and return ``(terminated, truncated)``."""
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins
    is_success = env._is_success

    # Authoritative updates on goal-hit.
    env._successes = env._successes + is_success.long()
    goal_reset_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
    if goal_reset_ids.numel() > 0:
        reset_goal_trackers(env, goal_reset_ids)
        # zero the length buf so truncation doesn't fire
        env.episode_length_buf[goal_reset_ids] = 0

    # Termination causes.
    object_z_local = env.object.data.root_pos_w[:, 2] - env_origins[:, 2]
    fall = object_z_local < 0.1

    if term_cfg.max_consecutive_successes > 0:
        max_successes_reached = env._successes >= term_cfg.max_consecutive_successes
    else:
        max_successes_reached = torch.zeros_like(fall)

    hand_far = env._curr_fingertip_distances.max(dim=-1).values > 1.5

    terminated = fall | max_successes_reached | hand_far
    truncated = env.episode_length_buf >= env.max_episode_length
    env._termination_reasons = {
        "fall": fall,
        "max_successes": max_successes_reached,
        "hand_far": hand_far,
        "timeout": truncated,
    }
    return terminated, truncated


__all__ = ["update_tolerance_curriculum", "compute_terminations"]
