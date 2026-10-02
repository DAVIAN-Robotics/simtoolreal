"""Task metric publishing for SimToolReal."""

from __future__ import annotations

import torch


def _size_bands(env) -> dict:
    """Envs split into thirds by their object's keypoint corner radius."""
    if not hasattr(env, "_size_band_masks"):
        rew = env.cfg.reward
        radius = torch.linalg.norm(
            env._object_scale_per_env * rew.object_base_size * rew.keypoint_scale * 0.5,
            dim=-1,
        )
        lo, hi = torch.quantile(radius, torch.tensor([1 / 3, 2 / 3], device=radius.device))
        env._size_band_masks = {
            "small": radius <= lo,
            "mid": (radius > lo) & (radius <= hi),
            "large": radius > hi,
        }
    return env._size_band_masks


def log_step_metrics(env) -> None:
    """Publish step-level extras consumed by RL-Games observers."""
    term_cfg = env.cfg.termination
    if term_cfg.max_consecutive_successes > 0:
        all_goals_hit = env._successes >= term_cfg.max_consecutive_successes
    else:
        all_goals_hit = torch.zeros_like(env._successes, dtype=torch.bool)

    episode_final = {
        "successes": env._successes.float(),
        "all_goals_hit": all_goals_hit.float(),
    }
    episode_final.update(
        {
            f"done_{name}": value.float()
            for name, value in env._termination_reasons.items()
        }
    )

    # The gate measures the object's own corners, so its angular tolerance scales
    # with the object. One shared success_tolerance advances on the mean across
    # every env, which lets small objects carry the curriculum while large ones
    # never clear it. Split the same successes by object size so that shows.
    successes = env._prev_episode_successes.float()
    for name, mask in _size_bands(env).items():
        episode_final[f"successes_{name}"] = successes[mask]

    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = successes
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)
    if term_cfg.tolerance_curriculum_per_object:
        tol = env._object_success_tolerance
        env.extras["current_success_tolerance_min"] = float(tol.min())
        env.extras["current_success_tolerance_median"] = float(tol.median())
        env.extras["current_success_tolerance_max"] = float(tol.max())
        env.extras["current_success_tolerance_is_max"] = float(
            (tol == term_cfg.success_tolerance).float().mean()
        )
        env.extras["current_success_tolerance_is_min"] = float(
            (tol == term_cfg.target_success_tolerance).float().mean()
        )


__all__ = ["log_step_metrics"]
