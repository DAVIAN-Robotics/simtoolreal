"""EigenDEXplore exploration settings in the policy's canonical action order, from a hand basis npz.

The npz holds joint_names (our XHand joint names), components (PCA rows over those joints, radians, highest variance
first), component_std (radians per row) and k (rows kept for 90% retained variance). Each hand column is divided by
that joint's half range from the env's own joint_pos_limits, the same map apply_action_pipeline uses to turn a hand
action in [-1, 1] into a target, so the basis is in normalized action units. Arm columns get no eigen noise. The
noise settings are the eigendexplore README's: hand IID sigma sqrt(0.5), eigen RMS sqrt(0.5), spectrum power 1, and
the arm keeps IID sigma 1 as in its policy_noise.
"""
import math

import numpy as np
import torch

from eigendexplore import EigenDExplore

IID_HAND = math.sqrt(0.5)
EIGEN_RMS = math.sqrt(0.5)
SPECTRUM_POWER = 1.0


def eigendexplore_settings(env, path: str) -> dict:
    z = np.load(path)
    names = [str(n) for n in z["joint_names"]]
    order = list(env.cfg.robot.joint_order)
    n_arm = len(order) - len(names)
    hand = order[n_arm:]
    assert sorted(names) == sorted(hand), (names, hand)
    cols = [names.index(j) for j in hand]
    components = torch.as_tensor(z["components"][:, cols], dtype=torch.float64)
    component_std = torch.as_tensor(z["component_std"], dtype=torch.float64)
    limits = env.robot.data.joint_pos_limits[0].double().cpu()
    lab = list(env.robot.data.joint_names)
    ids = [lab.index(j) for j in hand]
    half = (limits[ids, 1] - limits[ids, 0]) / 2
    noise = EigenDExplore.from_pca(components, component_std, rank=int(z["k"]), eigen_rms=EIGEN_RMS,
                                   spectrum_power=SPECTRUM_POWER, coordinate_scale=half)
    basis = torch.zeros(noise.basis.shape[0], len(order), dtype=torch.float64)
    basis[:, n_arm:] = noise.basis
    iid = torch.full((len(order),), IID_HAND, dtype=torch.float64)
    iid[:n_arm] = 1.0
    eigen_sigma = noise.log_scale.detach().exp()
    print(f"[eigendexplore] {path} k {basis.shape[0]} over {len(hand)} hand joints after {n_arm} arm joints, "
          f"half ranges {[round(float(h), 4) for h in half]}, eigen sigma {[round(float(s), 4) for s in eigen_sigma]}",
          flush=True)
    return {"basis": basis.tolist(), "eigen_sigma": eigen_sigma.tolist(), "iid_sigma": iid.tolist()}
