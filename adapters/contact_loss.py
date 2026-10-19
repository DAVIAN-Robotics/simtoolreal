"""Whole-hand/object contact-loss termination, with a consecutive-step grace period."""
import json
import math
import re

import torch


def duration_steps(seconds, step_dt):
    ratio = seconds / step_dt
    nearest = round(ratio)
    # Serialized dt is slightly rounded (e.g. 1/120 -> 0.00833333333).
    return nearest if math.isclose(ratio, nearest, rel_tol=1e-8) else math.ceil(ratio)


def advance_contact_loss(counter, touching, limit):
    """Count consecutive contact-free policy steps independently for each environment."""
    counter.copy_(torch.where(touching, 0, (counter + 1).clamp(max=limit)))
    return counter >= limit


class HandObjectContactLoss:
    def __init__(self, env, seconds=2.0):
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdPhysics

        self.env = env
        self.seconds = float(seconds)
        self.limit = duration_steps(self.seconds, env.step_dt)
        self.counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.touching = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        bodies = []
        for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')):
            name = prim.GetName()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (
                name in ('palm', 'back_link', 'ee_link', 'fr3_link7')
                or name.startswith(('thumb', 'index', 'mid', 'ring', 'pinky'))
            ):
                bodies.append((name, str(prim.GetPath())))
        bodies.sort()
        assert len(bodies) == 13 and any(n == 'fr3_link7' for n, _ in bodies), bodies
        filters = [p.replace('/env_0/', '/env_*/') for _, p in bodies]
        pattern = re.sub(r'/env_\d+/', '/env_*/', env.object.root_physx_view.prim_paths[0])
        physics = SimulationManager.get_physics_sim_view()
        rigid_view = physics.create_rigid_body_view(pattern)
        ids = [int(re.search(r'/env_(\d+)/', p).group(1)) for p in rigid_view.prim_paths]
        assert sorted(ids) == list(range(env.num_envs)), (pattern, len(ids))
        self.order = torch.tensor(sorted(range(len(ids)), key=ids.__getitem__), device=env.device)
        self.view = physics.create_rigid_contact_view(
            pattern, filter_patterns=filters, max_contact_data_count=0)
        assert self.view.sensor_count == env.num_envs and self.view.filter_count == len(filters)
        self.last_forces = torch.zeros((env.num_envs, len(filters)), device=env.device)
        original_step = env.sim.step

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.last_forces = self.view.get_contact_force_matrix(dt=env.physics_dt)[self.order].norm(dim=-1)
            # Contact on either physics substep counts as contact for this policy step.
            self.touching.logical_or_((self.last_forces > 0.0).any(dim=-1))
            return result

        env.sim.step = step
        self.metadata = dict(
            seconds=self.seconds, policy_steps=self.limit,
            policy_hz=1.0/env.step_dt, physics_hz=1.0/env.physics_dt,
            contact_force_threshold_n=0.0, links=[n for n, _ in bodies],
            object_pattern=pattern, filter_patterns=filters,
            scope='object versus palm and all finger rigid bodies; excludes table/fixture',
            aggregation='any positive contact force on either physics substep',
            reset='zero on any contact and on episode reset; consecutive loss only',
            palm_note='fr3_link7 contains fixed-merged palm/back and terminal wrist collider')
        print('HAND_OBJECT_CONTACT_LOSS_READY ' + json.dumps(self.metadata), flush=True)

    def begin(self):
        self.touching.zero_()

    def reset(self, ids):
        self.counter[ids] = 0
        self.touching[ids] = False

    def update(self):
        return advance_contact_loss(self.counter, self.touching, self.limit)
