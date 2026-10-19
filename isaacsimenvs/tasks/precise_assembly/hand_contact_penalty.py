"""Pair-filtered hand contact costs, sampled on every physics substep."""
import json
import re
import torch


class HandContactPenalty:
    def __init__(self, env):
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdPhysics
        self.env = env
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
        physics = SimulationManager.get_physics_sim_view()
        self.views, self.orders = [], []
        for obj in (env.table, env.hole):
            pattern = re.sub(r'/env_\d+/', '/env_*/', obj.root_physx_view.prim_paths[0])
            rv = physics.create_rigid_body_view(pattern)
            ids = [int(re.search(r'/env_(\d+)/', p).group(1)) for p in rv.prim_paths]
            assert sorted(ids) == list(range(env.num_envs)), (pattern, len(ids))
            order = torch.tensor(sorted(range(len(ids)), key=ids.__getitem__), device=env.device)
            cv = physics.create_rigid_contact_view(pattern, filter_patterns=filters, max_contact_data_count=0)
            assert cv.sensor_count == env.num_envs and cv.filter_count == len(filters)
            self.views.append(cv)
            self.orders.append(order)
        self.peak = torch.zeros((env.num_envs, 2, len(filters)), device=env.device)
        original_step = env.sim.step
        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            for i, (view, order) in enumerate(zip(self.views, self.orders)):
                force = view.get_contact_force_matrix(dt=env.physics_dt)[order].norm(dim=-1)
                torch.maximum(self.peak[:, i], force, out=self.peak[:, i])
            return result
        env.sim.step = step
        print('HAND_CONTACT_READY ' + json.dumps({
            'links': [n for n, _ in bodies], 'filters': filters,
            'sources': ['table', 'fixture'], 'num_envs': env.num_envs,
            'weights': [env.cfg.precise_assembly.hand_fixture_force_coef],
            'units': 'N; per-body substep peak, then sum over hand bodies',
            'physics_dt': env.physics_dt, 'policy_dt': env.step_dt,
            'palm_note': 'fr3_link7 includes fixed-merged palm/back and terminal wrist collider'
        }), flush=True)

    def begin(self):
        self.peak.zero_()

    def forces(self):
        return self.peak.sum(dim=-1)

    def apply(self, reward):
        forces = self.forces()
        cfg = self.env.cfg.precise_assembly
        for i, (name, coef) in enumerate((('fixture', cfg.hand_fixture_force_coef),)):
            penalty = -float(coef) * forces[:, i + 1]
            reward = reward + penalty
            self.env._reward_terms[f'hand_{name}_collision_penalty'] = penalty
            self.env.extras[f'hand_{name}_force_N'] = forces[:, i + 1].mean()
            self.env.extras[f'hand_{name}_contact_fraction'] = (forces[:, i + 1] > 0.01).float().mean()
            self.env.extras[f'hand_{name}_force_max_N'] = forces[:, i + 1].max()
        self.env._reward_terms['total_reward'] = reward
        return reward
