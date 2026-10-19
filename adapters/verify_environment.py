"""Short allocated-worker smoke check, separate from the fresh training process."""
import json
import torch
from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv


def verify(env, work):
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.common import vecenv
    from sampler_checkpoint import register_sampler_checkpoint
    assert type(env)._get_observations is PreciseAssemblyEnv._get_observations
    assert type(env)._get_rewards is PreciseAssemblyEnv._get_rewards
    assert env._hand_contact_penalty is None
    assert env.cfg.precise_assembly.hand_force_termination_n == 0
    assert env.single_action_space.shape == (12,)
    preparation=json.loads((work/'reset_preparation.json').read_text())
    assert len(env.pool['progress']) == preparation['total_rows']
    assert env.leg_names == ['leg1','leg2','leg3','leg4']
    assert env.goals.shape == (4,9,7)
    assert not env.cfg.precise_assembly.enable_retract
    assert not env.reach_disjoint().any(), 'A reset starts outside the approved reach box'
    # Cover every leg and every starting progress, without changing the training sampler.
    rows=torch.stack([torch.where((env.pool['leg_id']==leg)&(env.pool['progress']==progress))[0][0]
                      for leg in range(4) for progress in range(8)])
    ids=torch.arange(32,device=env.device)
    sample=env.sample_reset_rows
    try:
        env.sample_reset_rows=lambda count:(env.pool['progress'][rows],rows)
        env._reset_idx(ids)
    finally:
        env.sample_reset_rows=sample
    assert torch.equal(env.active_leg[ids],env.pool['leg_id'][rows])
    assert torch.equal(env.env_max_goals[ids],8-env.pool['progress'][rows])
    assert not env.reach_disjoint()[ids].any()
    for advanced in range(8):
        env._successes[ids]=torch.minimum(torch.full_like(ids,advanced),env.env_max_goals[ids]-1)
        env._write_goal_pose(ids)
        expected=env.goals[env.active_leg[ids],env.start_progress[ids]+1+env._successes[ids]].clone()
        expected[:,:3]+=env.scene.env_origins[ids]
        torch.testing.assert_close(env.goal_viz.data.root_pos_w[ids],expected[:,:3])
        torch.testing.assert_close(env.goal_viz.data.root_quat_w[ids],expected[:,3:])
    # Reaching the last pose finishes immediately without opening/retracting.
    final=env.goals[env.active_leg[ids],8].clone()
    final[:,:3]+=env.scene.env_origins[ids]
    env.object.write_root_pose_to_sim(final,env_ids=ids)
    env.object.write_root_velocity_to_sim(torch.zeros(32,6,device=env.device),env_ids=ids)
    env._near_goal_steps[ids]=env.cfg.termination.success_steps-1
    terminated,_=env._get_dones()
    assert terminated[ids].all() and env.completed[ids].all()
    assert env._termination_reasons['max_successes'][ids].all()
    assert not env.retract_phase.any()
    env._reset_idx(torch.arange(env.num_envs,device=env.device))
    register_rlgames_env(env,rl_device='cuda:0',clip_obs=10.,clip_actions=1.)
    register_sampler_checkpoint(env)
    adapter=vecenv.create_vec_env('rlgpu',env.num_envs)
    state=adapter.get_env_state()
    assert 'sgs' in state
    adapter.set_env_state(state)
    observed_body_contacts=torch.zeros(len(env.contact_loss.metadata['links']),device=env.device,dtype=torch.bool)
    for step in range(120):
        actions = (2*(env.last_targets-env.hlo)/(env.hhi-env.hlo)-1)
        actions = (actions + .15*torch.randn_like(actions)).clamp(-1, 1)
        obs, rew, term, trunc, info = env.step(actions)
        assert torch.isfinite(rew).all()
        assert all(torch.isfinite(v).all() for v in obs.values())
        assert torch.equal(env._cur_targets[:,env.aids], env.arm_hold)
        assert torch.equal(env._prev_targets[:,env.aids], env.arm_hold)
        assert not env._termination_reasons['hand_force'].any()
        observed_body_contacts |= (env.contact_loss.last_forces > 0).any(dim=0)
    from verify_contact_loss import verify_contact_loss
    contact_report=verify_contact_loss(env,observed_body_contacts)
    # Controlled out-of-reach probe: translated object, identity velocity.
    ids = torch.tensor([0], device=env.device)
    pose = env.object.data.root_state_w[ids,:7].clone()
    pose[:,0] += 2.
    env.object.write_root_pose_to_sim(pose,env_ids=ids)
    assert env.reach_disjoint()[0], 'Translated object was not separated'
    terminated, _ = env._get_dones()
    assert terminated[0] and env._termination_reasons['hand_unreachable'][0]
    report = dict(steps=120, num_envs=env.num_envs,
                  actor_dim=obs['policy'].shape[-1], critic_dim=obs['critic'].shape[-1],
                  action_dim=12, inherited_observations=True, arm_targets_held=True,
                  all_four_legs_and_eight_starting_stages=True,
                  goals_follow_sampled_leg=True, final_goal_ends_without_retract=True,
                  global_sgs_pool_rows=len(env.pool['progress']),
                  force_penalty_and_termination_disabled=True,
                  immediate_reach_termination=True,
                  observed_sgs_episodes=int(env.sampler.total.sum()))
    report.update(contact_report)
    (work/'rl_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
