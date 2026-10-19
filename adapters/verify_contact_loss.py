"""Allocated-simulator checks for contact reporting and termination wiring."""
import torch


def verify_contact_loss(env, observed_body_contacts):
    contact = env.contact_loss
    assert contact.seconds == 2.0 and contact.limit == 120
    assert contact.metadata['contact_force_threshold_n'] == 0.0
    assert contact.metadata['links'][0] == 'fr3_link7'
    assert all('/Robot/' in p for p in contact.metadata['filter_patterns'])
    assert '/Object/' in contact.metadata['object_pattern']
    assert observed_body_contacts.any(), 'No real hand/object contact detected during smoke rollout'
    # Exercise the actual done path at both sides of the 120-step boundary.
    contact.counter.fill_(118)
    contact.touching.zero_()
    env._get_dones()
    assert not env._termination_reasons['hand_object_contact_lost'].any()
    assert (contact.counter == 119).all()
    terminated, _ = env._get_dones()
    assert env._termination_reasons['hand_object_contact_lost'].all() and terminated.all()
    contact.touching.fill_(True)
    env._get_dones()
    assert not env._termination_reasons['hand_object_contact_lost'].any()
    assert not contact.counter.any()
    # A genuinely separated object must produce zero filtered force, regardless
    # of any robot/table or robot/fixture contacts elsewhere in the same scene.
    ids = torch.tensor([0], device=env.device)
    pose = env.object.data.root_state_w[ids, :7].clone()
    pose[:, 2] += 10.0
    env.object.write_root_pose_to_sim(pose, env_ids=ids)
    env.object.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=ids)
    for _ in range(3):
        contact.begin()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    assert not contact.touching[0] and not contact.last_forces[0].any()
    contact.counter.fill_(119)
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    assert not contact.counter.any() and not contact.touching.any()
    return dict(contact_loss_seconds=contact.seconds,
                contact_loss_policy_steps=contact.limit,
                contact_sensor_links=contact.metadata['links'],
                observed_contact_links=[n for n, yes in zip(contact.metadata['links'], observed_body_contacts.tolist()) if yes],
                actual_hand_object_contact_detected=True,
                separated_object_zero_filtered_force=True,
                done_path_119_vs_120_steps=True,
                contact_and_episode_reset_clear_timer=True)
