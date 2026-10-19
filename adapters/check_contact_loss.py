"""Boundary, recovery and independent-episode checks for the contact timer."""
import json
import torch
from contact_loss import advance_contact_loss, duration_steps


def main():
    assert duration_steps(2.0, 1.0/60.0) == 120
    assert duration_steps(2.0, 2*0.00833333333) == 120
    assert duration_steps(2.01, 1.0/60.0) == 121
    counter = torch.zeros(4, dtype=torch.long)
    for step in range(1, 121):
        touching = torch.tensor([False, True, step == 60, step % 2 == 0])
        ended = advance_contact_loss(counter, touching, 120)
        assert ended.tolist() == [step == 120, False, False, False]
    assert counter.tolist() == [120, 0, 60, 0]
    ended = advance_contact_loss(counter, torch.tensor([True, False, False, False]), 120)
    assert counter.tolist() == [0, 1, 61, 1] and not ended.any()
    # Reset one episode independently; other counters retain their histories.
    counter[2] = 0
    advance_contact_loss(counter, torch.zeros(4, dtype=torch.bool), 120)
    assert counter.tolist() == [1, 2, 1, 2]
    # Last-step recovery prevents termination; repeated short gaps never accumulate.
    counter.fill_(119)
    assert not advance_contact_loss(counter, torch.ones(4, dtype=torch.bool), 120).any()
    assert not counter.any()
    print(json.dumps(dict(threshold_policy_steps=120, no_early_termination=True,
                          renewed_contact_resets_timer=True, independent_reset=True,
                          nonconsecutive_gaps_do_not_accumulate=True)), flush=True)


if __name__ == '__main__':
    main()
