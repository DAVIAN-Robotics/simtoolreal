"""CPU regression checks against recorded geometry and a deque SGS reference."""
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch
from sgs import SuccessGuidedSampler
from reachability import boxes_disjoint, quaternion_matrix


def main():
    torch.set_num_threads(1)
    torch.manual_seed(347)
    n, h = 37, 100
    sampler = SuccessGuidedSampler(n, 'cpu', history=h)
    assert torch.equal(sampler.weights, sampler.weights[0].expand(n))
    histories = [deque(maxlen=h) for _ in range(n)]
    total = np.zeros(n, dtype=int)
    for size in [1, 50, 400, 3000, 12, 12000]:
        rows = torch.randint(n, (size,))
        if size == 3000:
            rows[:] = 2  # many concurrent completions of the same reset row
        wins = torch.rand(size) < .4
        for row, win in zip(rows.tolist(), wins.tolist()):
            histories[row].append(win)
            total[row] += 1
        sampler.update(rows, wins)
        np.testing.assert_array_equal(sampler.total.numpy(), total)
        np.testing.assert_array_equal(sampler.successes.numpy(), [sum(x) for x in histories])
        for row, hist in enumerate(histories):
            ids = torch.arange(max(0, total[row]-h), total[row]) % h
            assert sampler.outcomes[row, ids].tolist() == list(hist)
    clone = SuccessGuidedSampler(n, 'cpu')
    clone.load_state_dict(sampler.state_dict())
    assert torch.equal(clone.weights, sampler.weights)
    sampler.update(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool))
    p = torch.tensor([0., .25, .5, .75, 1.])
    w = sampler._weights(p)
    assert (w > 0).all() and w[2] > w[1] > w[0] and w[0] == w[-1]
    expected = sampler.weights / sampler.weights.sum()
    draws = torch.bincount(sampler.sample(200000), minlength=n).float() / 200000
    assert (draws-expected).abs().max() < .003
    geometry = json.loads((Path(__file__).parent/'reach_geometry.json').read_text())
    records = np.load(Path(__file__).parents[1]/'experiments/furniturebench_square_table/checks/reach_reference.npz')
    for dtype in [torch.float32, torch.float64]:
        hb = torch.tensor(geometry['hand_bounds_m'], dtype=dtype)
        ob = torch.tensor(geometry['object_bounds_m'], dtype=dtype)
        palm = torch.tensor(records['palm_wxyz'], dtype=dtype)
        obj = torch.tensor(records['object_wxyz'], dtype=dtype)
        actual = boxes_disjoint(palm[:, :3], quaternion_matrix(palm[:, 3:]),
                                obj[:, :3], quaternion_matrix(obj[:, 3:]),
                                hb.mean(0), (hb[1]-hb[0])/2, ob.mean(0), (ob[1]-ob[0])/2)
        np.testing.assert_array_equal(actual.numpy(), records['disjoint'])
    print(json.dumps(dict(sgs_duplicate_and_ring_history=True,
                          sgs_global_distribution=True,
                          recorded_box_frames=len(actual),
                          float32_and_float64_match=True)), flush=True)


if __name__ == '__main__':
    main()
