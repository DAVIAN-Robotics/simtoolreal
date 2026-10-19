"""Assemble all four screw reset pools using the existing leg4 selection rules."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source, leg, template):
    chunks = {k: [] for k in template}
    sources = []
    kinds = ['thread_engaged'] + [f'turn_{i}' for i in range(1, 8)]
    for path in sorted((source / 'event-pools/table_assembly').glob(f'*__{leg}__layout0.npz')):
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        assert str(d['active_object']) == leg and int(d['layout_id']) == 0
        request = str(d['request_id'])
        with np.load(source / 'packed' / f'{request}.npz') as z:
            squeeze = z['squeeze_qpos'][:, 7:]
        sources.append(dict(path=str(path), sha256=sha(path)))
        for progress, kind in enumerate(kinds):
            mask = (d['kind'] == kind) & d['native_pass'] & (d['mode'] == 'grasp')
            if progress == 0:
                lead = d['grasp'][(d['kind'] == 'lead_in') & d['native_pass']]
                mask &= np.isin(d['grasp'], lead)
            ix = np.flatnonzero(mask)
            counts = d['arm_count'][ix].astype(np.int64)
            repeated = np.repeat(ix, counts)
            if not len(repeated):
                continue
            branches = np.arange(len(repeated)) - np.repeat(np.cumsum(counts)-counts, counts)
            values = dict(
                arm_q=d['arm'][repeated, branches], hand_q=d['q'][repeated],
                hand_target=squeeze[d['grasp'][repeated]], object_xyzw=d['object_pose'][repeated],
                palm_xyzw=d['palm'][repeated], progress=np.full(len(repeated), progress),
                grasp=d['grasp'][repeated], request=np.full(len(repeated), request), arm_branch=branches)
            assert set(values) == set(template)
            for key, value in values.items():
                chunks[key].append(np.asarray(value, dtype=template[key].dtype))
    assert all(chunks.values()), leg
    return {k: np.concatenate(v) for k, v in chunks.items()}, sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--leg4', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    start = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.leg4) as z:
        original = {k: z[k] for k in z.files}
    assert len(original['progress']) == 198234
    events = json.loads((a.source/'inputs/event-specifications.json').read_text())['tasks']['table_assembly']
    scene = json.loads((a.source/'inputs/scenes.json').read_text())['table_assembly']
    names = ['leg1', 'leg2', 'leg3', 'leg4']
    instances = {x['object']: x for x in events['instances']}
    report = dict(source=str(a.source), leg4_source=str(a.leg4), leg4_sha256=sha(a.leg4),
                  layout=0, selection='native_pass, grasp mode, all accepted arm branches; engagement also requires lead-in', legs={})
    parts = []
    goal_parts = []
    for leg_id, leg in enumerate(names):
        # Rebuilding leg4 independently verifies the extraction rules and ordering.
        rebuilt, sources = extract(a.source, leg, original)
        if leg == 'leg4':
            for key in original:
                np.testing.assert_array_equal(rebuilt[key], original[key], err_msg=key)
            pool = original
        else:
            pool = rebuilt
            np.savez_compressed(a.out/f'{leg}_screw_resets.npz', **pool)
        goals = [e for e in instances[leg]['events'] if e['kind'] == 'thread_engaged'
                 or e['kind'].startswith('turn_') or e['kind'] == 'fully_screwed']
        assert [e['kind'] for e in goals] == ['thread_engaged']+[f'turn_{i}' for i in range(1,8)]+['fully_screwed']
        poses = np.asarray([e['representative_object_xyzw'] for e in goals])
        assert set(pool['progress']) == set(range(8)), leg
        for key in ['arm_q','hand_q','hand_target','object_xyzw','palm_xyzw']:
            assert np.isfinite(pool[key]).all(), (leg,key)
        expected = poses[pool['progress']]
        np.testing.assert_allclose(pool['object_xyzw'][:,:3], expected[:,:3], atol=1e-7)
        # q and -q represent the same orientation; retain the source quaternion.
        np.testing.assert_allclose(np.linalg.norm(pool['object_xyzw'][:,3:],axis=1),1.,atol=1e-7)
        np.testing.assert_allclose(np.abs(np.sum(pool['object_xyzw'][:,3:]*expected[:,3:],axis=1)),1.,atol=1e-7)
        goal_parts.append(dict(object=leg, events=goals))
        parts.append(dict(**pool, leg_id=np.full(len(pool['progress']),leg_id,dtype=np.int64)))
        report['legs'][leg] = dict(rows=len(pool['progress']), per_progress=np.bincount(pool['progress'],minlength=8).tolist(),
                                  sources=sources, unchanged_existing_pool=(leg=='leg4'))
        print(leg,report['legs'][leg]['rows'],report['legs'][leg]['per_progress'],flush=True)
    combined = {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    assert np.array_equal(combined['leg_id'],np.repeat(np.arange(4),[len(p['progress']) for p in parts]))
    np.savez_compressed(a.out/'screw_resets.npz',**combined)
    (a.out/'screw_goals.json').write_text(json.dumps(dict(objects=names,instances=goal_parts,scene=scene),indent=2))
    report.update(total_rows=len(combined['progress']), leg4_arrays_identical=True,
                  combined_sha256=sha(a.out/'screw_resets.npz'), wall_seconds=time.time()-start)
    (a.out/'reset_preparation.json').write_text(json.dumps(report,indent=2))
    print('RESET_POOLS_READY',report['total_rows'],flush=True)


if __name__ == '__main__':
    main()
