"""Bounded actual-HSX validation of seed selection, complete actions and spawn pools."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import sys
import time
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.q_fci_return_campaign import campaign as c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    args = SimpleNamespace(input_root=a.workspace, output=a.output, workers=2, seeds=1)
    identity = c.verify(args)
    cid = c.digest(identity)
    report = {'source': c.source_identity(), 'resolutions': {}, 'scope': 'One complete ordinary owner at each N; axis/wall/seam probes. Not full preflight or global qualification.'}
    for n in (32, 48, 64):
        started = time.monotonic()
        c.initialize(n, str(a.workspace), str(a.output), seeds=1)
        ctx, m = c.CTX, c.NUM
        _, _, _, tracks = c.bounded_selection(ctx)
        owner = next(t['owner'] for t in tracks if t['name'] == 'ordinary')
        owners = np.array([owner])
        faces = np.flatnonzero((ctx['lower'] == owner) | (ctx['upper'] == owner))
        raw = np.flatnonzero(ctx['topology']['compact_raw_owner'].ravel() == owner)
        ids = np.unique(np.concatenate([m.candidate_ids(n, ctx['keys'][f], 2) for f in faces]))
        root = a.output / f'bounded/N{n}'
        root.mkdir(parents=True, exist_ok=True)
        cat = c.trace_stage(args, n, root, ids, cid, chunk=4)
        assert not c.stage_units(args, n, root, faces, 'face', 8, c.do_faces, cid, cat)
        c.stage_units(args, n, root, raw, 'volume', 8, c.do_volume, cid)
        summary = c.assemble(root, ctx, owners, faces, raw, cid)
        rows = c.load_catalogue(cat)
        assert rows['source'].shape == (2*n**3, 1, 3)
        assert rows['numerical'].shape == (2*n**3, 4)
        ijk = m.row_keys(n, ids)[:, :3]
        grid = ctx['artifact'].geometry.grid
        expected = np.column_stack([.5*(getattr(grid, axis).faces[ijk[:, k]] + getattr(grid, axis).faces[ijk[:, k]+1]) if k<2 else grid.z.centers[ijk[:, k]] for k, axis in enumerate('xyz')])
        np.testing.assert_allclose(rows['source'][ids, 0], expected, rtol=0, atol=2e-15)
        good = ids[rows['valid'][ids]]
        # One-seed magnetic weights cancel from the directional observation.
        exact = m.fields(ctx, np.vstack((rows['source'][good, 0], rows['endpoint'][good, 0])))[0]
        direct = np.where(good % 2, 1, -1)[:, None] * (exact[len(good):] - exact[:len(good)]) / rows['ell'][good]
        np.testing.assert_allclose(rows['g_sec'][good], direct, atol=1e-13, rtol=1e-12)
        assert summary['seed_count'] == 1 and summary['constant_max'] < 1e-8
        assert max(summary['balance'].values()) < 1e-11
        assert summary['fit']['rank_min'] == 19
        if n == 32:
            before = {str(p): (p.stat().st_mtime_ns, c.sha(p)) for p in (root/'trace_chunks').glob('*.npz')}
            c.trace_stage(args, n, root, ids, cid, chunk=4)
            assert before == {p: (Path(p).stat().st_mtime_ns, c.sha(p)) for p in before}
            report['checkpoint_resume_preserved'] = len(before)
        probes = {}
        ctx['trace_capacity'] = 8
        for direction in (-1, 1):
            keys = [[i, j, 0, direction] for i in (0, 1, n-2, n-1) for j in (0, n-1)]
            rr = m.row_ids(n, keys)
            ctx['config']['policy']['trace_substeps'] = 64
            x = m.trace_rows(ctx, rr)
            ctx['config']['policy']['trace_substeps'] = 128
            y = m.trace_rows(ctx, rr)
            np.testing.assert_array_equal(x['valid'], y['valid'])
            valid = x['valid']
            error = float(np.max(abs(m.regular(x['endpoint'][valid].reshape(-1,3),0)-m.regular(y['endpoint'][valid].reshape(-1,3),0))))
            assert error < 1e-6
            probes[str(direction)] = {'rows': len(rr), 'valid': int(valid.sum()), 'endpoint_step_doubling_max': error}
        report['resolutions'][str(n)] = {'rows': len(ids), 'summary': summary, 'probes': probes, 'seconds': time.monotonic()-started}
        c.write(a.output/'single_seed_validation.json', report)
        print(f'N{n} single-seed complete-owner/pool/checkpoint/probe checks passed', flush=True)
    # New parameterization preserves the previous four-seed implementation.
    c.initialize(32, str(a.workspace), str(a.output), seeds=4, trace_capacity=96)
    old = a.workspace/'work/q_fci_rk4_validation_20260923/smoke_64_1.npz'
    with np.load(old) as z:
        new = c.NUM.trace_rows(c.CTX, z['ids'])
        for key in ('source', 'F', 'valid', 'endpoint', 'ell', 'numerical'):
            np.testing.assert_allclose(new[key], z[key], rtol=1e-12, atol=1e-13, equal_nan=True)
    wrong = SimpleNamespace(input_root=a.workspace, output=a.output, seeds=4)
    try:
        c.verify(wrong)
    except RuntimeError as exc:
        assert 'incompatible campaign' in str(exc)
    else:
        raise AssertionError('seed variants reused one campaign folder')
    report.update(passed=True, four_seed_regression_passed=True, mixed_seed_resume_rejected=True)
    c.write(a.output/'single_seed_validation.json', report)
    print('All bounded single-seed checks passed', flush=True)


if __name__ == '__main__':
    main()
