"""Filtered Q09 campaign authority, independent of raw-field run receipts."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GRIDS = (32, 48, 64)
STEPS = {32: 100, 48: 225, 64: 400}
END = 1e-4
ARM = 'e61e1d70a7f14697a10a10fe2b1f1262a8f0a9752a79c237d78e3ebe83c23aae'
GEOMETRY_ID = 'compact_c3:eta_filtered_m3:' + ARM
TABLE = 'eta_filter_table.npz'
TESTS = ('tests/test_q09_filtered_global.py', 'tests/test_q09_evolved_mms.py',
         'tests/test_q09_evolved_inputs.py', 'tests/test_q09_refinement_pilot.py',
         'tests/test_q09_evolved_global.py')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)


def configuration():
    return dict(schema='q09-filtered-evolved-v1', grids=GRIDS, steps=STEPS, end=END,
        arm=ARM, filter_quantity='J*B^i', max_harmonic_per_period=3, nfp=4, samples_per_period=64,
        table_grid=[257,256,1.02], diffusion_span=1/32, inner_span=1/32, outer_span=1/16,
        rk4_trace_steps=64, modes=['diffusion','complete'],
        boundaries=[['DDDDDD','D'],['NNNNNN','N'],['DNDNDN','N'],['NDNDND','D']],
        cases=24, time_steps=5800, snapshots=5, checkpoint_steps=25,
        field='one smooth six-field catalogue state; phi prescribed',
        reference='same independent fourth-order coordinate flux derivative, steps 1e-4/5e-5/2.5e-5',
        support='unchanged gradient_guard policy recomputed with filtered b',
        trace_order=['outer_minus','outer_plus','inner_minus','inner_plus'],
        backend='GPU batched RK4-64 and GPU RK4 evolution; CPU parallel row/reference preparation',
        scope='verification filtered arm; no uniform order, production, wall-crossing or long-time stability claim')


def files():
    roots = list(HERE.glob('*.py')) + [HERE/'README.md', HERE/'input_manifest.json', HERE/'inputs.tar.gz']
    for folder in ('q09_evolved_mms','q09_evolved_global'):
        roots += list((REPO/'scripts'/folder).rglob('*.py'))
    roots += [REPO/'scripts/q09_evolved_mms/manifest.json']
    for rel in ('common.py','gpu.py','design.json','input_manifest.json','inputs.tar.gz'):
        roots.append(REPO/'scripts/q08_extraction_global'/rel)
    roots += [REPO/'scripts/q08_rhs_mms_global/science.py']
    roots += [REPO/p for p in TESTS]
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(set(roots))}


def freeze():
    from scripts.q09_evolved_global.campaign import runtime_matches_pilot
    runtime_matches_pilot()
    write(HERE/'manifest.json', dict(configuration=configuration(), files=files()))
    return sha(HERE/'manifest.json')


def identity():
    actual = dict(configuration=configuration(), files=files())
    if read(HERE/'manifest.json') != json.loads(json.dumps(actual)):
        raise ValueError('frozen filtered source/configuration changed')
    return sha(HERE/'manifest.json')


def verified(run, ident):
    rec = read(Path(run)/'verification.json')
    if not rec.get('passed') or rec.get('identity') != ident: raise ValueError('verification identity')
    im = read(HERE/'input_manifest.json')
    for name, expected in im['files'].items():
        if sha(Path(run)/'inputs'/name) != expected: raise ValueError('changed input '+name)
    return rec


def verify(run, canonical_root, ident):
    run = Path(run); canonical_root = Path(canonical_root).resolve()
    im = read(HERE/'input_manifest.json')
    if sha(HERE/'inputs.tar.gz') != im['archive_sha256']: raise ValueError('input archive')
    inp = run/'inputs'; inp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(HERE/'inputs.tar.gz') as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name not in im['files'] or '..' in Path(member.name).parts:
                raise ValueError('unlisted/unsafe archive member')
            path = inp/member.name
            if path.exists():
                if sha(path) != im['files'][member.name]: raise ValueError('changed extracted input')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, path.open('wb') as dst: shutil.copyfileobj(src,dst)
    for name, expected in im['files'].items():
        if sha(inp/name) != expected: raise ValueError('small input mismatch '+name)
    for name, expected in im['canonical'].items():
        if sha(canonical_root/name) != expected: raise ValueError('canonical mismatch '+name)
    config = dict(schema='q09-filtered-run-v1', canonical_root=str(canonical_root), identity=ident)
    if (run/'run_config.json').exists() and read(run/'run_config.json') != config: raise ValueError('existing RUN binding differs')
    write(run/'run_config.json', config)
    rec = dict(passed=True, identity=ident, canonical_root=str(canonical_root), canonical=im['canonical'], inputs=im['files'])
    write(run/'verification.json',rec)
    return rec


def require(run, stage, ident):
    rec = read(Path(run)/(stage+'.json'))
    if rec.get('passed') is not True or rec.get('identity') != ident: raise ValueError('missing/invalid '+stage)
    for name,h in rec.get('files',{}).items():
        if sha(Path(run)/name) != h: raise ValueError('changed receipt payload '+name)
    return rec
