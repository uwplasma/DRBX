"""Local maintainer packaging, never a remote regeneration command."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, tarfile, subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def ah(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve()
    prior=REPO/'scripts/q07_c3_global';baseline=w/'work/q08_implementation_20261002/baseline'
    inp=HERE/'inputs';inp.mkdir(exist_ok=True)
    shutil.copytree(prior/'source/frozen_reference/frozen',inp/'geometry_model',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(inp));from geometry_model import model
    plan=json.loads((prior/'inputs/traces/plan.json').read_text());origins={};small={}
    identity='ab607bf62cda21dd67935334da9fdea35c2ad0a0c529ff555006aa3a8ed83bc3'
    traces=w/'work/q07_c3_return_20261001/q07-c3-global-ab607bf6-20261001T202426Z/results/traces'
    # Use the exact global work plan. Extra bounded/pilot trace files in the
    # returned directory must never silently overwrite global endpoints.
    for n in (32,48,64):
        _,t=model.context(n,w,physical=False,magnetic=False)
        ends=np.empty((n**3,4,3));seen=np.zeros(n**3,bool);records=[];units=[]
        for i,owners in enumerate(plan[str(n)]['global_']):
            raw=np.concatenate([t.order[t.starts[o]:t.starts[o+1]] for o in owners])
            # Frozen q_parallel.sha256_array uses dtype.str and JSON shape.
            from importlib.util import spec_from_file_location,module_from_spec
            # Obtain hash from frozen trace receipts by exact raw coverage.
            matches=[]
            for path in (traces/f'N{n}').glob(f'block_{i:06d}_*.npz'):
                with np.load(path) as z:
                    if np.array_equal(z['raw'],raw):matches.append(path)
            if len(matches)!=1:raise ValueError(('trace coverage',n,i,len(matches)))
            path=matches[0];r=json.loads(path.with_suffix('.json').read_text())
            if not(r['passed'] and r['identity']==identity and r['steps']==64 and r['crossings']==r['reentries']==0 and sha(path)==r['sha256']):raise ValueError('trace receipt')
            with np.load(path) as z:
                if seen[raw].any() or not np.array_equal(z['points'],t.pts[raw]) or not np.isfinite(z['ends']).all():raise ValueError('trace payload')
                ends[raw]=z['ends'];seen[raw]=True
            records.append(dict(path=path.name,receipt=r))
            group=[];count=0
            for owner in owners:
                size=int(t.starts[owner+1]-t.starts[owner])
                if group and count+size>128:units.append(group);group=[];count=0
                group.append(owner);count+=size
            if group:units.append(group)
        if not seen.all():raise ValueError('missing global raw traces')
        np.savez_compressed(inp/f'N{n}_traces.npz',ends=ends)
        shutil.copy2(prior/f'inputs/choices_N{n}.npz',inp/f'choices_N{n}.npz')
        small[str(n)]=dict(chunks=units,n_owner=len(t.vol),n_raw=n**3,last_aggregate=plan[str(n)]['last_aggregate'])
        origins[str(n)]=records
        fixture=inp/'bounded';fixture.mkdir(exist_ok=True)
        for name in (f'N{n}_h16.npz',f'N{n}_h32.npz',f'N{n}_inputs.npz'):
            shutil.copy2(w/'work/q08_implementation_20261002/c3_fixtures'/name,fixture/name)
        shutil.copy2(w/f'work/q07_six_field_assembly_20261002/actions_N{n}.npz',fixture/f'actions_N{n}.npz')
    shutil.copy2(w/'work/q08_implementation_20261002/c3_preparation_review.json',inp/'bounded/selection.json')
    (inp/'plan.json').write_text(json.dumps(small,indent=2)+'\n')
    (inp/'trace_origins.json').write_text(json.dumps(dict(identity=identity,records=origins),indent=2)+'\n')
    # Independent literal baseline includes numerical helpers, not aliases to
    # the extracted implementation. Record each byte's pre-extraction origin.
    legacy=HERE/'legacy';sources={}
    for group in ('native','stencils'):
        folder=legacy/group;folder.mkdir(parents=True,exist_ok=True);(folder/'__init__.py').write_text('')
        for p in (baseline/f'src/drbx/{group}').glob('q*.py'):
            shutil.copy2(p,folder/p.name);sources[str(p.relative_to(baseline))]=sha(p)
    (legacy/'__init__.py').write_text('')
    for name in ('fci_parallel_production_flux.py','characteristic_wall_residual.py'):
        p=baseline/'src/drbx/native'/name
        target=legacy/'native'/name
        if p.exists():shutil.copy2(p,target)
        else:
            revision=json.loads((baseline.parent/'baseline.json').read_text())['head']
            target.write_bytes(subprocess.check_output(['git','show',revision+':src/drbx/native/'+name],cwd=REPO))
        sources[str(p.relative_to(baseline))]=sha(target)
    (HERE/'legacy_provenance.json').write_text(json.dumps(dict(baseline=json.loads((baseline.parent/'baseline.json').read_text())['head'],files=sources),indent=2)+'\n')
    files={str(p.relative_to(inp)):sha(p) for p in sorted(inp.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    manifest=dict(schema='q08-inputs-v1',canonical=json.loads((prior/'inputs_manifest.json').read_text())['canonical'],files=files)
    (HERE/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(HERE/'inputs.tar.gz','w:gz') as tar:
        for rel in files:tar.add(inp/rel,arcname='inputs/'+rel,recursive=False)
    (HERE/'inputs.sha256').write_text(sha(HERE/'inputs.tar.gz')+'  inputs.tar.gz\n')
    print(json.dumps(dict(archive_bytes=(HERE/'inputs.tar.gz').stat().st_size,counts={n:len(p['chunks']) for n,p in small.items()})))
if __name__=='__main__':main()
