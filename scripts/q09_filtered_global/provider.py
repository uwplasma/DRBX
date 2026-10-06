"""Full-domain filtered lineage adapter for the unchanged Q09 evolution kernel."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
from scripts.q09_evolved_mms.provider import PreparedProvider
from scripts.q09_evolved_mms.mms import Manufactured,ContinuumReference
from scripts.q08_extraction_global.gpu import merge_chunks,estimate_merge
from scripts.q08_rhs_mms_global.science import masks,REGIONS
from . import common as c
from .geometry import setup
from .reference import FIELDS

class FilteredProvider(PreparedProvider):
    def _validate_lineage(self):
        h=self.input_hashes
        if (h.get('filtered_arm')!=c.ARM or h.get('prepared_bank_identity')!=self.bank.identity
            or self.bank.metadata.get('geometry_identity')!=c.GEOMETRY_ID
            or 'q08_receipt_identity' in h or not h.get('filtered_campaign')):
            raise ValueError('filtered provider lineage/geometry mismatch')


def configure_grid(run,n,ident,host_gib):
    run=Path(run); c.require(run,f'cpu_N{n}',ident)
    _,estimate=estimate_merge(run,n,ident,host_gib)
    grid=run/f'N{n}'
    cfg=dict(schema='q09-filtered-provider-v1',run=str(run.resolve()),n=n,identity=ident,host_gib=host_gib)
    target=grid/'config/inputs.json'
    if target.exists() and c.read(target)!=cfg: raise ValueError('grid binding changed')
    c.write(target,cfg)
    im=c.read(c.HERE/'input_manifest.json')
    hashes=dict(filtered_campaign=ident,filtered_arm=c.ARM,table=im['files'][c.TABLE],cpu_gate=c.sha(run/f'cpu_N{n}.json'))
    c.write(grid/'verification.json',dict(passed=True,identity=ident,inputs=hashes,config_sha256=c.sha(target),n=n,
        owners=c.read(run/'inputs/plan.json')[str(n)]['n_owner'],estimate=estimate))
    c.require(run,'tests',ident);c.require(run,'preflight_cpu',ident);c.require(run,'preflight_gpu',ident)
    c.write(grid/'tests.json',dict(passed=True,identity=ident,parent=c.sha(run/'tests.json')))
    c.write(grid/'preflight.json',dict(passed=True,identity=ident,parent=c.sha(run/'preflight_gpu.json')))


def load(directory):
    cfg=c.read(Path(directory)/'inputs.json');run=Path(cfg['run']);n=cfg['n'];ident=cfg['identity']
    c.verified(run,ident);c.require(run,f'cpu_N{n}',ident)
    with jax.default_device(jax.devices('cpu')[0]):
        bank,geometry,views,estimate=merge_chunks(run,n,ident,cfg['host_gib'])
        t=setup(run,n,physical=False);refs=[]
        for v in views:
            for name,h in v.receipt['files'].items():
                if c.sha(v.path/name)!=h: raise ValueError('filtered reference payload changed')
            with np.load(v.path/'reference.npz') as z:
                arrays={k:z[k].copy() for k in FIELDS}
            np.testing.assert_array_equal(arrays['owner_raw'],v.bank.owner_raw)
            np.testing.assert_array_equal(arrays['owner_weight'],v.bank.owner_weight)
            refs.append(ContinuumReference(**arrays,diagnostics=c.read(v.path/'reference.json')))
        ref=ContinuumReference(*(np.concatenate([np.asarray(getattr(r,k)) for r in refs],axis=0) for k in FIELDS[:6]),bank.owner_raw,bank.owner_weight,dict(chunks=[r.diagnostics for r in refs],filtered_arm=c.ARM))
        m=Manufactured.from_members(bank,bank.diagnostics['slot_points'][:,2],bank.raw_to_owner,t.rv[bank.raw],t.vol)
        m=replace(m,owner_initial=np.asarray(m.owner_initial),phi_initial=np.asarray(m.phi_initial),boundary_initial=jax.tree.map(np.asarray,m.boundary_initial),boundary_constant=jax.tree.map(np.asarray,m.boundary_constant))
        radial=(t.pts[t.order[t.starts[:-1]],0]*n).astype(int)
        last=c.read(run/'inputs/plan.json')[str(n)]['last_aggregate']
        regions=dict(zip(REGIONS[1:],masks(radial,n,last)[1:],strict=True))
        hashes=dict(c.read(Path(directory).parent/'verification.json')['inputs'],prepared_bank_identity=bank.identity)
        provider=FilteredProvider(bank,geometry,m,ref,t.vol,regions,hashes).validate()
    return provider
