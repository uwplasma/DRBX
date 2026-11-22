"""File observations for the pinned direct-owner test."""

import json
import os
from pathlib import Path
import sys

LOG_ENV = 'DRBX_TEST_OWNER_IO_LOG'
STAGE_ENV = 'DRBX_TEST_OWNER_IO_STAGE'
_SUFFIXES = {'.bin','.json','.nc','.npy','.npz','.pkl','.sqlite','.db'}
_GATHERED = {
    'bundle.npz','q_geometry.npz','geometry.npz','operator.npz','C.npz','M.npz',
    'K_N.npz','MB_Gram.npz','Rw.npz','owner_fits.npz','A1_degree2.npz','faces.json',
    'runtime_manifest.json',
}
_OWNER_FILES = {
    'wall_plan.npz','limiter_graph.npz','parallel_geometry.npz','campaign_T0.npz',
    'campaign_T1.npz','wall_hit_parallel_derivatives.npz','prepared.json','hit_layout.npz',
    'static_rhs.npz','U.npy','observations.npz','node_ids.npy',
    'manufactured_field_data_conducting_wall.npz','manufactured_field_data_floating_wall.npz','nodal_state.npy',
}


def install_open_audit():
    directory = os.environ.get(LOG_ENV)
    if not directory:
        return
    fd = os.open(Path(directory)/f'files-{os.getpid()}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    seen = set()

    def audit(event,args):
        if not (stage := os.environ.get(STAGE_ENV)):
            return
        if event=='open':
            path,_,flags = args
            operation = 'write' if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) else 'read'
            pairs = ((path,operation),)
        elif event=='sqlite3.connect':
            pairs = ((args[0],'sqlite'),)
        elif event in ('os.rename','os.link','os.symlink'):
            pairs = ((args[0],'source'),(args[1],'write'))
        elif event=='os.remove':
            pairs = ((args[0],'remove'),)
        else:
            return
        for path,operation in pairs:
            if not isinstance(path,(str,bytes,os.PathLike)):
                continue
            path = os.path.abspath(os.fsdecode(path))
            if Path(path).suffix not in _SUFFIXES:
                continue
            row = (stage,operation,path)
            if row not in seen:
                seen.add(row)
                os.write(fd,(json.dumps(row)+'\n').encode())

    sys.addaudithook(audit)


def forbidden(path, pinned):
    if any(path.is_relative_to(pinned/stage)
           for stage in ('p_bundle','q_operators','mms_prepare','mms_run')):
        return True
    if path.name in _GATHERED or (path.name=='owners.npz' and path.parent.name=='jump'):
        return True
    if path.name.startswith(('nodal_metric_','laplacian_metric_','J1_')) and path.suffix=='.npz':
        return True
    if path.name in _OWNER_FILES:
        return not any(part.startswith('owner') and part[5:].isdigit() for part in path.parts)
    return False


def file_observations(log, case, pinned):
    stages = {}
    for file in log.glob('files-*.jsonl'):
        for line in file.read_text().splitlines():
            stage,operation,value = json.loads(line)
            stages.setdefault(stage,{}).setdefault(operation,set()).add(value)
    activity = {s:{k:sorted(v) for k,v in ops.items()} for s,ops in stages.items()}
    inventory = {p.name:sorted(str(q.relative_to(case)) for q in p.rglob('*') if q.is_file())
                 for p in case.iterdir() if p.is_dir()}
    illegal = {path for ops in activity.values() for values in ops.values() for path in values
               if forbidden(Path(path),pinned)}
    illegal.update(str(case/value) for paths in inventory.values() for value in paths
                   if forbidden(case/value,pinned))
    return dict(activity=activity,inventory=inventory,forbidden=sorted(illegal))


def observe_owner_worker():
    """Save stage states per owner outside the case's runtime products."""
    import numpy as np
    from unittest.mock import patch

    from drbx.fci_nodal.case._worker import main
    from drbx.fci_nodal.drb import owner_operator

    directory = Path(os.environ['DRBX_TEST_OWNER_SNAPSHOTS'])
    actions = owner_operator.owner_actions
    advance = owner_operator.owner_step
    resolution = None
    step_number = 0

    def bind(owner_resolution, *, switches):
        nonlocal resolution
        resolution = owner_resolution
        return actions(resolution,switches=switches)

    def observe(*args, **kwargs):
        nonlocal step_number
        result = advance(*args,**kwargs)
        step_number += 1
        state = {shard.device:shard.data for shard in result[0].addressable_shards}
        for ordinal,device in enumerate(resolution.q.mesh.devices.flat):
            folder = directory/f'owner{ordinal:04d}'
            folder.mkdir(exist_ok=True)
            valid = resolution.q.nodes.ids_by_owner[ordinal]>=0
            np.save(folder/f'step_{step_number}.npy',np.asarray(state[device])[valid],allow_pickle=False)
        return result

    with patch.object(owner_operator,'owner_actions',bind), patch.object(owner_operator,'owner_step',observe):
        main()
