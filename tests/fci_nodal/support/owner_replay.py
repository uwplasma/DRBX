"""Observe arrays emitted by an owner worker in pinned replays."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


def gather(value, layout):
    rows = np.empty((layout.n,*value.shape[1:]),value.dtype)
    for owner, piece in enumerate(sorted(value.addressable_shards,key=lambda s:s.index[0].start or 0)):
        ids = layout.ids_by_owner[owner]
        active = ids>=0
        rows[ids[active]] = np.asarray(piece.data)[active]
    return rows


def observe_case_worker():
    """Record existing owner outputs while the real worker runs the case stage."""
    from drbx.fci_nodal.case._worker import main
    from drbx.fci_nodal.drb import owner_operator

    outputs = {}
    actions, step = owner_operator.owner_actions, owner_operator.owner_step
    nodes = None
    step_number = 0

    def observe_actions(resolution, *, switches):
        nonlocal nodes
        nodes = resolution.q.nodes
        raw,force,project = actions(resolution,switches=switches)
        def observed_raw(state,wall):
            result,aux = raw(state,wall)
            for key in ('psi','A','current'):
                outputs['j' if key=='current' else key] = gather(aux[key],nodes)
            return result,aux
        return observed_raw,force,project

    def observe_step(*args, **kwargs):
        nonlocal step_number
        result = step(*args,**kwargs)
        step_number += 1
        outputs[f'step_{step_number}'] = gather(result[0],nodes)
        return result

    with patch.object(owner_operator,'owner_actions',observe_actions), \
         patch.object(owner_operator,'owner_step',observe_step):
        main()
    np.savez(Path(sys.argv[2])/'observations.npz',**outputs)


