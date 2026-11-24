"""Bind cubic geometry kernels to one build's field, layout and dense sampler."""
from .dual_cells import build as build_cells
from functools import partial
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

from ..evaluators.magnetic_field import Evaluator, _MISSING
from .compiled_cell_locator import locate
from ..evaluators.metric_derivatives import volume_metric
from .rays import Rays


def context(config):
    from ..operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout

    files = config['files']
    ev = Evaluator(files.get('metric_cache', _MISSING), files['field_table'],
                   filter_spec=config.get('filter_spec', _MISSING),
                   field_periods=config.get('field_periods', _MISSING), table_first=True)
    ev.capacity = config['config']['geometry_capacity']
    ev._evaluate = ev.fn
    ev.beta_min = float('inf')
    ev.beta_max = -float('inf')
    ev.max_u = 0.
    n = config['N']
    layout = build_zernike_core_ring_layout(n, n_eta=n)
    tau = 2 * np.pi
    ns = dict(np=np, build_zernike_core_ring_layout=build_zernike_core_ring_layout,
              roots_legendre=roots_legendre, N=n, P=layout.P, NN=n*layout.P,
              TAU=tau, HOP=tau/n, metric=volume_metric, locate=locate)
    sampler = Rays(ev, config['trace_substeps'])
    ns['sample'] = sampler.sample
    ns['build'] = partial(build_cells, n)
    path = Path(__file__).with_name('_volume_base.py')
    exec(compile(path.read_bytes(), str(path), 'exec'), ns)
    return ns, ev
