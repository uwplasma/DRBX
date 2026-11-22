"""Resolution 16 ratios recorded under jax 0.12.1 for comparing two owners with one."""

import numpy as np

FIELDS = ('n','Te','Ti','Vi','w','Omega')
RECORDED_MAX = {
    'rhs.n': 7.895666478998123e-14,
    'rhs.Te': 1.2631710388997943e-14,
    'rhs.Ti': 7.885423216009865e-14,
    'rhs.Vi': 1.695406431686299e-16,
    'rhs.w': 5.856587783578515e-13,
    'rhs.Omega': 6.757357407820975e-13,
    'rhs_manufactured.n': 1.0954297365121854e-13,
    'rhs_manufactured.Te': 7.840032761145022e-13,
    'rhs_manufactured.Ti': 3.6096350040699933e-13,
    'rhs_manufactured.Vi': 3.797616473377707e-15,
    'rhs_manufactured.w': 2.0830732327714058e-13,
    'rhs_manufactured.Omega': 2.0597694425169015e-14,
    'psi': 3.966643885196037e-16,
    'A': 2.28107419398464e-14,
    'j': 5.6342590368733365e-12,
    'step_1.n': 3.560981233870587e-16,
    'step_1.Te': 5.245056728960362e-15,
    'step_1.Ti': 5.340664342846368e-16,
    'step_1.Vi': 1.8726366767433066e-16,
    'step_1.w': 2.8659765970463347e-15,
    'step_1.Omega': 2.2008994511419117e-16,
    'step_2.n': 5.332763153569142e-16,
    'step_2.Te': 9.069931931788523e-15,
    'step_2.Ti': 1.2440782128585864e-15,
    'step_2.Vi': 1.097882951789102e-16,
    'step_2.w': 4.5422751733140514e-15,
    'step_2.Omega': 4.366086099561616e-16,
    'step_3.n': 5.324112771377025e-16,
    'step_3.Te': 1.1882204560602847e-14,
    'step_3.Ti': 1.774307097734811e-15,
    'step_3.Vi': 1.8294350720219715e-16,
    'step_3.w': 7.116304135697611e-15,
    'step_3.Omega': 4.3310670368839026e-16,
}


def assert_recorded_bounds(actual, reference):
    assert actual.keys()==reference.keys()
    for name, value in actual.items():
        a,b = np.asarray(value),np.asarray(reference[name])
        assert a.shape==b.shape and a.dtype==b.dtype, name
        assert np.isfinite(a).all() and np.isfinite(b).all(), name
        if name in ('dt','positivity'):
            assert a.tobytes()==b.tobytes(), name
            continue
        fields = zip(FIELDS,a.T,b.T,strict=True) if a.ndim==2 else [(None,a,b)]
        for field,x,y in fields:
            key = name if field is None else f'{name}.{field}'
            recorded = RECORDED_MAX[key]
            # The recorded ratios are floating-point round-off; ten times each leaves a margin.
            ratio = float(np.max(np.abs(x-y)) / np.max(np.abs(y)))
            assert ratio <= 10 * recorded, (key,ratio,recorded)
