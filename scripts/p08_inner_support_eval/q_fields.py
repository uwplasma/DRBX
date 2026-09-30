"""Q's field catalogue for the C0-vs-candidate evaluation: values, gradients and (jax) Hessians as real columns.

Source: ``scripts/q_fci_layered_global/fields.py`` (``FIELDS``, ``field``, ``evaluate``/``WAVES``).  The 26 ``FIELDS``
are scored; the 20 plane waves are complex, so every wave enters as two real columns (real and imaginary part), the six
real fields (``common``, ``homogeneous_D``, ``simple_zero_N``, ``constant``, ``fA``, ``fB``) as one.  The short waves
(lambda = 0.5 and 0.25, x and y; ``evaluate()``'s ``WAVES``) are a response report only: extra columns, never part of any
order or pooled statistic.

Hessians come from ``jax.hessian`` of a jnp port of ``field`` (``verify_port`` asserts the port's value and gradient
against ``field`` to 1e-13 and its Hessian against a central difference of ``field``'s analytic gradient).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parents[1]                  # .../DRBX/scripts
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from q_fci_layered_global import fields as qf  # noqa: E402

FIELDS = tuple(qf.FIELDS)                                         # the 26 scored fields
SHORT_FIELDS = tuple(name for name, _d, lam in qf.WAVES if lam < 1.0)   # x/y_lambda0.5_m1, x/y_lambda0.25_m1
REAL_FIELDS = frozenset(('common', 'homogeneous_D', 'simple_zero_N', 'constant', 'fA', 'fB'))
ALL_NAMES = FIELDS + SHORT_FIELDS

#: column list: (field name, part) in ``ALL_NAMES`` order; the first ``N_MAIN_COLUMNS`` belong to ``FIELDS``
COLUMNS = tuple((name, part) for name in ALL_NAMES for part in (('re',) if name in REAL_FIELDS else ('re', 'im')))
N_MAIN_COLUMNS = sum(1 if name in REAL_FIELDS else 2 for name in FIELDS)
N_COLUMNS = len(COLUMNS)
COLUMN_INDEX = {c: i for i, c in enumerate(COLUMNS)}
FIELD_COLUMNS = {name: [i for i, (n, _p) in enumerate(COLUMNS) if n == name] for name in ALL_NAMES}


def column_to_field_matrix(names):
    """``(N_COLUMNS, len(names))`` 0/1 matrix: column -> field, for combining re/im parts as sqrt(re^2 + im^2)."""
    m = np.zeros((N_COLUMNS, len(names)))
    for j, name in enumerate(names):
        m[FIELD_COLUMNS[name], j] = 1.0
    return m


MAIN_MATRIX = column_to_field_matrix(FIELDS)
SHORT_MATRIX = column_to_field_matrix(SHORT_FIELDS)


def _columns_of(name, value):
    value = np.asarray(value)
    return [value.real] if name in REAL_FIELDS else [value.real, value.imag]


def exact_columns(points, names=ALL_NAMES):
    """Exact value ``(P, C)`` and logical gradient ``(P, 3, C)`` at ``points`` (Q's ``field``), real columns."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vals, grads = [], []
    for name in names:
        v, g = qf.field(name, p)
        vals.extend(_columns_of(name, v))
        grads.extend(_columns_of(name, g))
    return np.stack(vals, axis=-1), np.stack(grads, axis=-1)


# ---------------------------------------------------------------------------
# jnp port of ``field`` (real/imag stacked) and jax Hessians
# ---------------------------------------------------------------------------
def _parse_wave(name):
    """(kind, direction/angle, wavelength) for a plane-wave name, else None."""
    if name.startswith(('wave_a', 'fresh_a')):
        angle, lam = name.removeprefix('wave_a').removeprefix('fresh_a').split('_lambda')
        return 'angle', np.deg2rad(float(angle)), float(lam)
    for wave_name, direction, lam in qf.WAVES:
        if wave_name == name:
            return 'axis', direction, float(lam)
    return None


def _port(name):
    import jax.numpy as jnp

    wave = _parse_wave(name)

    def f(p):
        u, t, e = p[0], p[1], p[2]
        x, y = u * jnp.cos(t), u * jnp.sin(t)
        if wave is not None:
            kind, arg, lam = wave
            k = 2 * np.pi / lam
            z = k * u * jnp.cos(t - arg) if kind == 'angle' else (k * x if arg == 'x' else k * y)
            phase = z + e
            return jnp.stack([jnp.cos(phase), jnp.sin(phase)])
        if name == 'constant':
            return jnp.stack([1.0 + 0.0 * u])
        if name == 'fA':
            a = 1 + .1 * x + .07 * y
            return jnp.stack([a * jnp.cos(3 * e + .37)])
        if name == 'fB':
            a = jnp.exp(.2 * x - .15 * y)
            return jnp.stack([a * jnp.sin(5 * e + .61)])
        v = 1 + .1 * x * jnp.cos(e) + .05 * y * jnp.sin(e) + .02 * (x * x - y * y) * jnp.cos(2 * e)
        if name == 'common':
            return jnp.stack([v])
        if name in ('homogeneous_D', 'simple_zero_N'):
            return jnp.stack([(1 - u * u) ** 2 * v + (1 if name == 'simple_zero_N' else 0)])
        raise ValueError(name)

    return f


_VGH = {}


def _vgh_fn(name):
    if name not in _VGH:
        import jax
        f = _port(name)
        g = jax.jacfwd(f)
        h = jax.jacfwd(g)
        _VGH[name] = jax.jit(jax.vmap(lambda p: (f(p), g(p), h(p))))
    return _VGH[name]


def jax_vgh(points, names=FIELDS):
    """Value ``(P, C)``, gradient ``(P, 3, C)`` and Hessian ``(P, 3, 3, C)`` columns from the jnp port (x64)."""
    import jax
    jax.config.update('jax_enable_x64', True)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vals, grads, hess = [], [], []
    for name in names:
        v, g, h = (np.asarray(a) for a in _vgh_fn(name)(p))       # (P, c), (P, c, 3), (P, c, 3, 3)
        vals.append(v); grads.append(np.moveaxis(g, 1, -1)); hess.append(np.moveaxis(h, 1, -1))
    return np.concatenate(vals, axis=-1), np.concatenate(grads, axis=-1), np.concatenate(hess, axis=-1)


def verify_port(seed=0, count=64, tol=1e-13):
    """Assert the port's value and gradient equal ``field`` to ``tol`` (scored fields; 10 tol for the short waves, whose
    gradients reach 2 pi / 0.25) and its Hessian
    matches a central difference of ``field``'s gradient (scored fields).  Returns the max defects."""
    rng = np.random.default_rng(seed)
    p = np.column_stack((rng.uniform(0.02, 0.98, count), rng.uniform(0, 2 * np.pi, count), rng.uniform(0, 2 * np.pi, count)))
    out = {}
    for label, names, limit in (('scored', FIELDS, tol), ('short', SHORT_FIELDS, 10 * tol)):   # short: |grad| ~ 2 pi / 0.25
        v0, g0 = exact_columns(p, names)
        v1, g1, _h = jax_vgh(p, names)
        dv, dg = float(np.max(np.abs(v1 - v0))), float(np.max(np.abs(g1 - g0)))
        if not (dv <= limit and dg <= limit):
            raise AssertionError(f'jnp port differs from q fields ({label}): value {dv:.2e}, gradient {dg:.2e} (tol {limit})')
        out[label] = dict(value=dv, gradient=dg)
    _v, _g, h1 = jax_vgh(p, FIELDS)
    step, dh = 1e-6, 0.0
    for a in range(3):
        d = np.zeros(3); d[a] = step
        _vp, gp = exact_columns(p + d, FIELDS)
        _vm, gm = exact_columns(p - d, FIELDS)
        fd = (gp - gm) / (2 * step)                               # (P, 3, C): d/dx_a of gradient component b
        an = h1[:, a, :, :N_MAIN_COLUMNS]
        dh = max(dh, float(np.max(np.abs(fd - an) / (1.0 + np.abs(an)))))
    if dh > 1e-5:
        raise AssertionError(f'jnp Hessian differs from a finite difference of the analytic gradient: {dh:.2e}')
    out['hessian_fd_rel'] = dh
    return out
