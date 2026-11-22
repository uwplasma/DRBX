"""Small checks of Q construction with synthetic geometry."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from drbx.fci_nodal.evaluators.metric_evaluator import MetricEvaluator
from drbx.fci_nodal.evaluators.eta_filtered_field import EtaFilterSpec, EtaFilterTable
from drbx.fci_nodal.evaluators.exact_plane_metric import ExactPlaneMetricEvaluator
from drbx.fci_nodal.evaluators.jax_metric_evaluator import JaxMetricEvaluator
from drbx.fci_nodal.evaluators.spline_scalar_potential import SplineScalarPotentialEvaluator
from drbx.fci_nodal.operator_build.parallel import ParallelBuildOptions
from drbx.fci_nodal.operator_build.parallel.face_operators import csr_rows
from drbx.fci_nodal.operator_build.parallel.build_by_owner import Evaluator
from drbx.fci_nodal.operators.parallel.geometry_io import Layout
from drbx.fci_nodal.operator_build.parallel.indexed_partials import Indexed, Tree
from drbx.fci_nodal.operator_build.parallel.pair_cubature import moments, reduce_pair
from drbx.fci_nodal.operator_build.parallel.weights import WallExpandedBlockFit
from drbx.fci_nodal.operators.parallel.weights import FitTable


def options_for(data, out):
    return ParallelBuildOptions(
        geometry=data/'geometry.npz',
        metric_cache=data/'metric_cache.npz', field_table=data/'eta_table.npz',
        out=out,
    )


def write_torus_cache(options):
    u = np.linspace(0., 1., 4)
    theta = np.arange(8) * (2*np.pi/8)
    eta = np.arange(4) * (2*np.pi/(4*options.field_periods))
    U, T, E = np.meshgrid(u, theta, eta, indexing='ij')
    R = 2. + .25*U*np.cos(T)
    positions = np.stack((R*np.cos(E), R*np.sin(E), -.25*U*np.sin(T)), axis=-1)
    metric = MetricEvaluator(u, theta, eta, positions, nfp=options.field_periods,
                             topology='toroidal', radial_degree=2, poloidal_modes=1, toroidal_modes=0)
    base = metric.to_cache_payload(prefix='metric_evaluator_')
    for evaluator in (MetricEvaluator, JaxMetricEvaluator):
        with pytest.raises(ValueError, match='must describe exact level-set planes'):
            evaluator.from_cache_payload(base, prefix='metric_evaluator_')
    # Zero periodic coefficients give eta = phi exactly for this synthetic map.
    spline = SplineScalarPotentialEvaluator(
        np.zeros((9, 90, 9)), G=1., R_lo=.5, h_R=.5, Z_lo=-1., h_Z=.5, nfp=options.field_periods,
    )
    exact = ExactPlaneMetricEvaluator(metric, spline)
    np.savez(options.metric_cache, **exact.to_cache_payload(prefix='metric_evaluator_'))


def test_options_and_filter_table(tmp_path):
    opts = options_for(tmp_path, tmp_path/'runtime')
    assert (opts.resolution, opts.eta_planes, opts.symmetry, opts.field_periods) == (16, 16, 'nfp', 4)
    assert opts.symmetry_period_planes == 4
    assert replace(opts, symmetry='none').symmetry_period_planes == opts.eta_planes
    for change, message in [
        ({'symmetry': 'invalid'}, "symmetry must be"),
        ({'field_periods': 0}, "eta_planes must be divisible"),
        ({'field_periods': 3}, "eta_planes must be divisible"),
        ({'eta_planes': 15}, "eta_planes must be divisible"),
    ]:
        with pytest.raises(ValueError, match=message):
            _ = replace(opts, **change).symmetry_period_planes

    write_torus_cache(opts)
    spec = EtaFilterSpec()
    u = np.linspace(0., 1., 5)
    flux = np.zeros((len(u), 8, spec.n_coefficients, 3))
    flux[..., 0, 2] = np.maximum(u, 1e-6)[:, None]
    EtaFilterTable(flux, spec, u_max=1.).save(opts.field_table)
    Evaluator(replace(opts, filter_spec=spec))
    with pytest.raises(ValueError, match='Saved field table spec differs from options'):
        Evaluator(replace(opts, filter_spec=replace(spec, max_harmonic_per_period=1)))
    with pytest.raises(ValueError, match='Saved field table nfp differs'):
        Evaluator(replace(opts, field_periods=2))


def replicated_table(opts):
    radial = np.linspace(.3, .9, 5)
    theta = np.arange(8) * (2*np.pi/8)
    rings = np.repeat(np.arange(len(radial)), len(theta))
    node_u = radial[rings]
    node_theta = np.tile(theta, len(radial))
    nodes_per_plane = len(node_u)
    offsets = np.arange(opts.eta_planes+1) * nodes_per_plane
    period = opts.symmetry_period_planes
    layout = Layout(opts.resolution, opts.eta_planes, offsets[-1], offsets,
                    period, node_u, node_theta, rings)
    allowed = np.ones((period, nodes_per_plane), dtype=bool)
    trace = [(plane, int(i)) for plane in range(period) for i in np.flatnonzero(rings == rings.max())]
    reconstruction = WallExpandedBlockFit(layout, allowed, opts.support_widths, opts.expand_widths,
                                    trace_owners=trace)
    owners = np.arange(layout.nodes)
    return FitTable(reconstruction, owners), reconstruction.xy


def test_owner_fit_reproduces_cubic(tmp_path):
    opts = replace(options_for(tmp_path, tmp_path/'runtime'), resolution=8, eta_planes=4, field_periods=2)
    fits, xy = replicated_table(opts)
    owners = np.array([0, 14, 32, fits.P+5, 3*fits.P+25])
    feet = xy[owners % fits.P] + np.array([[.02, -.04], [-.03, .01], [-.05, .02],
                                         [.025, .03], [-.02, -.025]])
    rows = fits.rows(owners, feet, fits.layout.nodes)
    nodes = np.tile(xy, (fits.layout.eta_planes, 1))
    for total in range(4):
        for a in range(total+1):
            b = total-a
            nodal = nodes[:, 0]**a * nodes[:, 1]**b
            expected = feet[:, 0]**a * feet[:, 1]**b
            np.testing.assert_allclose(rows@nodal, expected, rtol=0., atol=128*np.finfo(float).eps)


def test_pair_cubature_preserves_quadratic_moments():
    xy = np.array([(x, y) for x in (-.5, 0., .5) for y in (-.25, 0., .25)])
    weights = np.arange(1, len(xy)+1, dtype=float)/64
    x, y = xy.T
    expected_basis = np.vstack([np.ones(len(xy)), y, x, y*y, x*y, x*x])
    np.testing.assert_array_equal(moments(xy, 2), expected_basis)
    selected, reduced, _ = reduce_pair(xy, weights, 2)
    assert 0 < len(selected) <= len(expected_basis)
    assert len(np.unique(selected)) == len(selected)
    assert np.all((selected >= 0) & (selected < len(xy)))
    assert np.all(reduced > 0)
    np.testing.assert_allclose(expected_basis[:, selected]@reduced, expected_basis@weights,
                               rtol=0., atol=64*np.finfo(float).eps)


def test_carry_tree_matches_ordered_sum_across_windows():
    leaves = [
        Indexed([i % 4, i % 4 + 2], [i % 3, i % 3 + 3],
                sparse.csr_matrix([[.1*(i+1), -.2], [0., .07*(i+1)]]))
        if i != 3 else None
        for i in range(7)
    ]
    size = 7
    masses = [Indexed(leaf.rows, None, np.array([.15*(i+1), .06*(i+1)]))
              if leaf is not None else None for i, leaf in enumerate(leaves)]
    expected_matrix = np.zeros((size, size))
    expected_mass = np.zeros(size)
    for leaf, mass in zip(leaves, masses):
        if leaf is not None:
            expected_matrix[np.ix_(leaf.rows, leaf.cols)] += leaf.value.toarray()
            expected_mass[mass.rows] += mass.value

    def reduce_windows(window):
        rows = []
        mass = np.zeros(size)
        for lo in range(0, size, window):
            hi = min(lo+window, size)
            tree = Tree()
            mass_tree = Tree()
            for leaf, sub_mass in zip(leaves, masses):
                tree.push(leaf.subset(lo, hi) if leaf is not None else None)
                mass_tree.push(sub_mass.subset(lo, hi) if sub_mass is not None else None)
            rows.append(csr_rows(tree.total(), lo, hi, size))
            result = mass_tree.total()
            mass[result.rows] = result.value
        return sparse.vstack(rows).toarray(), mass

    full_matrix, full_mass = reduce_windows(size)
    np.testing.assert_allclose(full_matrix, expected_matrix, rtol=0., atol=16*np.finfo(float).eps)
    np.testing.assert_allclose(full_mass, expected_mass, rtol=0., atol=16*np.finfo(float).eps)
    for window in (1, 2, 3):
        matrix, mass = reduce_windows(window)
        np.testing.assert_array_equal(matrix, full_matrix)
        np.testing.assert_array_equal(mass, full_mass)


def test_none_fit_bank_equals_unfolded_nfp(tmp_path):
    opts = replace(options_for(tmp_path, tmp_path/'runtime'), resolution=8, eta_planes=6, field_periods=3)
    folded, _ = replicated_table(opts)
    full, _ = replicated_table(replace(opts, symmetry='none'))
    small, large = folded.bank(), full.bank()
    period, repetitions = int(small['period_nodes']), opts.field_periods
    index = np.arange(full.layout.nodes)
    repeats = np.arange(repetitions)
    expected_bank = dict(
        keys=np.concatenate([small['keys']+p*period for p in repeats]),
        index=np.concatenate([small['index']+p*len(small['keys']) for p in repeats]),
        indptr=np.r_[np.concatenate([small['indptr'][:-1]+p*len(small['indices']) for p in repeats]),
                     repetitions*len(small['indices'])],
        indices=np.tile(small['indices'], repetitions),
        plane_offsets=small['plane_offsets'],
        period_nodes=repetitions*period,
    )
    folded_rows = small['index'][index % period]
    for name, expected in expected_bank.items():
        np.testing.assert_array_equal(large[name], expected, err_msg=name)
        assert np.asarray(large[name]).tobytes() == np.asarray(expected).tobytes(), name
    for name in ('ids', 'ncol', 'C', 'Vt', 'center'):
        expected = getattr(folded, name)[folded_rows]
        actual = getattr(full, name)[large['index'][index]]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
        assert actual.tobytes() == expected.tobytes(), name
