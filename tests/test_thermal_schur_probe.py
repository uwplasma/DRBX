"""Block elimination controls for the work-only thermal Schur candidate."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'work/boundary_load_audit'))
from thermal_schur_probe import thermal_schur_action, bounded_fgmres


TARGET_FIELDS = ("Te", "Ti", "Vi", "Ve", "vorticity")
PRIMITIVE_FIELDS = ("density", "Te", "Ti", "Vi", "Ve")
THERMAL_FEEDBACK_FIELDS = ("Te", "Ti", "Vi")


def _capture_problem(fields):
    """Construct the coordinate-only part of the reduced runner cheaply."""
    from run_reduced_implicit_capture import CaptureReducedProblem, FIELDS
    from types import SimpleNamespace

    problem = object.__new__(CaptureReducedProblem)
    problem.selected_fields = tuple(fields)
    problem.selected_indices = tuple(FIELDS.index(name) for name in fields)
    problem.excluded_fields = tuple(name for name in FIELDS if name not in fields)
    problem.adapter = SimpleNamespace(
        n_active=3,
        base_material=np.arange(18, dtype=float).reshape(6, 3),
        embed_material_direction=lambda value: np.asarray(value, dtype=float),
    )
    problem.nu = len(fields) * problem.adapter.n_active
    problem.diagnostic_arrays = {}
    return problem


def _primitive_response_to_full_material(primitive, vorticity):
    """Embed primitive ``B_q`` rows and the vorticity repair in six lanes."""
    from run_reduced_implicit_capture import FIELDS

    primitive = np.asarray(primitive, dtype=float)
    full = np.zeros((len(FIELDS), primitive.shape[1]), dtype=float)
    for row, name in enumerate(PRIMITIVE_FIELDS):
        full[FIELDS.index(name)] = primitive[row]
    full[FIELDS.index("vorticity")] = np.asarray(vorticity, dtype=float)
    return full


def _thermal_c_input(problem, selected):
    """Select Te/Ti/Vi from target coordinates before applying C."""
    selected = np.asarray(selected, dtype=float).reshape(
        len(problem.selected_fields), problem.adapter.n_active
    )
    thermal = np.zeros_like(selected)
    positions = {name: problem.selected_fields.index(name)
                 for name in THERMAL_FEEDBACK_FIELDS}
    for name, position in positions.items():
        thermal[position] = selected[position]
    return thermal


def test_thermal_elimination_reproduces_coupled_solve_including_gauge():
    # The final algebraic coordinate is a multiplier. Material-dependent
    # gauge data make C's final row nonzero, which must survive elimination.
    m = np.array([[2., .3], [-.1, 3.]])
    b = np.array([[1., -.2, 0.], [.5, .1, 0.]])
    c = np.array([[.1, -.3], [.2, .1], [.4, -.2]])
    a = np.array([[2., -1., 1.], [-1., 2., 1.], [.4, .6, 0.]])
    rhs = np.arange(1., 6.)
    action = thermal_schur_action(lambda z:a@z,
        lambda z:np.linalg.solve(m,b@z),lambda q:c@q)
    schur = np.column_stack([action(e) for e in np.eye(3)])
    z = np.linalg.solve(schur, rhs[2:]-c@np.linalg.solve(m,rhs[:2]))
    u = np.linalg.solve(m,rhs[:2]-b@z)
    full = np.block([[m,b],[c,a]])
    assert np.linalg.norm(full@np.r_[u,z]-rhs)<1.e-13


def test_omitted_vorticity_is_measured_as_approximation_error():
    a=np.array([[2., 1.],[1., 0.]])
    b=np.array([[.3, 0.],[.2, 0.]])
    c=np.array([[.2, .1],[.1, 0.]])
    p=np.array([[.7, .1],[0., .8]])
    omega=np.array([[.03,0.],[0.,0.]])
    action=thermal_schur_action(lambda z:a@z,lambda z:p@b@z,lambda q:c@q)
    full_h=a-c@p@b-omega
    z=np.array([2.,1.])
    assert np.allclose(action(z)-full_h@z,omega@z,rtol=1.e-13,atol=1.e-14)


def test_bounded_replay_handles_variable_preconditioner_and_checks_true_residual():
    matrix=np.array([[3.,1.,0.],[-.2,2.,.4],[0.,.5,1.]])
    rhs=np.array([1.,2.,-1.]); calls=[]; progress=[]
    def precondition(v):
        calls.append(1)
        return v/np.diag(matrix)*(1.+.1*len(calls))
    x,info=bounded_fgmres(lambda v:matrix@v,rhs,precondition,budget=3,rtol=1.e-12,
        progress=lambda i,r:progress.append((i,r)))
    assert np.linalg.norm(matrix@x-rhs)/np.linalg.norm(rhs)<1.e-12
    assert info['true_relative']<1.e-12
    assert len(progress)==info['iterations']==len(calls)


def test_reduced_schur_preconditioner_inverts_reduced_operator_with_exact_blocks():
    from thermal_reduced_capture import reduced_schur_preconditioner
    m = np.array([[2., .3], [-.1, 3.]])
    b = np.array([[1., -.2, 0.], [.5, .1, 0.]])
    c = np.array([[.1, -.3], [.2, .1], [.4, -.2]])
    a = np.array([[2., -1., 1.], [-1., 2., 1.], [.4, .6, 0.]])
    h = a-c@np.linalg.solve(m,b)
    reduced = m-b@np.linalg.solve(a,c)
    apply = reduced_schur_preconditioner(lambda v:np.linalg.solve(m,v),
        lambda v:b@v, lambda v:c@v, lambda v:np.linalg.solve(h,v))
    rhs = np.array([.7, -.2])
    np.testing.assert_allclose(reduced@apply(rhs), rhs, rtol=1.e-12, atol=1.e-13)


def test_host_krylov_accepts_read_only_operator_outputs_without_mutating_them():
    from scipy.sparse.linalg import gmres
    from thermal_schur_probe import host_linear_operator
    matrix=np.array([[3., .4], [-.2, 2.]])
    rhs=np.array([.7, -.3])
    outputs=[]
    def readonly_action(v):
        image=matrix@v
        image.setflags(write=False)
        outputs.append((image, image.copy()))
        return image
    x,info=gmres(host_linear_operator(readonly_action,2),rhs,
                 restart=2,maxiter=1,rtol=1.e-12,atol=0.)
    assert info == 0
    np.testing.assert_allclose(matrix@x,rhs,rtol=1.e-12,atol=1.e-13)
    flexible, report=bounded_fgmres(readonly_action,rhs,lambda v:v,budget=2,rtol=1.e-12)
    assert report['true_relative'] < 1.e-12
    np.testing.assert_allclose(matrix@flexible,rhs,rtol=1.e-12,atol=1.e-13)
    for actual,original in outputs:
        np.testing.assert_array_equal(actual,original)


def test_selected_primitive_b_action_maps_by_field_name_and_keeps_vorticity_repair():
    """The primitive inverse has density first; target coordinates do not."""
    problem = _capture_problem(TARGET_FIELDS)
    primitive = np.arange(15, dtype=float).reshape(5, 3) + 0.25
    vorticity = np.array([-7.0, 8.0, 9.0])
    full_response = _primitive_response_to_full_material(primitive, vorticity)

    selected = problem.reduce_material(full_response)
    expected = np.vstack((primitive[1:], vorticity))
    np.testing.assert_array_equal(selected, expected)
    assert selected.shape == (5, 3)

    # A selected B action must never return a density coordinate.  Expanding
    # its result restores the frozen density base while retaining every
    # primitive row and the vorticity repair in their authoritative lanes.
    expanded = problem.expand_material(selected)
    np.testing.assert_array_equal(expanded[0], problem.adapter.base_material[0])
    np.testing.assert_array_equal(expanded[1:], expected)


def test_reduced_jacobian_blocks_use_selected_b_and_c_rows_without_density_leakage():
    """Exercise the runner's matrix-free B/C row views on a tiny synthetic action."""
    from run_reduced_implicit_capture import CaptureReducedProblem, FIELDS
    from types import SimpleNamespace

    problem = _capture_problem(TARGET_FIELDS)
    n = problem.adapter.n_active
    problem.adapter.active_flat = np.arange(n)
    problem.adapter.size = n
    problem.adapter.shape = (n,)
    problem.full_size = 7 * n + 1
    problem.nz = n
    problem.material_rows = np.concatenate(
        [i * n + problem.adapter.active_flat for i in problem.selected_indices]
    )
    problem.algebraic_rows = np.arange(6 * n, 7 * n + 1)
    problem.latest_trial = SimpleNamespace(z=np.array([0.1, 0.2, 0.3]))
    problem.calls = {}
    problem.full_eval = lambda material, z: (
        np.zeros(problem.full_size), None, np.zeros(problem.full_size)
    )
    problem._build_material_preconditioner = lambda vector: None
    problem._embed_z_direction = lambda direction: np.asarray(direction, dtype=float)

    b_matrix = np.arange(15 * n, dtype=float).reshape(15, n) / 7.0
    m_matrix = np.diag(np.arange(1.0, 16.0))
    c_matrix = np.arange((n + 1) * (5 * n), dtype=float).reshape(n + 1, 5 * n) / 11.0

    def fake_jvp(vector, tangent, lane):
        del vector
        tangent = np.asarray(tangent, dtype=float).ravel()
        image = np.zeros(problem.full_size)
        if lane == "algebraic":
            image[problem.material_rows] = b_matrix @ tangent
        else:
            selected = tangent.reshape(6, n)[list(problem.selected_indices)].ravel()
            image[problem.material_rows] = m_matrix @ selected
            image[problem.algebraic_rows] = c_matrix @ selected
        return image

    problem._full_jvp = fake_jvp
    material = np.arange(problem.nu, dtype=float) + 1.0
    blocks = CaptureReducedProblem.jacobian_blocks(problem, material, problem.latest_trial.z)
    z = np.array([0.3, -0.5, 0.7])
    du = np.arange(problem.nu, dtype=float) - 0.4

    np.testing.assert_allclose(blocks.B(z), b_matrix @ z)
    np.testing.assert_allclose(blocks.M(du), m_matrix @ du)
    np.testing.assert_allclose(blocks.C(du), c_matrix @ du)
    assert problem.material_rows.size == 5 * n
    assert all(FIELDS[i] != "density" for i in problem.selected_indices)


def test_thermal_c_feedback_selects_te_ti_vi_by_target_positions():
    """The thermal Schur C term excludes Ve and vorticity in five-field order."""
    problem = _capture_problem(TARGET_FIELDS)
    selected = np.arange(problem.nu, dtype=float).reshape(5, 3) + 0.5
    thermal = _thermal_c_input(problem, selected)

    np.testing.assert_array_equal(thermal[0], selected[0])  # Te
    np.testing.assert_array_equal(thermal[1], selected[1])  # Ti
    np.testing.assert_array_equal(thermal[2], selected[2])  # Vi
    np.testing.assert_array_equal(thermal[3:], 0.0)  # Ve, vorticity

    # Use distinct coefficients so an off-by-one slice cannot pass by
    # accidental equality.  This is the selected-coordinate C contribution.
    c = np.array([
        [2.0, -1.0, 0.5, 1.0, 0.25, -3.0, 4.0, 0.75, -2.0],
        [-1.5, 3.0, -2.0, 0.5, 2.5, 1.25, -0.25, 5.0, 1.0],
    ])
    expected = c @ selected[:3].ravel()
    np.testing.assert_allclose(c @ thermal[:3].ravel(), expected)


def test_runtime_thermal_mapping_helpers_match_five_and_six_field_layouts():
    from thermal_reduced_capture import (
        _primitive_to_selected,
        _selected_to_thermal_full,
    )

    primitive = np.arange(20, dtype=float).reshape(5, 4)
    active = np.array([0, 2, 3])
    five = _primitive_to_selected(primitive, TARGET_FIELDS, active, 4)
    np.testing.assert_array_equal(five[:4], primitive[1:, active])
    np.testing.assert_array_equal(five[4], 0.0)

    six_fields = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")
    six = _primitive_to_selected(primitive, six_fields, active, 4)
    np.testing.assert_array_equal(six[:5], primitive[:, active])
    np.testing.assert_array_equal(six[5], 0.0)

    selected = np.arange(15, dtype=float).reshape(5, 3) + 1.0
    full = _selected_to_thermal_full(selected, TARGET_FIELDS, 3).reshape(6, 3)
    np.testing.assert_array_equal(full[0], 0.0)
    np.testing.assert_array_equal(full[1:4], selected[:3])
    np.testing.assert_array_equal(full[4:], 0.0)


def test_thermal_coordinate_pack_unpack_round_trip_preserves_six_field_default():
    """Selected material packing is reversible and the historical six lanes remain unchanged."""
    from run_reduced_implicit_capture import FIELDS

    six = _capture_problem(FIELDS)
    six_material = np.arange(18, dtype=float).reshape(6, 3) - 2.0
    packed = six.reduce_material(six_material)
    np.testing.assert_array_equal(six.expand_material(packed), six_material)
    np.testing.assert_array_equal(six.reduce_material(six.expand_material(packed)), packed)

    five = _capture_problem(TARGET_FIELDS)
    five_material = np.arange(15, dtype=float).reshape(5, 3) + 11.0
    five_expanded = five.expand_material(five_material)
    np.testing.assert_array_equal(five.reduce_material(five_expanded), five_material)
    np.testing.assert_array_equal(five_expanded[0], five.adapter.base_material[0])

    # Tangent packing is distinct from material packing: excluded density is
    # zero, rather than the base density, and all selected lanes are retained.
    tangent = five.embed_material_direction(five_material.ravel()).reshape(6, 3)
    np.testing.assert_array_equal(tangent[0], 0.0)
    np.testing.assert_array_equal(tangent[1:], five_material)
    assert five.zero_leakage_check(five_material.ravel())["passed"]
