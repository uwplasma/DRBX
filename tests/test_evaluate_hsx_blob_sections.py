"""Regression tests for constant-eta HSX section surface topology."""

import sys

import numpy as np

sys.path.insert(0, str(__file__.split("/DRBX/tests")[0]))

import evaluate_hsx_blob as evaluate


def test_section_scalar_surface_renders_both_faces_without_lighting():
    class Plotter:
        def __init__(self):
            self.calls = []

        def add_mesh(self, surface, **kwargs):
            self.calls.append((surface, kwargs))

    plotter = Plotter()
    surface = object()
    evaluate._add_two_sided_scalar_surface(
        plotter,
        surface,
        scalar_name="phi",
        colorscale="coolwarm",
        color_limits=(-1.0, 1.0),
        opacity=0.9,
        scalar_bar_args={"title": "phi"},
    )

    assert [call[1]["culling"] for call in plotter.calls] == ["back", "front"]
    assert [call[1]["show_scalar_bar"] for call in plotter.calls] == [True, False]
    assert all(call[1]["lighting"] is False for call in plotter.calls)
    assert all(call[0] is surface for call in plotter.calls)


def test_full_torus_position_replication_rotates_field_periods():
    positions = np.zeros((2, 2, 1, 3), dtype=np.float64)
    positions[..., 0] = 1.0

    full = evaluate._replicate_positions_full_torus(positions, 4, 1)

    assert full.shape == (2, 2, 4, 3)
    assert np.allclose(
        full[0, 0, :, :2],
        np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))),
        atol=1.0e-14,
    )


def test_shared_camera_is_centered_on_reference_geometry():
    points = np.asarray(((-2.0, -1.0, -0.5), (4.0, 3.0, 1.5)))

    camera, center, up = evaluate._camera_position_from_points(points)

    assert np.allclose(center, (1.0, 1.0, 0.5))
    assert np.linalg.norm(np.asarray(camera) - np.asarray(center)) > 0.0
    assert up == [0.0, 0.0, 1.0]


def test_eta_sections_are_evenly_distributed_around_full_torus():
    eta = (np.arange(64, dtype=np.float64) + 0.5) * 2.0 * np.pi / 64.0

    indices = evaluate._evenly_spaced_eta_indices(eta, section_count=4)

    assert np.array_equal(indices, np.asarray([0, 16, 32, 48]))


def test_eta_sections_default_to_one_field_period():
    eta = (np.arange(64, dtype=np.float64) + 0.5) * 2.0 * np.pi / 64.0

    indices = evaluate._section_eta_indices(
        eta,
        nfp=4,
        section_count=4,
        section_periods=1,
    )

    assert np.array_equal(indices, np.asarray([0, 4, 8, 12]))


def test_eta_sections_can_span_two_field_periods():
    eta = (np.arange(64, dtype=np.float64) + 0.5) * 2.0 * np.pi / 64.0

    indices = evaluate._section_eta_indices(
        eta,
        nfp=4,
        section_count=8,
        section_periods=2,
    )

    assert np.array_equal(indices, np.asarray([0, 4, 8, 12, 16, 20, 24, 28]))


def test_eta_sections_can_select_the_other_two_field_periods():
    eta = (np.arange(64, dtype=np.float64) + 0.5) * 2.0 * np.pi / 64.0

    indices = evaluate._section_eta_indices(
        eta,
        nfp=4,
        section_count=8,
        section_periods=2,
        section_start_period=2,
    )

    assert np.array_equal(indices, np.asarray([32, 36, 40, 44, 48, 52, 56, 60]))


def test_parser_defaults_section_periods_to_one():
    args = evaluate._build_parser().parse_args(("history.npz",))
    assert args.section_periods == 1
    assert args.section_start_period == 0


def test_parser_accepts_arbitrary_section_period_counts_through_four():
    parser = evaluate._build_parser()
    for periods in range(1, 5):
        args = parser.parse_args(("history.npz", "--section-periods", str(periods)))
        assert args.section_periods == periods


def test_retired_descriptive_section_spans_remain_accepted():
    parser = evaluate._build_parser()
    expected = {
        "field-period": 1,
        "two-field-periods": 2,
        "full-torus": 4,
    }
    for name, periods in expected.items():
        args = parser.parse_args(("history.npz", "--section-span", name))
        assert args.section_periods == periods


def test_periodic_stride_uses_unique_endpoint_exclusive_samples():
    assert np.array_equal(
        evaluate._periodic_stride_indices(8, 2),
        np.asarray([0, 2, 4, 6]),
    )
    assert np.array_equal(
        evaluate._periodic_stride_indices(8, 20),
        np.asarray([0, 7]),
    )


def test_closed_poloidal_columns_fill_the_structured_mesh_seam():
    positions = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
    closed = evaluate._close_periodic_columns(positions)

    assert closed.shape == (3, 5, 3)
    assert np.array_equal(closed[:, -1], closed[:, 0])

    vertices, triangles, _ = evaluate._combine_surface_grids(
        (("closed section", closed),)
    )
    assert vertices.shape == (15, 3)
    assert triangles.shape == (2 * (3 - 1) * 4, 3)

    # The final pair of cells reaches the repeated first-poloidal column.
    repeated_column = {4, 9, 14}
    assert repeated_column.intersection(set(triangles[-2:].ravel()))


def test_scalar_values_close_with_the_same_poloidal_column_order():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    closed = evaluate._close_periodic_columns(values)

    assert closed.shape == (3, 5)
    assert np.array_equal(closed[:, -1], values[:, 0])


def test_periodic_section_outline_has_no_radial_seam_segment():
    positions = np.zeros((3, 5, 2, 3), dtype=np.float32)
    theta = np.linspace(0.0, 2.0 * np.pi, 5)
    for radial_index, radius in enumerate((0.1, 0.5, 1.0)):
        positions[radial_index, :, :, 0] = radius * np.cos(theta)[:, None]
        positions[radial_index, :, :, 1] = radius * np.sin(theta)[:, None]
        positions[radial_index, :, 1, 2] = 1.0

    points, lines = evaluate._periodic_section_boundary_geometry(positions)

    assert points.shape == (10, 3)
    assert np.allclose(np.linalg.norm(points[:, :2], axis=1), 1.0)
    assert np.array_equal(lines[:6], np.asarray([5, 0, 1, 2, 3, 4]))
    assert np.array_equal(lines[6:], np.asarray([5, 5, 6, 7, 8, 9]))
