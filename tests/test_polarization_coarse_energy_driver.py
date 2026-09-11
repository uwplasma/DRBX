"""Contract tests for the full-operator coarse setup in the HSX driver."""

from pathlib import Path


DRIVER = Path(__file__).resolve().parents[1] / "simulate_hsx_blob.py"


def _coarse_setup_source() -> str:
    source = DRIVER.read_text()
    start = source.index('    if gmres_preconditioner in ("coarse-additive", "coarse-multiplicative"):')
    end = source.index("    rung3_wall_layer_effective_cells =", start)
    return source[start:end]


def test_coarse_callback_uses_full_physical_operator():
    source = _coarse_setup_source()

    callback_start = source.index("apply_host_operator =")
    callback_end = source.index("        coarse_data = build_coarse_data(", callback_start)
    callback = source[callback_start:callback_end]

    assert "_apply_A(" in callback
    assert "face_bc=host_hface" in callback
    assert "control_volume_boundary_bc=host_hcv" in callback
    assert "project_mean_zero=True" in callback
    assert "boundary_is_homogeneous=True" in callback
    assert "include_boundary_flux=False" not in callback
    assert "_apply_A_support_paired(" not in callback
    assert "apply_host_operator(values)" in source
