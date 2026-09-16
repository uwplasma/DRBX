"""Focused wiring checks for causal FCI vorticity parallel advection."""

from __future__ import annotations

import inspect

from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs


def test_fci_parallel_terms_builds_h_second_order_vorticity_with_directional_fallback() -> None:
    source = inspect.getsource(LocalFciDrbEBRhs._fci_parallel_terms)

    # Keep the canonical mapped P action only for an invalid immediate H/MF
    # representation.  A missing selected second hop falls that H/MF direction
    # to first order without discarding a valid opposite-direction hop.
    assert 'fields["vorticity"][owned]' in source
    assert 'traces["vorticity"]' in source
    assert "build_local_fci_stencil_from_field" in source
    assert "parallel_vorticity_upwind_residual" in source
    assert "_fci_second_order_vorticity_data" in source
    assert "parallel_vorticity_second_order_upwind_residual" in source
    assert 'center[..., 3]' in source
    assert 'vorticity_second_order["minus2"]' in source
    assert 'vorticity_second_order["plus2"]' in source
    assert '"first_row_valid"' in source
    assert '"backward_second_valid"' in source
    assert '"forward_second_valid"' in source
    assert "vorticity_h_second_order_used" in source
    assert "vorticity_h_first_order_fallback_used" in source
    assert "vorticity_legacy_p_fallback_used" in source
    assert "vorticity_stencil.minus" in source
    assert "vorticity_stencil.plus" in source
    assert "vorticity_stencil.dx_min" in source
    assert "vorticity_stencil.dx_plus" in source
    assert '"vorticity_parallel_advection": vorticity_parallel_advection' in source


def test_h_second_order_vorticity_is_a_static_opt_in() -> None:
    source = inspect.getsource(LocalFciDrbEBRhs._fci_parallel_terms)

    assert "self.parallel_vorticity_advection_scheme" in source
    assert '== "h-mf-second-order"' in source
    assert "else None" in source


def test_vorticity_h_builder_applies_prolongation_once_before_all_maps() -> None:
    source = inspect.getsource(LocalFciDrbEBRhs._fci_second_order_vorticity_data)

    assert source.count("apply_rlp_cell_average_prolongation(") == 1
    assert source.count("_fci_material_fine_stencil(") == 3
    assert "first.minus" in source
    assert "first.plus" in source
    assert "_fci_prepare_q" not in source
    assert 'second_order_material["backward_second_valid"]' in source
    assert 'second_order_material["forward_second_valid"]' in source


def test_rhs_uses_upwind_lane_only_for_fci_production_and_keeps_legacy_fallback() -> None:
    source = inspect.getsource(LocalFciDrbEBRhs.evaluate_stage)

    # Production FCI consumes the scalar action; coordinate and legacy FCI
    # configurations retain the existing compatible-gradient expression.
    assert 'stage_parallel_terms["vorticity_parallel_advection"]' in source
    assert 'self.parallel_operator_scheme == "fci" and production_parallel' in source
    assert "-Vi * grad_parallel_vorticity" in source
