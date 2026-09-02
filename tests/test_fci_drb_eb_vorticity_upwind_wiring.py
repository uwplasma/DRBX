"""Focused wiring checks for causal FCI vorticity parallel advection."""

from __future__ import annotations

import inspect

from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs


def test_fci_parallel_terms_builds_raw_vorticity_stencil_and_upwind_action() -> None:
    source = inspect.getsource(LocalFciDrbEBRhs._fci_parallel_terms)

    # The scalar action must receive the mapped vorticity stencil, not the
    # legacy compatible gradient.  In particular this confirms the wall trace
    # is prepared through the same operator-level q path as other FCI fields.
    assert 'fields["vorticity"][owned]' in source
    assert 'traces["vorticity"]' in source
    assert "build_local_fci_stencil_from_field" in source
    assert "parallel_vorticity_upwind_residual" in source
    assert "vorticity_stencil.minus" in source
    assert "vorticity_stencil.plus" in source
    assert "vorticity_stencil.dx_min" in source
    assert "vorticity_stencil.dx_plus" in source
    assert "center[..., 3]" in source
    assert '"vorticity_parallel_advection": vorticity_parallel_advection' in source


def test_rhs_uses_upwind_lane_only_for_fci_production_and_keeps_legacy_fallback() -> None:
    source = inspect.getsource(LocalFciDrbEBRhs.evaluate_stage)

    # Production FCI consumes the scalar action; coordinate and legacy FCI
    # configurations retain the existing compatible-gradient expression.
    assert 'stage_parallel_terms["vorticity_parallel_advection"]' in source
    assert 'self.parallel_operator_scheme == "fci" and production_parallel' in source
    assert "-Vi * grad_parallel_vorticity" in source
