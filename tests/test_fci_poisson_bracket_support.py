"""Focused contracts for operator-specific perpendicular-advection halos."""

from __future__ import annotations

import ast
from pathlib import Path


RHS_SOURCE = Path(__file__).parents[1] / "src/drbx/native/fci_drb_EB_rhs.py"


def test_pb_support_halo_is_neumann_on_template_masks_and_preserves_owner():
    source = RHS_SOURCE.read_text()
    tree = ast.parse(source)
    helper = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_prepare_poisson_bracket_support_halo"
    )
    text = ast.get_source_segment(source, helper)
    assert text is not None
    assert "_prepare_poisson_bracket_support_face_bc(template_bc)" in text
    assert "_prepare_poisson_bracket_halo(" in text
    assert "positivity_floor" not in text
    face_helper = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_prepare_poisson_bracket_support_face_bc"
    )
    face_text = ast.get_source_segment(source, face_helper)
    assert face_text is not None
    assert "BC_NEUMANN" in face_text
    assert "replace(" in face_text
    assert all(f"template_bc.mask_{axis}" in face_text for axis in "xyz")


def test_pb_calls_use_support_halos_and_support_traces_with_vorticity_generator():
    tree = ast.parse(RHS_SOURCE.read_text())
    source = RHS_SOURCE.read_text()
    assert 'POISSON_BRACKET_SUPPORT_FIELD_NAMES = RHS_TERM_FIELD_NAMES' in source
    assert 'phi_pb_conservative_stencil = build_local_conservative_stencil_from_field(' in source
    assert 'phi_gradient = build_gradient(state_halo.phi, self.geometry, context)' in source
    assert 'phi_poisson_bracket_halo = self._prepare_poisson_bracket_halo(' in source
    assert 'phi_pb_gradient = build_gradient(' in source
    assert "poisson_bracket_support_traces" not in source
    assert "phi_conservative_stencil = phi_pb_conservative_stencil" in source
    calls = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id.startswith("poisson_")
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "_poisson_bracket_over_B"
    ]
    assert len(calls) == 6
    for call in calls:
        call_text = ast.get_source_segment(source, call)
        assert call_text is not None
        assert "phi_pb_gradient" in call_text
        assert "phi_pb_conservative_stencil" in call_text
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert isinstance(keywords["g_field_halo"], ast.Subscript)
        assert isinstance(keywords["g_field_halo"].value, ast.Name)
        assert keywords["g_field_halo"].value.id == "poisson_bracket_support_halos"
        assert isinstance(keywords["f_boundary_trace"], ast.Attribute)
        assert keywords["f_boundary_trace"].value.id == "operator_boundary"
        assert isinstance(keywords["g_boundary_trace"], ast.Attribute)
        assert keywords["g_boundary_trace"].value.id in {
            "operator_boundary",
            "perpendicular_operator_boundary",
        }
    vorticity = [
        c for c in calls
        if any(k.arg == "equation_family" for k in c.keywords)
    ]
    assert len(vorticity) == 1
    family = next(k.value for k in vorticity[0].keywords if k.arg == "equation_family")
    assert isinstance(family, ast.Constant) and family.value == "vorticity"
