"""Step-5.3 combined campaign on real HSX data, bounded (slow): the reference stage's compute and the JAX stage's
data wiring with the **final operator options** on the N32 owner closure (12 owners, no pool, no full-grid run).

* ``test_reference_chunk_on_the_closure_owners``: :func:`references.reference_chunk` (the pool-free core of the reference
  stage) on the closure owners for the ``main`` field set: every array finite and of the owners' shape; the positive
  owner operator ``O(psi)`` and the autodiff midpoint reference ``R_mid(psi)`` agree to the N32 truncation level
  (``O - R_mid`` is the diffusion discretisation error of the reference construction, not a roundoff check).
* ``test_closure_wiring_of_the_arms``: on the closure plan, ``jaxstage.variant_boundary_data`` equals the five columns
  of ``JaxOwnerClosure``'s own P06N boundary data; the exported Dirichlet P07 operator applied to the frozen owner
  average ``psi_bar = phi_bar + tau Ti_bar`` plus ``B g`` (``g`` = exact trace of psi) reproduces the reference
  ``O(psi)`` at the closure owners to the N32 truncation level (this also fixes the SIGN of ``O``: a sign error would give a
  relative difference of 2), and the prescribed arm (``perpendicular_rhs`` with the frozen ``phi_bar``) is within the
  step-3 G3.3 accuracy of the re-frozen references.

Needs the HSX N32 geometry / sidecar and the frozen P06N N32 oracle arrays (skipped otherwise). A few minutes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
CAMPAIGNS = ("p06n",)
FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}
VARIANT = "main_phi_dirichlet"

pytestmark = pytest.mark.slow

_have = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file()
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


def _paths():
    from p_shared.replay_support import DEFAULT_PATHS
    return dict(DEFAULT_PATHS)


@pytest.fixture(scope="module")
def world():
    from p_shared import jax_replay as jr
    from p_shared import owner_closure as oc
    from p_shared import perpendicular_reference_rhs as prr
    from p_shared import replay_units as ru
    from p_shared.replay_support import build_environment
    from p08_step5_combined import campaign, references

    if not oc.oracle_available(_paths(), CAMPAIGNS, n=N):
        pytest.skip("the frozen P06N N32 oracle arrays are unavailable")
    cfg = campaign.config()
    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, **FINAL)
    owners = np.asarray(sorted(set(oc.select_owners(env.t, env.census).values())), dtype=np.int64)
    states = {"main": prr.p06n_state(env, cfg["field_sets"]["main"]["representative"])}
    ref = references.reference_chunk(env, owners, states=states, params=cfg["params"])
    built = oc.build_owner_rows(env, owners.tolist(), provider=oc.load_provider_for_env(
        SIDECAR, curvature=FINAL["curvature"], face_quadrature=FINAL["face_quadrature"]))
    oracle = ru._load_oracle_owner_values(env, _paths(), CAMPAIGNS)
    closure = jr.JaxOwnerClosure(env, built, CAMPAIGNS, oracle)
    return dict(cfg=cfg, env=env, owners=owners, ref=ref, closure=closure, oracle=oracle, prr=prr)


@needs_inputs
def test_reference_chunk_on_the_closure_owners(world):
    from p08_step5_combined import references
    from p_shared.replay_support import owner_weighted_l2

    ref, owners, env = world["ref"], world["owners"], world["env"]
    vol = np.asarray(env.t.vol)[owners]
    assert all(v.shape == owners.shape and np.all(np.isfinite(v)) for v in ref.values())
    np.testing.assert_array_equal(ref["owners"], owners)
    for field in references.FIELDS:
        for term in references.REF_TERMS:
            if (field, term) == ("vorticity", "curvature_remainder"):
                continue                                    # omega has no curvature remainder (coefficient 0)
            assert owner_weighted_l2(ref[references.ref_key("main", field, term)], vol) > 0, (field, term)
    o, r = ref[references.psi_key("main", "O")], ref[references.psi_key("main", "R_mid")]
    rel = owner_weighted_l2(o - r, vol) / owner_weighted_l2(r, vol)
    print("\nO(psi) - R_mid(psi) relative L2 at the N32 closure owners: %.3e" % rel)
    assert rel < 0.1 and owner_weighted_l2(o + r, vol) > 1.5 * owner_weighted_l2(o - r, vol)   # not the opposite sign


@needs_inputs
def test_closure_wiring_of_the_arms(world):
    from drbx.native.fci_perpendicular_p06_operator import bc_columns
    from drbx.native.fci_perpendicular_p07_sparse import boundary_source, export_p07_sparse
    from drbx.native.fci_perpendicular_rhs import FIELDS, PHI, PerpendicularParams, perpendicular_rhs
    from p08_step5_combined import jaxstage, references
    from p_shared.replay_support import owner_weighted_l2

    cfg, env, owners, ref, closure, prr = (world[k] for k in ("cfg", "env", "owners", "ref", "closure", "prr"))
    plan = closure.plan
    adapter = closure.adapters["p06n"]
    rec = adapter.reconstructions[VARIANT]
    columns = np.asarray(rec.columns)
    prep = SimpleNamespace(plan=plan, env=env, exact_state=lambda v: prr.p06n_state(env, v), boundary_batch=8192)
    bc5 = jaxstage.variant_boundary_data(prep, VARIANT)
    theirs = bc_columns(closure.bc["p06n"], columns)
    np.testing.assert_allclose(np.asarray(bc5.dirichlet_value), np.asarray(theirs.dirichlet_value), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(bc5.dirichlet_tangential), np.asarray(theirs.dirichlet_tangential),
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(bc5.neumann_normal), np.asarray(theirs.neumann_normal), rtol=0, atol=1e-13)

    tau = float(cfg["params"]["tau"])
    values = np.asarray(adapter.owner_values)
    phi_bar, ti_bar = values[:, columns[4]], values[:, columns[2]]
    psi_bar = phi_bar + tau * ti_bar
    bc_psi = jaxstage.psi_boundary_data(bc5, tau)
    op = export_p07_sparse(plan, "dirichlet")
    n_psi = (op.matrix @ psi_bar + boundary_source(op, bc_psi)[:, 0])[owners]
    vol = np.asarray(env.t.vol)[owners]
    o = ref[references.psi_key("main", "O")]
    rel = owner_weighted_l2(n_psi - o, vol) / owner_weighted_l2(o, vol)
    print("\nN(psi_bar) - O(psi) relative L2 at the N32 closure owners: %.3e" % rel)
    assert rel < 0.1 and owner_weighted_l2(n_psi + o, vol) > 1.5 * owner_weighted_l2(n_psi - o, vol)

    params = PerpendicularParams(rho_star=0.05, tau=tau, diffusion={f: 0.01 for f in FIELDS})
    state = {name: np.asarray(values[:, c]) for name, c in zip(FIELDS, columns[:4])}
    kinds = dict(zip((*FIELDS, PHI), rec.field_kinds))
    out = jaxstage.collect(perpendicular_rhs(plan, state, phi_bar, bc5, kinds, params))
    worst = 0.0
    for field in FIELDS:
        scale = max(owner_weighted_l2(ref[references.ref_key("main", field, t)], vol)
                    for t in ("poisson_bracket", "curvature", "perpendicular_diffusion"))
        total = out[(field, "total")][owners] - ref[references.ref_key("main", field, "total")]
        worst = max(worst, owner_weighted_l2(total, vol) / scale)
    print("prescribed arm total vs re-frozen reference, worst relative L2 over the fields: %.3e" % worst)
    assert worst < 0.1
