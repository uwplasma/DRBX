"""Package geometry bundle of the nodal SBP perpendicular stack (``drbx.native.fci_nodal_geometry_bundle``): assembly, identity,
round trip and the portable-identity rule (m1), on the n = 16 family-A synthetic inputs of the P10 harness (no HSX files).

The P10 harness ``bundle.py`` is a thin layer over this module; ``test_p10_evolved_bundle.py`` (unchanged) covers it through the
harness API, the tests here pin the package API, the equality of the two layers and the cross-platform behaviour.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.native import fci_nodal_geometry_bundle as GB                    # noqa: E402
from drbx.native.fci_nodal_perpendicular_model import NODAL_PERPENDICULAR_DEFAULTS as DEFAULTS   # noqa: E402
from p10_evolved_mms import bundle as P10                                  # noqa: E402
from p10_evolved_mms import synthetic as syn                               # noqa: E402

TIMINGS = ("plans_seconds", "build_seconds")


def _inputs(arm="raw"):
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = syn.synthetic_inputs(16, 8, arm)
    return arm, layout, nodal, copy.deepcopy(nodal_meta), dict(lap), copy.deepcopy(lap_meta), dict(prov)


def _assemble(arm="raw", **kw):
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs(arm)
    return GB.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, lap, lap_meta, prov, build_preconditioner=False, **kw)


def _stable(identity: dict) -> dict:
    """The identity without the git commit and the preconditioner timings."""
    out = {k: v for k, v in identity.items() if k != "git_commit"}
    out["preconditioner"] = {k: v for k, v in identity["preconditioner"].items() if k not in TIMINGS + ("info",)}
    return GB.jsonable(out)


@pytest.fixture(scope="module")
def b():
    return _assemble("filtered")


# ----------------------------------------------------------------------------------------------------- the API
def test_package_identity_and_pytree(b):
    idn = b.identity
    assert idn["schema"] == GB.NODAL_BUNDLE_SCHEMA and idn["family"] == DEFAULTS.layout_family == "A"
    assert idn["arm"] == "filtered" and idn["n"] == 16 and idn["n_eta"] == 8 and idn["git_commit"] == "unknown"
    assert idn["preconditioner"]["method"] is None                                  # build_preconditioner=False
    assert idn["agreement"] == {"jac": "bitwise", "Hp": "bitwise"}
    assert idn["bundle_sha256"] == GB.bundle_sha(idn) and re.fullmatch(r"[0-9a-f]{64}", idn["arrays_sha256"])
    assert (b.E, b.P, b.N) == (8, 254, 16) and b.arm == "filtered" and b.n == 16
    leaves, tree = jax.tree_util.tree_flatten(b)                                    # ctx and ref are traced, static is not
    assert jax.tree_util.tree_unflatten(tree, leaves).static is b.static
    f = jax.jit(lambda bundle: jnp.sum(bundle.ref.jac))
    f(b), f(b)
    assert f._cache_size() == 1
    # the commit and the schema are the caller's and enter as documented (the commit is volatile, the schema is not)
    other = _assemble("filtered", schema="x-schema", git_commit="abc")
    assert other.identity["git_commit"] == "abc" and other.identity["schema"] == "x-schema"
    assert other.identity["bundle_sha256"] != b.identity["bundle_sha256"]
    assert _assemble("filtered", git_commit="def").identity["bundle_sha256"] == b.identity["bundle_sha256"]


def test_default_preconditioner_is_the_accepted_one_and_is_recorded():
    bundle = syn.synthetic_bundle()                                                 # prec built through the harness layer
    info = bundle.identity["preconditioner"]
    assert {k: info[k] for k in ("method", "factor_dtype")} == DEFAULTS.preconditioner == \
        {"method": "core_schur", "factor_dtype": "float32"}
    assert info["info"]["factor_dtype"] == "float32"
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs("raw")
    pkg = GB.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, lap, lap_meta, prov)       # package defaults, prec built
    assert pkg.identity["preconditioner"]["method"] == "core_schur" and pkg.ctx.prec is not None
    assert pkg.identity["preconditioner"]["factor_dtype"] == "float32"


def test_harness_layer_gives_the_package_identity_and_arrays_bit_for_bit():
    """The P10 bundle is the package assembly with the P10 schema: same arrays, same identity, hence the same ``bundle_sha256``."""
    for arm in ("raw", "filtered"):
        arm_, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs(arm)
        harness = P10.assemble_bundle(arm_, layout, nodal, nodal_meta, lap, lap_meta, prov, build_preconditioner=False)
        pkg = GB.assemble_nodal_bundle(arm_, layout, nodal, copy.deepcopy(nodal_meta), lap, copy.deepcopy(lap_meta), prov,
                                       schema=P10.SCHEMA, build_preconditioner=False, git_commit=P10.git_commit())
        assert harness.identity["schema"] == "drbx.p10-bundle-v1"
        assert _stable(harness.identity) == _stable(pkg.identity)
        assert harness.identity["bundle_sha256"] == pkg.identity["bundle_sha256"]
        for name in GB.NodalReferenceArrays._fields:
            assert np.array_equal(np.asarray(getattr(harness.ref, name)), np.asarray(getattr(pkg.ref, name))), name
    assert P10.RefArrays is GB.NodalReferenceArrays and P10.Bundle is GB.NodalGeometryBundle


def test_identity_helpers_are_the_p09_campaign_helpers(b):
    """The package re-implements ``layout_identity`` / ``plan_identity`` / ``sha256_arrays`` of ``scripts/p09_sbp_laplacian/common.py``
    (the identities every saved P09 / P10 artifact records); they must agree exactly."""
    C = P10.laplacian_common()
    assert GB.layout_identity(b.layout) == C.layout_identity(b.layout)
    assert GB.laplacian_plan_identity(b.ctx.lplan) == C.plan_identity(b.ctx.lplan)
    arrays = {"a": np.arange(6.0).reshape(2, 3), "b": np.ones(4, dtype=np.float32)}
    assert GB.sha256_arrays(arrays) == C.sha256_arrays(arrays) and GB.sha256_arrays(arrays, "x") == C.sha256_arrays(arrays, "x")


def test_round_trip_schema_and_expected_identity(tmp_path, b):
    out = GB.save_nodal_bundle(b, tmp_path / "bundle")
    assert sorted(p.name for p in out.iterdir()) == ["bundle.npz", "identity.json"]
    c = GB.load_nodal_bundle(out, expected_identity=b.identity["bundle_sha256"], build_preconditioner=False)
    for name in GB.NodalReferenceArrays._fields:
        assert np.array_equal(np.asarray(getattr(c.ref, name)), np.asarray(getattr(b.ref, name))), name
    assert _stable(c.identity) == _stable(b.identity) and "rebuilt" not in c.identity       # same platform: shas reproduced
    with pytest.raises(ValueError, match="schema"):
        GB.load_nodal_bundle(out, schema="drbx.other-v1", build_preconditioner=False)
    with pytest.raises(ValueError, match="expected"):
        GB.load_nodal_bundle(out, expected_identity="0" * 64, build_preconditioner=False)
    with pytest.raises(ValueError, match="mismatch"):
        GB.load_nodal_bundle(out, expected_identity={"arm": "raw"}, build_preconditioner=False)
    # the schema is the campaign's: the P10 loader does not read a package-schema bundle
    with pytest.raises(ValueError, match="schema"):
        P10.load_bundle(out, build_preconditioner=False)


# ----------------------------------------------------------------------------------------------------- m1: portable identity
def test_m1_a_differing_stored_layout_sha_is_recorded_and_the_agreement_is_checked_to_tolerance():
    """Float-byte shas hash LAPACK-derived Gauss weights, so they differ between platforms: a stored sha that differs is provenance
    only. What decides is the tolerance check of nodes, wall points, |J| and Hp."""
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs("filtered")
    same = GB.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, lap, {**lap_meta, "layout_sha256": GB.layout_identity(layout)},
                                    prov, build_preconditioner=False)
    assert same.identity["agreement"]["layout"] == "sha equal"
    moved = GB.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, lap, {**lap_meta, "layout_sha256": "0" * 64}, prov,
                                     build_preconditioner=False)
    assert "differs" in moved.identity["agreement"]["layout"] and "1e-13" in moved.identity["agreement"]["layout"]
    # ... and a matching sha does not excuse a geometry that disagrees beyond the tolerance
    pts = np.array(lap["points"])
    pts[..., 1] += 1e-6
    with pytest.raises(ValueError, match="nodes"):
        GB.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, {**lap, "points": pts},
                                 {**lap_meta, "layout_sha256": GB.layout_identity(layout)}, prov, build_preconditioner=False)
    near = np.array(lap["points"])
    near[..., 1] += 5e-15                                                            # ulp-level platform difference: accepted
    GB.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, {**lap, "points": near}, {**lap_meta, "layout_sha256": "0" * 64}, prov,
                             build_preconditioner=False)
    jac = {**lap, "J": lap["J"] * (1.0 + 1e-6 * np.cos(np.arange(lap["J"].size)).reshape(lap["J"].shape))}
    with pytest.raises(ValueError, match=r"\|J\|"):
        GB.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, jac, lap_meta, prov, build_preconditioner=False)


def _as_built_elsewhere(path: Path, **replace) -> Path:
    """Rewrite ``identity.json`` as if the bundle had been built on a platform whose layout / plan shas differ (the stored
    ``bundle_sha256`` is consistent with the stored identity, as it is for a real bundle)."""
    idn = json.loads((path / "identity.json").read_text())
    idn.update(replace)
    idn["bundle_sha256"] = GB.bundle_sha(idn)
    (path / "identity.json").write_text(json.dumps(idn, indent=1, sort_keys=True))
    return path


def test_m1_load_records_the_rebuilt_shas_of_a_bundle_built_on_another_platform(tmp_path, b):
    foreign = dict(layout_sha256="a" * 64, nodal_plan_sha256="b" * 64, laplacian_plan_sha256="c" * 64)
    out = _as_built_elsewhere(GB.save_nodal_bundle(b, tmp_path / "bundle"), **foreign)
    stored = json.loads((out / "identity.json").read_text())
    c = GB.load_nodal_bundle(out, expected_identity=stored["bundle_sha256"], build_preconditioner=False)
    # the stored identity of the build is kept ...
    assert c.identity["bundle_sha256"] == stored["bundle_sha256"] != b.identity["bundle_sha256"]
    for key, value in foreign.items():
        assert c.identity[key] == value
    # ... and the shas of this platform's rebuild are recorded (volatile: not part of the bundle sha)
    rebuilt = c.identity["rebuilt"]
    assert rebuilt["differs"] == sorted(foreign)
    for key in foreign:
        assert rebuilt[key] == b.identity[key]
    assert GB.bundle_sha(c.identity) == stored["bundle_sha256"]
    for name in GB.NodalReferenceArrays._fields:
        assert np.array_equal(np.asarray(getattr(c.ref, name)), np.asarray(getattr(b.ref, name))), name
    # a single differing sha is enough, the equal ones are not listed
    out2 = _as_built_elsewhere(GB.save_nodal_bundle(b, tmp_path / "bundle2"), laplacian_plan_sha256="d" * 64)
    assert GB.load_nodal_bundle(out2, build_preconditioner=False).identity["rebuilt"]["differs"] == ["laplacian_plan_sha256"]
    # same-platform strictness is available on request
    with pytest.raises(ValueError, match="rebuilt nodal_plan_sha256"):
        GB.load_nodal_bundle(out, build_preconditioner=False, strict_plan_shas=True)
    GB.load_nodal_bundle(GB.save_nodal_bundle(b, tmp_path / "bundle3"), build_preconditioner=False, strict_plan_shas=True)


def test_m1_what_is_verified_bit_for_bit_is_the_stored_data(tmp_path, b):
    """The shas that are not portable are not gates; the stored arrays and the stored identity are."""
    out = _as_built_elsewhere(GB.save_nodal_bundle(b, tmp_path / "bundle"), layout_sha256="a" * 64)
    with np.load(out / "bundle.npz") as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    arrays["B"] = arrays["B"] * (1.0 + 1e-9)
    np.savez(out / "bundle.npz", **arrays)
    with pytest.raises(ValueError, match="arrays_sha256"):
        GB.load_nodal_bundle(out, build_preconditioner=False)
    out2 = _as_built_elsewhere(GB.save_nodal_bundle(b, tmp_path / "bundle2"), layout_sha256="a" * 64)
    idn = json.loads((out2 / "identity.json").read_text())
    idn["arm"] = "raw"                                                                # identity edited without its sha
    (out2 / "identity.json").write_text(json.dumps(idn))
    with pytest.raises(ValueError, match="bundle_sha256"):
        GB.load_nodal_bundle(out2, build_preconditioner=False)
    out3 = GB.save_nodal_bundle(b, tmp_path / "bundle3")                             # an entry that changes the geometry identity
    idn = json.loads((out3 / "identity.json").read_text())
    idn["nodal_metric_identity"] = "f" * 64
    (out3 / "identity.json").write_text(json.dumps(idn))
    with pytest.raises(ValueError, match="bundle_sha256"):
        GB.load_nodal_bundle(out3, build_preconditioner=False)
