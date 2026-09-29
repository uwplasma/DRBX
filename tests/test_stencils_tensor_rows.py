"""Exact tensor-factored storage of unconditioned point rows (P08 step 2, task D1).

Fast synthetic tests on a small toy grid (see ``work/p08_step2_layout_loader_design_20260929/design.md``
sections 5 and 6): the opt-in factor capture leaves the rows bitwise unchanged; every factored family
(singleton incl. wrapped layers and cross-plane aggregate owners, ringwise incl. an aggregate owner that
spans two planes, centered_radial) expands bit for bit (signed zeros included); the encoder falls back to CSR
for a source whose expansion does not reproduce its rows; v3 files without ``src_encoding`` still decode.
The real-geometry gate is in ``tests/test_stencils_tensor_rows_real.py`` (slow).
"""
from __future__ import annotations

import dataclasses
import io
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext, StructuredReconstruction
from drbx.stencils import artifact as art
from drbx.stencils import tensor_rows as tr_mod
from drbx.stencils.artifact import chunk_mismatches, decode_chunk, encode_chunk, expand_point_rows, pack_point_rows

N = 8


def _context(merges=()):
    faces = (np.linspace(0, 1, N + 1), np.linspace(0, 2 * np.pi, N + 1), np.linspace(0, 2 * np.pi, N + 1))
    centers = tuple((x[:-1] + x[1:]) / 2 for x in faces)
    ijk = np.array(np.unravel_index(np.arange(N ** 3), (N, N, N))).T
    pts = np.column_stack([centers[a][ijk[:, a]] for a in range(3)])
    xy = np.column_stack((pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1])))
    owner = np.arange(N ** 3)
    for group in merges:
        raws = [np.ravel_multi_index(c, (N, N, N)) for c in group]
        owner[raws] = owner[raws[0]]
    _, owner = np.unique(owner, return_inverse=True)
    volume = np.bincount(owner)
    centroid = np.column_stack([np.bincount(owner, weights=xy[:, a]) / volume for a in range(2)])
    return PointRowContext.from_arrays(
        faces=faces, centers=centers, raw_to_owner=owner, raw_volume=np.ones(N ** 3), owner_volume=volume.astype(float),
        owner_centroid_xy=centroid, eta_period=2 * np.pi, dr=1 / N, dtheta=2 * np.pi / N, deta=2 * np.pi / N)


#: layer 4 is a ring of 7 owners; one owner spans planes 0 and 1 -> ringwise, and it accumulates
RING_MERGE = [[(4, 0, 0), (4, 0, 1), (4, 1, 0)]]
#: all rings keep 8 owners but one owner spans two planes -> singleton with an accumulating owner
PLANE_MERGE = [[(4, 0, 0), (4, 0, 1)]]


def _jitter(context, key, q, rng, nodes=False):
    base = context.pts[np.ravel_multi_index(key, (N, N, N))]
    scale = np.array([1 / N, 2 * np.pi / N, 2 * np.pi / N])
    points = base + rng.uniform(-.3, .3, size=(q, 3)) * scale
    if nodes:                                            # targets exactly on nodes: one-hot factors, signed zeros
        points[0] = base
        if q > 1:
            points[1] = base + np.array([0, 2 * np.pi / N, 0])
    return points


def _sources(context, plan, seed=0):
    """Rows, factors and tags for ``plan`` = [(request, key, location, q, nodes)], packed like the builder."""
    rng = np.random.default_rng(seed)
    S = StructuredReconstruction(context)
    rows, factors, tags = [], [], []
    for request, key, location, q, nodes in plan:
        points = _jitter(context, key if location == "cell" else key[1:], q, rng, nodes)
        if request == "R3":
            for side, (row, fac) in enumerate(S.side_rows_with_factors(key, points)):
                if row is not None:
                    rows.append(row), factors.append(fac), tags.append(("R3", len(tags) * 2 + side))
        else:
            row, fac = S.rows_with_factors(key, points, location)
            rows.append(row), factors.append(fac), tags.append((request, len(tags)))
    chunk = pack_point_rows(
        rows, request=[t[0] for t in tags], entity_id=[t[1] for t in tags],
        bc_variant=["D" if r.boundary_conditioned else "" for r in rows],
        radial_degree=[3 if r.boundary_conditioned else 0 for r in rows],
        store_gradient=[t[0] != "R3" for t in tags])
    return S, rows, factors, chunk


IDENTITY_PLAN = [
    ("R1", (3, 2, 5), "cell", 1, False), ("R1", (0, 3, 2), "cell", 1, False),      # singleton; wrapped layer -1
    ("R1", (1, 0, 0), "cell", 1, False), ("R1", (7, 1, 1), "cell", 1, False),       # ...; conditioned (stays CSR)
    ("R2", (1, 2, 3, 4), "face", 9, True), ("R2", (0, 1, 2, 4), "face", 9, True),    # singleton, wrapped radial faces
    ("R2", (0, 3, 2, 4), "face", 9, True), ("R2", (0, 4, 1, 7), "face", 9, False),   # centered_radial
    ("R2", (0, 7, 2, 2), "face", 9, False), ("R2", (0, 0, 1, 1), "face", 9, False),  # quartic_wall, collapsed_r0
    ("R3", (1, 3, 2, 4), "face", 9, False), ("R3", (0, 1, 2, 4), "face", 9, True),
]
RING_PLAN = [
    ("R1", (3, 0, 0), "cell", 1, False), ("R1", (3, 0, 1), "cell", 1, True),
    ("R2", (1, 3, 1, 0), "face", 9, True), ("R2", (2, 4, 0, 1), "face", 9, False),
    ("R3", (1, 3, 0, 1), "face", 9, False), ("R3", (2, 3, 1, 1), "face", 9, True),
]


def _assert_identical(actual, expected):
    assert chunk_mismatches(actual, expected) == []
    for field in dataclasses.fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if not isinstance(b, str):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), field.name


def _roundtrip(chunk, factors):
    stats = {}
    data = encode_chunk("faces", chunk, factors=factors, stats=stats)
    return data, stats, decode_chunk("faces", data)


@pytest.fixture(scope="module")
def identity_case():
    return _sources(_context(), IDENTITY_PLAN)


@pytest.fixture(scope="module")
def ring_case():
    return _sources(_context(RING_MERGE), RING_PLAN)


@pytest.fixture(scope="module")
def plane_case():
    return _sources(_context(PLANE_MERGE), RING_PLAN)


def test_capture_is_opt_in_and_leaves_rows_bitwise_unchanged(identity_case, ring_case):
    for context, plan in ((_context(), IDENTITY_PLAN), (_context(RING_MERGE), RING_PLAN)):
        S = StructuredReconstruction(context)
        rng = np.random.default_rng(0)
        for request, key, location, q, nodes in plan:
            points = _jitter(context, key if location == "cell" else key[1:], q, rng, nodes)
            if request == "R3":
                plain = S.side_rows(key, points)
                with_factors = S.side_rows_with_factors(key, points)
                assert len(plain) == len(with_factors) == 2
                pairs = [(a, b[0], b[1]) for a, b in zip(plain, with_factors)]
            else:
                row, fac = S.rows_with_factors(key, points, location)
                pairs = [(S.rows(key, points, location), row, fac)]
            for expected, actual, fac in pairs:
                if expected is None:
                    assert actual is None and fac is None
                    continue
                for name in ("donor_ids", "value", "gradient", "trace_donor_points", "trace_target_points"):
                    a, b = getattr(actual, name), getattr(expected, name)
                    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (key, name)
                assert actual.diagnostics == expected.diagnostics
                factored = (expected.diagnostics["family"] in tr_mod.FAMILY_CODES and not expected.boundary_conditioned)
                assert (fac is not None) == factored, (key, expected.diagnostics["family"])


@pytest.mark.parametrize("case", ("identity_case", "ring_case", "plane_case"))
def test_tensor_encoding_decodes_bitwise_to_the_csr_chunk(case, request):
    S, rows, factors, chunk = request.getfixturevalue(case)
    data, stats, decoded = _roundtrip(chunk, factors)
    csr_data = encode_chunk("faces", chunk)
    _assert_identical(decode_chunk("faces", csr_data), chunk)
    _assert_identical(decoded, chunk)
    assert stats["fallback_sources"] == 0 and stats["tensor_sources"] == stats["candidate_sources"] > 0
    assert stats["tensor_sources"] == sum(f is not None for f in factors)
    assert len(data) < len(csr_data)
    for s, (actual, expected) in enumerate(zip(expand_point_rows(decoded), rows, strict=True)):
        stored_gradient = bool(chunk.has_gradient[chunk.source_ptr[s]])       # R3 sides keep values only
        for name in ("donor_ids", "value", "gradient", "trace_donor_points", "trace_target_points"):
            a, b = getattr(actual, name), getattr(expected, name)
            if (name != "trace_donor_points" or expected.boundary_conditioned) and (name != "gradient" or stored_gradient):
                assert a.shape == b.shape and a.tobytes() == b.tobytes(), name


def test_every_factored_family_is_covered(identity_case, ring_case):
    seen = set()
    for S, rows, factors, chunk in (identity_case, ring_case):
        _, stats, decoded = _roundtrip(chunk, factors)
        seen |= {name for name, count in stats["tensor_by_family"].items() if count}
        _assert_identical(decoded, chunk)
    assert seen == {"singleton", "ringwise", "centered_radial"}


def test_signed_zeros_are_reproduced_not_assigned(identity_case):
    """On-node targets give exact zeros. The plain products carry ``-0.0`` where a factor is negative, but the
    construction (add into zeros) stores ``+0.0``; the expansion must too (``np.array_equal`` would not notice)."""
    S, rows, factors, chunk = identity_case
    naive_negative = stored_negative = 0
    for fac, row in zip(factors, rows):
        if fac is not None and fac.family == "singleton":
            naive = (fac.radial[:, 0, :, None, None] * fac.eta_value[:, None, :, None]) * fac.theta_value[:, None, None, :]
            naive_negative += int(np.signbit(naive[naive == 0]).sum())
            stored = np.concatenate([row.value.ravel(), row.gradient.ravel()])
            stored_negative += int(np.signbit(stored[stored == 0]).sum())
    assert naive_negative > 0 and stored_negative == 0
    _, stats, decoded = _roundtrip(chunk, factors)
    assert stats["fallback_sources"] == 0
    _assert_identical(decoded, chunk)


def test_ringwise_aggregate_owner_spanning_two_planes_accumulates(ring_case):
    S, rows, factors, chunk = ring_case
    ringwise = [f for f in factors if f is not None and f.family == "ringwise"]
    assert ringwise
    repeated = [len(o) - len(np.unique(o)) for f in ringwise for o in f.ring_owner.reshape(-1, 112)]
    assert max(repeated) >= 1                               # a repeated owner within one target: `+=` accumulation
    _, stats, decoded = _roundtrip(chunk, factors)
    assert stats["tensor_by_family"]["ringwise"] == len(ringwise) and stats["fallback_sources"] == 0
    _assert_identical(decoded, chunk)


def test_singleton_aggregate_owner_spanning_two_planes_accumulates(plane_case):
    S, rows, factors, chunk = plane_case
    singles = [f for f in factors if f is not None and f.family == "singleton"]
    assert any(len(o) != len(np.unique(o)) for f in singles for o in f.owner.reshape(-1, 112))
    _, stats, decoded = _roundtrip(chunk, factors)
    assert stats["fallback_sources"] == 0
    _assert_identical(decoded, chunk)


def test_encoder_falls_back_to_csr_for_a_source_that_does_not_reproduce(identity_case, ring_case):
    for S, rows, factors, chunk in (identity_case, ring_case):
        good = [i for i, f in enumerate(factors) if f is not None]
        tampered = list(factors)
        # one ulp on a radial weight; a wrong owner id; a wrong table value; a wrong family
        bad = {good[0]: "radial", good[1]: "owner", good[2]: "theta", good[3]: "family"}
        for index, kind in bad.items():
            fac = dataclasses.replace(factors[index])
            if kind == "radial":
                fac.radial = fac.radial.copy()
                fac.radial[0, 0, 1] = np.nextafter(fac.radial[0, 0, 1], 10)
            elif kind == "owner":
                if fac.family == "ringwise":
                    fac.ring_owner = fac.ring_owner.copy()
                    fac.ring_owner[0, 0, 0, 0] += 1
                else:
                    fac.owner = fac.owner.copy()
                    fac.owner[0, 0, 0, 0] += 1
            elif kind == "theta":
                if fac.family == "ringwise":
                    fac.ring_value = fac.ring_value.copy()
                    fac.ring_value[0, 0, 0, 0] = np.nextafter(fac.ring_value[0, 0, 0, 0], 10)
                else:
                    fac.theta_value = fac.theta_value.copy()
                    fac.theta_value[0, 0] = np.nextafter(fac.theta_value[0, 0], 10)
            else:
                fac.family = "singleton" if fac.family != "singleton" else "ringwise"
            tampered[index] = fac
        data, stats, decoded = _roundtrip(chunk, tampered)
        assert stats["candidate_sources"] == len(good) and stats["fallback_sources"] >= 4
        assert stats["tensor_sources"] + stats["fallback_sources"] == len(good)
        assert sum(stats["fallback_by_family"].values()) == stats["fallback_sources"]
        with np.load(io.BytesIO(data)) as source:
            encoding = source["src_encoding"]
        assert all(encoding[i] == 0 for i in bad) and encoding.sum() == stats["tensor_sources"] > 0
        _assert_identical(decoded, chunk)             # bad sources are stored CSR, so nothing is lost


def test_all_sources_failing_gives_a_plain_v3_chunk(identity_case):
    S, rows, factors, chunk = identity_case
    tampered = []
    for f in factors:
        if f is not None:
            f = dataclasses.replace(f)
            f.radial = f.radial + 1e-9
        tampered.append(f)
    data, stats, decoded = _roundtrip(chunk, tampered)
    assert stats["tensor_sources"] == 0 and stats["fallback_sources"] == stats["candidate_sources"] > 0
    assert data == encode_chunk("faces", chunk)                    # no src_encoding member at all
    _assert_identical(decoded, chunk)


def test_old_v3_and_v2_files_still_decode_as_csr(identity_case):
    S, rows, factors, chunk = identity_case
    arrays = art._point_chunk_to_arrays(chunk)
    assert "src_encoding" not in arrays
    buffer = io.BytesIO(); np.savez(buffer, **arrays)
    _assert_identical(decode_chunk("faces", buffer.getvalue()), chunk)
    buffer = io.BytesIO(); np.savez(buffer, **art._point_chunk_to_arrays_v2(chunk))     # the v2 layout
    _assert_identical(decode_chunk("faces", buffer.getvalue()), chunk)
    # an explicit all-CSR src_encoding is the same thing
    buffer = io.BytesIO(); np.savez(buffer, src_encoding=np.zeros(len(chunk.source_ptr) - 1, np.int8), **arrays)
    _assert_identical(decode_chunk("faces", buffer.getvalue()), chunk)
    with pytest.raises(ValueError, match="one entry"):
        encode_chunk("faces", chunk, factors=factors[:-1])


def test_corrupt_tensor_members_are_rejected(identity_case):
    S, rows, factors, chunk = identity_case
    with np.load(io.BytesIO(encode_chunk("faces", chunk, factors=factors))) as source:
        arrays = {k: source[k] for k in source.files}
    broken = dict(arrays)
    broken["tr_t_eta"] = broken["tr_t_eta"][:-1]
    with pytest.raises(ValueError, match="corrupted"):
        art._arrays_to_point_chunk(broken)
    broken = dict(arrays)
    broken["src_encoding"] = np.where(arrays["src_conditioned"], 1, arrays["src_encoding"]).astype(np.int8)
    with pytest.raises(ValueError, match="src_encoding"):
        art._arrays_to_point_chunk(broken)


def test_tables_are_deduplicated_by_bit_pattern_and_members_are_compact(ring_case):
    S, rows, factors, chunk = ring_case
    keep = [i for i, f in enumerate(factors) if f is not None]
    tensor = tr_mod.build_tensor_rows([factors[i] for i in keep], [chunk.has_gradient[chunk.source_ptr[i]] for i in keep])
    for table in ((tensor.eta_index, tensor.eta_value, tensor.eta_derivative), (tensor.radial,),
                  (tensor.ring_owner, tensor.ring_value, tensor.ring_derivative)):
        bits = np.concatenate([np.ascontiguousarray(a.astype(np.float64).reshape(len(a), -1)).view(np.uint64)
                               for a in table], axis=1)
        assert len(np.unique(bits, axis=0)) == len(bits)
    assert len(tensor.ring_owner) < tensor.t_ring.size                     # rings are shared between targets
    members = tensor.to_arrays()
    assert members["tr_theta_index"].dtype == np.int16 and members["tr_t_eta"].dtype == np.uint16
    assert members["tr_ring_owner"].dtype == np.int32 and members["tr_family"].dtype == np.int8
    back = tr_mod.TensorRows.from_arrays(members, has_gradient=tensor.has_gradient, target_counts=np.diff(tensor.target_ptr))
    for field in dataclasses.fields(tensor):
        a, b = getattr(back, field.name), getattr(tensor, field.name)
        assert (a == b) if isinstance(b, int) else (a.dtype == b.dtype and a.tobytes() == b.tobytes()), field.name


@pytest.mark.parametrize("case", ("identity_case", "ring_case", "plane_case"))
def test_numpy_apply_matches_the_dense_rows(case, request):
    S, rows, factors, chunk = request.getfixturevalue(case)
    keep = [i for i, f in enumerate(factors) if f is not None]
    has_gradient = [bool(chunk.has_gradient[chunk.source_ptr[i]]) for i in keep]
    tensor = tr_mod.build_tensor_rows([factors[i] for i in keep], has_gradient)
    fields = np.random.default_rng(3).normal(size=(int(max(r.donor_ids.max() for r in rows if len(r.donor_ids))) + 1, 3))
    values, gradients = tr_mod.apply_tensor_rows_numpy(tensor, fields, block=5)
    v_ref, g_ref = [], []
    for i in keep:
        row = rows[i]
        v_ref.append(row.value @ fields[row.donor_ids])
        if chunk.has_gradient[chunk.source_ptr[i]]:
            g_ref.append(np.einsum('qad,df->qaf', row.gradient, fields[row.donor_ids]))
    v_ref, g_ref = np.concatenate(v_ref), np.concatenate(g_ref)
    assert values.shape == v_ref.shape and gradients.shape == g_ref.shape
    np.testing.assert_allclose(values, v_ref, rtol=1e-12, atol=1e-13 * np.abs(v_ref).max())
    np.testing.assert_allclose(gradients, g_ref, rtol=1e-12, atol=1e-13 * np.abs(g_ref).max())
    only_v, no_g = tr_mod.apply_tensor_rows_numpy(tensor, fields, gradients=False)
    assert no_g is None and only_v.tobytes() == values.tobytes()


def test_expansion_layout_and_source_rows(identity_case):
    S, rows, factors, chunk = identity_case
    keep = [i for i, f in enumerate(factors) if f is not None]
    has_gradient = [bool(chunk.has_gradient[chunk.source_ptr[i]]) for i in keep]
    tensor = tr_mod.build_tensor_rows([factors[i] for i in keep], has_gradient)
    expansion = tr_mod.expand_tensor_rows(tensor)
    for k, i in enumerate(keep):
        donor, value, gradient = expansion.source_rows(k, len(rows[i].trace_target_points), has_gradient[k])
        assert donor.tobytes() == rows[i].donor_ids.tobytes() and value.tobytes() == rows[i].value.tobytes()
        if has_gradient[k]:
            assert gradient.tobytes() == np.ascontiguousarray(rows[i].gradient).tobytes()
        else:
            assert gradient is None


def test_build_artifact_packs_tensor_sources_and_aggregates_the_counts(identity_case, tmp_path, monkeypatch):
    from drbx.stencils.builder import PointRowRequest
    from p_shared import build_artifact as ba, runner

    S, rows, factors, chunk = identity_case
    first = chunk.source_ptr[:-1]
    requests = [PointRowRequest(str(chunk.request[t]), int(chunk.entity_id[t]), str(chunk.bc_variant[t]),
                                int(chunk.radial_degree[t]), row, fac)
                for t, row, fac in zip(first, rows, factors, strict=True)]
    _assert_identical(ba._pack_point_rows(requests), chunk)
    arrays, stats = ba._point_chunk_arrays(chunk, requests, tensor=True)
    assert "src_encoding" in arrays and stats["tensor_sources"] == sum(f is not None for f in factors) > 0
    assert stats["fallback_sources"] == 0
    plain, no_stats = ba._point_chunk_arrays(chunk, requests, tensor=False)
    assert "src_encoding" not in plain and no_stats == {}
    _assert_identical(art._arrays_to_point_chunk(arrays), chunk)
    monkeypatch.setenv(ba.CSR_ONLY_ENV, "1")
    assert not ba.tensor_encoding_enabled()
    monkeypatch.setenv(ba.CSR_ONLY_ENV, "0")
    assert ba.tensor_encoding_enabled()

    output, plan = tmp_path / "out", {"cells": runner.chunk_units("cells", N, 8, 4)}
    for unit in plan["cells"]:
        runner.write_unit(output, unit, {"i": 1}, started=time.time(), chunks={"chunk": {"x": np.zeros(1)}},
                          extra={"tensor_encoding": stats})
    total = ba._aggregate_receipts(output, plan)["tensor_encoding"]
    assert total["tensor_sources"] == 2 * stats["tensor_sources"] and total["fallback_sources"] == 0
    assert total["tensor_by_family"]["singleton"] == 2 * stats["tensor_by_family"]["singleton"]


def test_decode_chunk_factored_exposes_the_tensor_sources_without_expanding_them(ring_case, identity_case):
    for S, rows, factors, chunk in (ring_case, identity_case):
        data, stats, full = _roundtrip(chunk, factors)
        stored, tensor, is_tensor = art.decode_chunk_factored("faces", data)
        assert is_tensor.sum() == stats["tensor_sources"] > 0 and tensor.n_sources == is_tensor.sum()
        first = chunk.source_ptr[:-1]
        assert np.array_equal(is_tensor, np.array([f is not None for f in factors]))
        # tags, targets and pointers are the full chunk's; tensor sources hold no donors, value or gradient
        for name in ("request", "entity_id", "quad_node", "family", "conditioned", "bc_variant", "radial_degree",
                     "target_point", "has_gradient", "source_ptr", "source_diagnostics_json", "query_table"):
            a, b = getattr(stored, name), getattr(full, name)
            assert a == b if isinstance(b, str) else a.tobytes() == b.tobytes(), name
        widths = np.diff(stored.donor_ptr)
        assert np.all(widths[np.repeat(is_tensor, np.diff(chunk.source_ptr))] == 0)
        csr_targets = ~np.repeat(is_tensor, np.diff(chunk.source_ptr))
        assert stored.value.tobytes() == full.value[np.repeat(csr_targets, np.diff(full.donor_ptr))].tobytes()
        # expanding the factors reproduces the omitted rows
        expansion = tr_mod.expand_tensor_rows(tensor)
        for k, s in enumerate(np.flatnonzero(is_tensor)):
            donor, value, gradient = expansion.source_rows(k, int(np.diff(chunk.source_ptr)[s]), bool(chunk.has_gradient[first[s]]))
            assert donor.tobytes() == rows[s].donor_ids.tobytes() and value.tobytes() == rows[s].value.tobytes()
    plain = art.decode_chunk_factored("faces", encode_chunk("faces", identity_case[3]))
    assert plain[1] is None and not plain[2].any()
    _assert_identical(plain[0], identity_case[3])
    buffer = io.BytesIO(); np.savez(buffer, **art._point_chunk_to_arrays_v2(identity_case[3]))
    v2 = art.decode_chunk_factored("faces", buffer.getvalue())
    assert v2[1] is None and not v2[2].any()
    with pytest.raises(ValueError, match="point chunks"):
        art.decode_chunk_factored("p07", b"")
