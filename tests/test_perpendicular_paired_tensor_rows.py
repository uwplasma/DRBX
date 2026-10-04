"""Paired tensor sources: a ``cell_stencil="symmetric"`` cell row ``1/2 (A + B)`` stored as the factors of its two parts.

The merged row is not one tensor factorization, so the row artifact stores it as two consecutive ``TensorRows``
sources (A then B, sharing the deduplicated tables) under ``src_encoding`` code 2 and merges their expansions at decode
exactly as ``StructuredReconstruction._average_rows`` does. Fast and synthetic: the n = 12 toy grid of
``tests/test_perpendicular_symmetric_cell_rows.py`` with its agglomerated rings 0..4 holding 1, 4, 8, 8, 8 owners per eta
plane, which gives every kind of R1 cell row: coupled (``i = 1, 2``), a ringwise row without a mirror (``i = 3``),
ringwise pairs (``i = 4, 5``), a singleton A with the ringwise fallback B (``i = 6``, stays CSR), singleton pairs
(``i = 7..9``) and boundary rows (``i = 10, 11``).
"""
from __future__ import annotations

import dataclasses
import io

import numpy as np
import pytest

from drbx.geometry.fci_perpendicular_reconstruction import (
    PairedFactors, PointFactors, PointRows, StructuredReconstruction)
from drbx.stencils import artifact as art
from drbx.stencils import tensor_rows as tr_mod
from drbx.stencils.artifact import (
    ENCODING_PAIRED, ENCODING_TENSOR, chunk_mismatches, decode_chunk, encode_chunk, expand_point_rows,
    load_row_artifact, pack_point_rows, save_row_artifact)
from tests.test_perpendicular_symmetric_cell_rows import _cell, _context

N = 12
RINGS = {0: 1, 1: 4, 2: 8, 3: 8, 4: 8}
#: per ring ``i``: what the symmetric row's capture and storage are (coupled and boundary rows have no factors)
KIND = {1: "csr", 2: "csr", 3: "single_ringwise", 4: "pair_ringwise", 5: "pair_ringwise", 6: "mixed", 7: "pair_singleton",
        8: "pair_singleton", 9: "pair_singleton", 10: "csr", 11: "csr"}
#: two cells per ring; (j, k) = (0, 0) and (11, 11) wrap the theta planes and reflect the eta planes through the seam
PLACES = ((2, 4), (7, 9), (0, 0), (11, 11))
KEYS = tuple((i, j, k) for i in sorted(KIND) for (j, k) in PLACES)


def _pack(rows, keys, **kwargs):
    return pack_point_rows(
        rows, request="R1", entity_id=[int(np.ravel_multi_index(k, (N,) * 3)) for k in keys],
        bc_variant=["D" if r.boundary_conditioned else "" for r in rows],
        radial_degree=[3 if r.boundary_conditioned else 0 for r in rows], **kwargs)


def _case(cell_stencil, inner_support="profile7", **kwargs):
    context = _context(N, RINGS)
    S = StructuredReconstruction(context, inner_support=inner_support, cell_stencil=cell_stencil)
    rows, factors = zip(*(S.rows_with_factors(key, _cell(context, key), "cell") for key in KEYS))
    return dict(context=context, S=S, rows=list(rows), factors=list(factors), chunk=_pack(rows, KEYS, **kwargs))


@pytest.fixture(scope="module")
def symmetric():
    return _case("symmetric")


@pytest.fixture(scope="module")
def biased():
    return _case("biased")


def _assert_identical(actual, expected):
    assert chunk_mismatches(actual, expected) == []
    for field in dataclasses.fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if not isinstance(b, str):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), field.name


def _arrays(data):
    with np.load(io.BytesIO(data)) as source:
        return {name: source[name] for name in source.files}


def _kind_of(key):
    return KIND[key[0]]


def _expected_encoding():
    return np.array([{"pair": ENCODING_PAIRED, "single": ENCODING_TENSOR}.get(_kind_of(k).split("_")[0], 0) for k in KEYS])


# ---------------------------------------------------------------------------
# the capture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("inner_support", ("profile7", "fixed_radius"))
def test_symmetric_cell_rows_capture_paired_factors_only_for_same_family_pairs(inner_support):
    case = _case("symmetric", inner_support)
    plain = StructuredReconstruction(case["context"], inner_support=inner_support, cell_stencil="symmetric")
    for key, row, factors in zip(KEYS, case["rows"], case["factors"]):
        kind = _kind_of(key)
        # capture is opt-in and the row is bitwise the same with or without it
        expected = plain.rows(key, _cell(case["context"], key), "cell")
        for name in ("donor_ids", "value", "gradient", "trace_donor_points", "trace_target_points"):
            a, b = getattr(row, name), getattr(expected, name)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (key, name)
        assert row.diagnostics == expected.diagnostics
        if kind.startswith("pair"):
            family = kind.split("_")[1]
            assert isinstance(factors, PairedFactors) and row.diagnostics["cell_stencil"] == "symmetric"
            assert factors.a.family == factors.b.family == family == row.diagnostics["family"]
            assert [int(v) for v in factors.a.layers] == list(range(key[0] - 1, key[0] + 3))     # A: layers i-1..i+2
            assert [int(v) for v in factors.b.layers] == list(range(key[0] - 2, key[0] + 2))     # B: its mirror
        elif kind == "mixed":
            assert factors is None                       # a singleton A with the ringwise fallback B: nothing captured
            assert row.diagnostics["family"] == "singleton" and row.diagnostics["mirror_family"] == "ringwise"
        elif kind == "single_ringwise":
            assert isinstance(factors, PointFactors) and factors.family == "ringwise"     # no mirror: the biased row
        else:
            assert factors is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        case["factors"][KEYS.index((7, 2, 4))].a = None


def test_mirror_factors_reflect_the_eta_planes_about_the_cell(symmetric):
    for key, factors in zip(KEYS, symmetric["factors"]):
        if isinstance(factors, PairedFactors):
            k = key[2]
            assert np.array_equal(factors.b.eta_index, (2 * k - factors.a.eta_index) % N)


# ---------------------------------------------------------------------------
# the encoding
# ---------------------------------------------------------------------------
def test_pure_pairs_encode_as_code_2_and_mixed_pairs_stay_csr(symmetric):
    chunk, factors = symmetric["chunk"], symmetric["factors"]
    stats = {}
    data = encode_chunk("cells", chunk, factors=factors, stats=stats)
    arrays = _arrays(data)
    encoding = arrays["src_encoding"]
    assert encoding.dtype == np.int8
    assert np.array_equal(encoding, _expected_encoding())
    assert (encoding == ENCODING_PAIRED).any() and (encoding == ENCODING_TENSOR).any() and (encoding == 0).any()
    # mixed pairs (ring 6) and the coupled / boundary rows are CSR: they keep their donors
    src_width = np.diff(arrays["src_donor_ptr"])
    assert np.all(src_width[encoding > 0] == 0) and np.all(src_width[encoding == 0] > 0)
    mixed = [s for s, key in enumerate(KEYS) if _kind_of(key) == "mixed"]
    assert mixed and np.all(encoding[mixed] == 0)
    pairs, singles = int((encoding == ENCODING_PAIRED).sum()), int((encoding == ENCODING_TENSOR).sum())
    # ``tensor_sources`` are the sources stored as factors, a pair included; ``paired_*`` count the pairs among them
    assert stats["sources"] == len(KEYS) and stats["candidate_sources"] == pairs + singles
    assert stats["tensor_sources"] == pairs + singles and stats["paired_sources"] == pairs and stats["fallback_sources"] == 0
    assert stats["tensor_targets"] == pairs + singles and stats["paired_targets"] == pairs       # one target per cell
    per_family = {name: sum(1 for k in KEYS if _kind_of(k) == f"pair_{name}") for name in ("singleton", "ringwise")}
    assert stats["paired_by_family"] == per_family and stats["fallback_by_family"] == {"ringwise": 0, "singleton": 0}
    assert stats["tensor_by_family"] == {"ringwise": per_family["ringwise"] + singles, "singleton": per_family["singleton"]}
    # two TensorRows sources per pair, one per single, A then B in chunk order
    assert len(arrays["tr_family"]) == 2 * pairs + singles
    csr = encode_chunk("cells", chunk)
    assert len(data) < 0.8 * len(csr)


def test_paired_encoding_decodes_bitwise_to_the_input_chunk(symmetric):
    chunk, factors = symmetric["chunk"], symmetric["factors"]
    data = encode_chunk("cells", chunk, factors=factors)
    _assert_identical(decode_chunk("cells", data), chunk)
    _assert_identical(art._arrays_to_point_chunk(_arrays(data)), chunk)
    for decoded, expected in zip(expand_point_rows(decode_chunk("cells", data)), symmetric["rows"], strict=True):
        for name in ("donor_ids", "value", "gradient", "trace_target_points"):
            a, b = getattr(decoded, name), getattr(expected, name)
            assert a.shape == b.shape and a.tobytes() == b.tobytes(), name


def _check_factored_view(chunk, rows, data):
    """``decode_chunk_factored``: tags and pointers as in the full decode; the rows of every source but the single tensor
    sources (pairs come back merged, as CSR) are the full chunk's; expanding the singles' factors gives their rows."""
    full = decode_chunk("cells", data)
    stored, tensor, is_tensor = art.decode_chunk_factored("cells", data)
    encoding = _arrays(data)["src_encoding"]
    assert np.array_equal(is_tensor, encoding == ENCODING_TENSOR)
    for name in ("request", "entity_id", "quad_node", "family", "conditioned", "bc_variant", "radial_degree",
                 "target_point", "has_gradient", "source_ptr", "source_diagnostics_json", "query_table"):
        a, b = getattr(stored, name), getattr(full, name)
        assert a == b if isinstance(b, str) else a.tobytes() == b.tobytes(), name
    counts = np.diff(chunk.source_ptr)                      # one target per source: target == source
    assert np.array_equal(np.diff(stored.donor_ptr) == 0, is_tensor)
    csr_entries = np.repeat(np.repeat(~is_tensor, counts), np.diff(full.donor_ptr))
    assert stored.donor.tobytes() == full.donor[csr_entries].tobytes()
    assert stored.value.tobytes() == full.value[csr_entries].tobytes()
    assert stored.gradient.tobytes() == full.gradient[:, csr_entries].tobytes()
    if not is_tensor.any():
        assert tensor is None
        return stored, tensor, is_tensor
    assert tensor.n_sources == is_tensor.sum()               # no TensorRows source belongs to a pair
    expansion = tr_mod.expand_tensor_rows(tensor)
    for k, s in enumerate(np.flatnonzero(is_tensor)):
        donor, value, gradient = expansion.source_rows(k, int(counts[s]), bool(chunk.has_gradient[chunk.source_ptr[s]]))
        assert donor.tobytes() == rows[s].donor_ids.tobytes() and value.tobytes() == rows[s].value.tobytes()
        assert np.ascontiguousarray(gradient).tobytes() == np.ascontiguousarray(rows[s].gradient).tobytes()
    return stored, tensor, is_tensor


def test_decode_chunk_factored_returns_pairs_as_csr_and_singles_as_factors(symmetric):
    chunk, factors, rows = symmetric["chunk"], symmetric["factors"], symmetric["rows"]
    data = encode_chunk("cells", chunk, factors=factors)
    stored, tensor, is_tensor = _check_factored_view(chunk, rows, data)
    encoding = _arrays(data)["src_encoding"]
    assert is_tensor.any() and (encoding == ENCODING_PAIRED).any()
    assert tensor.n_sources == int((encoding == ENCODING_TENSOR).sum()) and np.all(tensor.family == 1)    # ringwise single


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_pairs_singles_and_csr_sources_interleave_in_any_chunk_order(symmetric, seed):
    order = np.random.default_rng(seed).permutation(len(KEYS))
    rows = [symmetric["rows"][s] for s in order]
    chunk = _pack(rows, [KEYS[s] for s in order])
    stats = {}
    data = encode_chunk("cells", chunk, factors=[symmetric["factors"][s] for s in order], stats=stats)
    assert np.array_equal(_arrays(data)["src_encoding"], _expected_encoding()[order]) and stats["fallback_sources"] == 0
    _assert_identical(decode_chunk("cells", data), chunk)
    _check_factored_view(chunk, rows, data)


def test_a_chunk_of_pairs_only_has_no_tensor_rows_in_the_factored_view(symmetric):
    keep = [s for s, key in enumerate(KEYS) if _kind_of(key).startswith("pair")]
    rows = [symmetric["rows"][s] for s in keep]
    chunk = _pack(rows, [KEYS[s] for s in keep])
    stats = {}
    data = encode_chunk("cells", chunk, factors=[symmetric["factors"][s] for s in keep], stats=stats)
    assert stats["paired_sources"] == stats["tensor_sources"] == len(keep) and stats["fallback_sources"] == 0
    stored, tensor, is_tensor = art.decode_chunk_factored("cells", data)
    assert tensor is None and not is_tensor.any()
    _assert_identical(stored, chunk)                         # every source is CSR in this view
    _assert_identical(decode_chunk("cells", data), chunk)


def test_pairs_with_and_without_stored_gradient_roundtrip(symmetric):
    store = [bool(s % 3) for s in range(len(KEYS))]
    chunk = _pack(symmetric["rows"], KEYS, store_gradient=store)
    stats = {}
    data = encode_chunk("cells", chunk, factors=symmetric["factors"], stats=stats)
    assert stats["fallback_sources"] == 0 and stats["paired_sources"] > 0
    has_gradient = np.array([store[s] for s in range(len(KEYS))])
    pair = np.flatnonzero(_arrays(data)["src_encoding"] == ENCODING_PAIRED)
    assert has_gradient[pair].any() and not has_gradient[pair].all()
    _assert_identical(decode_chunk("cells", data), chunk)
    stored, tensor, is_tensor = art.decode_chunk_factored("cells", data)
    assert np.array_equal(is_tensor, _arrays(data)["src_encoding"] == ENCODING_TENSOR)
    assert np.array_equal(stored.has_gradient, chunk.has_gradient)


def test_several_targets_per_source_and_nodes_exactly_on_cell_centres():
    """``q = 3`` jittered targets (one of them the cell centre itself: one-hot factors, signed zeros) in one pair."""
    context = _context(N, RINGS)
    S = StructuredReconstruction(context, cell_stencil="symmetric")
    rng = np.random.default_rng(5)
    rows, factors, keys = [], [], []
    for key in ((4, 3, 5), (8, 6, 11), (9, 0, 0)):
        centre = _cell(context, key)[0]
        points = centre + rng.uniform(-.2, .2, size=(3, 3)) * np.array([1 / N, 2 * np.pi / N, 2 * np.pi / N])
        points[0] = centre
        row, fac = S.rows_with_factors(key, points, "cell")
        assert isinstance(fac, PairedFactors) and len(row.trace_target_points) == 3
        rows.append(row), factors.append(fac), keys.append(key)
    chunk = _pack(rows, keys)
    stats = {}
    data = encode_chunk("cells", chunk, factors=factors, stats=stats)
    assert stats["paired_sources"] == 3 and stats["paired_targets"] == 9 and stats["fallback_sources"] == 0
    assert stats["tensor_sources"] == 3 and stats["tensor_targets"] == 9
    _assert_identical(decode_chunk("cells", data), chunk)


def test_save_and_load_row_artifact_roundtrips_a_paired_chunk(symmetric, tmp_path):
    chunk, factors = symmetric["chunk"], symmetric["factors"]
    stats = {}
    grid = save_row_artifact(tmp_path, N, identity={"test": 1}, cells=(chunk,), point_factors={"cells": [factors]}, stats=stats)
    assert stats["paired_sources"] > 0
    loaded = load_row_artifact(tmp_path, N, {"test": 1})
    _assert_identical(loaded.cells[0], chunk)
    assert (grid / "rows" / "cells_0.npz").stat().st_size < 0.8 * len(encode_chunk("cells", chunk))


# ---------------------------------------------------------------------------
# biased artifacts are unchanged
# ---------------------------------------------------------------------------
def test_biased_rows_use_only_codes_0_and_1_and_the_same_arrays_as_before(biased):
    chunk, factors = biased["chunk"], biased["factors"]
    assert not any(isinstance(f, PairedFactors) for f in factors)
    stats = {}
    data = encode_chunk("cells", chunk, factors=factors, stats=stats)
    arrays, csr = _arrays(data), _arrays(encode_chunk("cells", chunk))
    encoding = arrays["src_encoding"]
    assert encoding.dtype == np.int8 and set(encoding.tolist()) == {0, 1}
    assert np.array_equal(encoding == 1, np.array([isinstance(f, PointFactors) for f in factors]))
    assert stats["paired_sources"] == 0 and stats["paired_targets"] == 0 and set(stats["paired_by_family"].values()) == {0}
    assert stats["tensor_sources"] == int(encoding.sum()) and stats["fallback_sources"] == 0
    # the members: those of the all-CSR encoding, ``src_encoding`` and the ``tr_*`` factors; nothing else
    tensor = tr_mod.build_tensor_rows([f for f in factors if f is not None],
                                      [True] * sum(f is not None for f in factors))
    assert set(arrays) == set(csr) | {"src_encoding"} | set(tensor.to_arrays())
    for name in set(csr) - {"src_donor_ptr", "src_donor", "value", "gradient"}:
        assert arrays[name].dtype == csr[name].dtype and arrays[name].tobytes() == csr[name].tobytes(), name
    for name, array in tensor.to_arrays().items():
        assert arrays[name].dtype == array.dtype and arrays[name].tobytes() == array.tobytes(), name
    # the CSR sources keep their rows; the tensor sources hold none
    csr_source = encoding == 0
    entry_csr = np.repeat(np.repeat(csr_source, np.diff(chunk.source_ptr)), np.diff(chunk.donor_ptr))     # per value entry
    assert arrays["src_donor"].tobytes() == csr["src_donor"][np.repeat(csr_source, np.diff(csr["src_donor_ptr"]))].tobytes()
    assert arrays["value"].tobytes() == csr["value"][entry_csr].tobytes()
    assert arrays["gradient"].tobytes() == csr["gradient"][:, entry_csr].tobytes()
    _assert_identical(decode_chunk("cells", data), chunk)
    factored = art.decode_chunk_factored("cells", data)
    assert factored[1].n_sources == int(encoding.sum()) and np.array_equal(factored[2], encoding == 1)


# ---------------------------------------------------------------------------
# verification: a pair is accepted only if the merge reproduces the stored row bit for bit
# ---------------------------------------------------------------------------
def _scaled(factors, name):
    """A copy of the ``PointFactors`` with its largest ``name`` entry (certainly one the row depends on) scaled by 1 + 1e-12."""
    inner = dataclasses.replace(factors)
    array = np.array(getattr(inner, name), dtype=np.float64)
    at = np.unravel_index(np.argmax(np.abs(array)), array.shape)
    array[at] *= 1 + 1e-12
    setattr(inner, name, array)
    return inner


def _bumped(factors, part, name):
    """``factors`` (a ``PairedFactors``) with one factor of its part ``part`` ("a" / "b") perturbed."""
    return dataclasses.replace(factors, **{part: _scaled(getattr(factors, part), name)})


@pytest.mark.parametrize("part", ("a", "b"))
@pytest.mark.parametrize("name", ("radial", "eta_value", "family-specific"))
def test_a_tampered_part_rejects_the_pair_to_csr(symmetric, part, name):
    chunk, factors = symmetric["chunk"], list(symmetric["factors"])
    pairs = [s for s, f in enumerate(factors) if isinstance(f, PairedFactors)]
    victims = {"singleton": next(s for s in pairs if factors[s].a.family == "singleton"),
               "ringwise": next(s for s in pairs if factors[s].a.family == "ringwise")}
    for family, victim in victims.items():
        field = name if name != "family-specific" else ("theta_value" if family == "singleton" else "ring_value")
        tampered = list(factors)
        tampered[victim] = _bumped(factors[victim], part, field)
        stats = {}
        data = encode_chunk("cells", chunk, factors=tampered, stats=stats)
        encoding = _arrays(data)["src_encoding"]
        assert encoding[victim] == 0 and np.array_equal(np.delete(encoding, victim), np.delete(_expected_encoding(), victim))
        assert stats["fallback_sources"] == 1 and stats["fallback_by_family"][family] == 1
        assert stats["paired_sources"] == len(pairs) - 1 and stats["tensor_sources"] == stats["candidate_sources"] - 1
        _assert_identical(decode_chunk("cells", data), chunk)           # the victim is stored CSR: nothing is lost


def test_a_raw_cell_with_two_owners_in_a_pair_drops_the_sources_touching_it_to_csr(symmetric):
    """The tables hold one owner per raw cell: a part of a pair that maps a raw cell to another owner than the other sources
    do makes every source touching that cell unstorable as factors (checked before any expansion), so they stay CSR."""
    chunk, factors = symmetric["chunk"], list(symmetric["factors"])
    pairs = [s for s, f in enumerate(factors) if isinstance(f, PairedFactors) and f.a.family == "singleton"]
    victim = pairs[len(pairs) // 2]
    part = dataclasses.replace(factors[victim].b)
    part.owner = np.array(part.owner)
    part.owner[0, 0, 0, 0] += 1
    tampered = list(factors)
    tampered[victim] = dataclasses.replace(factors[victim], b=part)
    stats = {}
    data = encode_chunk("cells", chunk, factors=tampered, stats=stats)
    encoding = _arrays(data)["src_encoding"]
    assert encoding[victim] == 0 and stats["fallback_sources"] > 1          # the victim and the neighbours sharing the raw cell
    assert stats["tensor_sources"] + stats["fallback_sources"] == stats["candidate_sources"]
    assert (encoding == ENCODING_PAIRED).any()                        # sources away from that raw cell are unaffected
    _assert_identical(decode_chunk("cells", data), chunk)


def test_a_pair_of_different_families_or_unrelated_factors_is_not_a_candidate(symmetric):
    chunk, factors = symmetric["chunk"], list(symmetric["factors"])
    pairs = [s for s, f in enumerate(factors) if isinstance(f, PairedFactors)]
    single_pair = next(s for s in pairs if factors[s].a.family == "singleton")
    ring_pair = next(s for s in pairs if factors[s].a.family == "ringwise")
    other_single = next(s for s in pairs if factors[s].a.family == "singleton" and s != single_pair)
    mixed = PairedFactors(factors[single_pair].a, factors[ring_pair].b)
    swapped = PairedFactors(factors[other_single].a, factors[single_pair].b)      # A of one cell with B of another
    tampered = list(factors)
    tampered[single_pair], tampered[other_single] = mixed, swapped
    stats = {}
    data = encode_chunk("cells", chunk, factors=tampered, stats=stats)
    encoding = _arrays(data)["src_encoding"]
    assert encoding[single_pair] == 0 and encoding[other_single] == 0
    assert stats["fallback_sources"] == 2 and stats["paired_sources"] == len(pairs) - 2
    _assert_identical(decode_chunk("cells", data), chunk)


def test_a_paired_factor_for_a_source_of_another_family_tag_is_not_a_candidate(symmetric):
    chunk, factors = symmetric["chunk"], list(symmetric["factors"])
    pairs = [s for s, f in enumerate(factors) if isinstance(f, PairedFactors)]
    wrong = next(s for s in pairs if factors[s].a.family == "ringwise")
    tampered = list(factors)
    tampered[wrong] = next(f for f in factors if isinstance(f, PairedFactors) and f.a.family == "singleton")
    stats = {}
    data = encode_chunk("cells", chunk, factors=tampered, stats=stats)
    assert _arrays(data)["src_encoding"][wrong] == 0 and stats["fallback_sources"] == 1
    _assert_identical(decode_chunk("cells", data), chunk)


def test_every_pair_failing_gives_a_chunk_with_only_single_tensor_sources(symmetric):
    chunk, factors = symmetric["chunk"], symmetric["factors"]
    tampered = [_bumped(f, "b", "radial") if isinstance(f, PairedFactors) else f for f in factors]
    stats = {}
    data = encode_chunk("cells", chunk, factors=tampered, stats=stats)
    encoding = _arrays(data)["src_encoding"]
    assert stats["paired_sources"] == 0 and set(encoding.tolist()) == {0, 1} and stats["fallback_sources"] > 0
    assert stats["tensor_sources"] == int(encoding.sum())                     # the single ringwise sources only
    _assert_identical(decode_chunk("cells", data), chunk)


def test_nothing_accepted_gives_a_plain_v3_chunk(symmetric):
    chunk, factors = symmetric["chunk"], symmetric["factors"]
    tampered = [None if f is None else _bumped(f, "a", "radial") if isinstance(f, PairedFactors) else _scaled(f, "radial")
                for f in factors]
    stats = {}
    data = encode_chunk("cells", chunk, factors=tampered, stats=stats)
    assert stats["paired_sources"] == stats["tensor_sources"] == 0 and stats["fallback_sources"] == stats["candidate_sources"] > 0
    assert data == encode_chunk("cells", chunk)              # no src_encoding member at all
    assert encode_chunk("cells", chunk, factors=[None] * len(KEYS)) == data


# ---------------------------------------------------------------------------
# the merge helper and the TensorRows subset
# ---------------------------------------------------------------------------
def _signed_zero_rows(rng, q, donors):
    value = rng.normal(size=(q, len(donors)))
    gradient = rng.normal(size=(q, 3, len(donors)))
    value[0, ::3] = -0.0                                    # a stored -0.0 must come out as +0.0
    gradient[:, 1, ::4] = -0.0
    return value, gradient


@pytest.mark.parametrize("with_gradient", (True, False))
def test_average_expanded_rows_is_average_rows_bit_for_bit(with_gradient):
    rng = np.random.default_rng(11)
    p = np.zeros((3, 3))
    for donors_a, donors_b in (([1, 3, 4, 9], [3, 4, 7, 9, 12]), ([2, 5], [6, 8, 10]), ([4, 5, 6], [4, 5, 6]), ([3], [3, 11])):
        da, db = np.array(donors_a), np.array(donors_b)
        (va, ga), (vb, gb) = _signed_zero_rows(rng, 3, da), _signed_zero_rows(rng, 3, db)
        a = PointRows(da, va, ga, False, np.empty((0, 3)), p.copy(), {"family": "singleton", "max_residual": 0.})
        b = PointRows(db, vb, gb, False, np.empty((0, 3)), p.copy(), {"family": "ringwise", "max_residual": 0., "min_rank": 7})
        merged = StructuredReconstruction._average_rows(a, b, p)
        donor, value, gradient = tr_mod.average_expanded_rows(da, va, ga if with_gradient else None,
                                                              db, vb, gb if with_gradient else None)
        assert donor.dtype == merged.donor_ids.dtype and donor.tobytes() == merged.donor_ids.tobytes()
        assert value.tobytes() == merged.value.tobytes() and value.shape == merged.value.shape
        if with_gradient:
            assert gradient.tobytes() == merged.gradient.tobytes() and gradient.shape == merged.gradient.shape
        else:
            assert gradient is None
        assert not np.signbit(value[value == 0]).any()
    with pytest.raises(ValueError, match="gradient"):
        tr_mod.average_expanded_rows(da, va, ga, db, vb, None)


def _tensor_of(symmetric, sources):
    factors = symmetric["factors"]
    parts = [part for s in sources for part in ((factors[s].a, factors[s].b) if isinstance(factors[s], PairedFactors) else (factors[s],))]
    multiplicity = np.array([2 if isinstance(factors[s], PairedFactors) else 1 for s in sources])
    return tr_mod.build_tensor_rows(parts, [True] * len(parts)), multiplicity


def test_merge_paired_expansion_collapses_pairs_in_place(symmetric):
    sources = [s for s, f in enumerate(symmetric["factors"]) if f is not None]
    tensor, multiplicity = _tensor_of(symmetric, sources)
    assert 2 in multiplicity and 1 in multiplicity
    counts = np.diff(tensor.target_ptr)
    merged = tr_mod.merge_paired_expansion(tr_mod.expand_tensor_rows(tensor), counts, tensor.has_gradient, multiplicity)
    for k, s in enumerate(sources):
        row = symmetric["rows"][s]
        donor, value, gradient = merged.source_rows(k, 1, True)
        assert donor.tobytes() == row.donor_ids.tobytes()
        assert value.tobytes() == row.value.tobytes()
        assert np.ascontiguousarray(gradient).tobytes() == np.ascontiguousarray(row.gradient).tobytes()
    plain = tr_mod.expand_tensor_rows(tensor)
    assert tr_mod.merge_paired_expansion(plain, counts, tensor.has_gradient, np.ones(len(counts), int)) is plain
    with pytest.raises(ValueError, match="multiplicity"):
        tr_mod.merge_paired_expansion(plain, counts, tensor.has_gradient, multiplicity[:-1])
    with pytest.raises(ValueError, match="multiplicity"):
        tr_mod.merge_paired_expansion(plain, counts, tensor.has_gradient, np.full(len(counts) // 2, 3))


def test_take_sources_keeps_the_rows_of_the_chosen_sources(symmetric):
    sources = [s for s, f in enumerate(symmetric["factors"]) if f is not None]
    tensor, _ = _tensor_of(symmetric, sources)
    full = tr_mod.expand_tensor_rows(tensor)
    counts = np.diff(tensor.target_ptr)
    for chosen in ([0], [1, 3, 4], [len(counts) - 1, 2], list(range(0, len(counts), 2)), list(range(len(counts)))):
        sub = tensor.take_sources(chosen)
        expansion = tr_mod.expand_tensor_rows(sub)
        assert sub.n_sources == len(chosen) and np.array_equal(sub.layers, tensor.layers[chosen])
        for k, s in enumerate(chosen):
            for a, b in zip(expansion.source_rows(k, int(counts[s]), True), full.source_rows(s, int(counts[s]), True)):
                assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    assert tensor.take_sources([]).n_sources == 0


# ---------------------------------------------------------------------------
# corrupt members
# ---------------------------------------------------------------------------
def test_corrupt_paired_members_are_rejected(symmetric):
    chunk, factors = symmetric["chunk"], symmetric["factors"]
    arrays = _arrays(encode_chunk("cells", chunk, factors=factors))
    encoding = arrays["src_encoding"]
    pair = int(np.flatnonzero(encoding == ENCODING_PAIRED)[0])
    csr = int(np.flatnonzero(encoding == 0)[0])
    at = np.arange(len(encoding))
    for mutated in (np.where(at == pair, 3, encoding),                       # an unknown code
                    np.where(at == csr, -1, encoding),
                    np.where(at == csr, ENCODING_PAIRED, encoding),          # a source that stores donors
                    np.where(arrays["src_conditioned"], ENCODING_PAIRED, encoding),
                    encoding[:-1]):
        broken = dict(arrays, src_encoding=mutated.astype(np.int8))
        for decode in (art._arrays_to_point_chunk, lambda x: art._arrays_to_point_chunk_v3(x, expand=False)):
            with pytest.raises(ValueError, match="src_encoding"):
                decode(broken)
    with pytest.raises(ValueError, match="corrupted"):
        art._arrays_to_point_chunk(dict(arrays, tr_t_eta=arrays["tr_t_eta"][:-1]))
    # the two parts of a pair must be of one family (a singleton A with a ringwise B is a valid TensorRows, not a valid pair)
    singleton = next(f for f in factors if isinstance(f, PairedFactors) and f.a.family == "singleton")
    ringwise = next(f for f in factors if isinstance(f, PairedFactors) and f.a.family == "ringwise")
    members = tr_mod.build_tensor_rows([singleton.a, ringwise.b], [True, True]).to_arrays()
    with pytest.raises(ValueError, match="differ in family"):
        art._tensor_rows(members, np.array([ENCODING_PAIRED]), np.array([1]), np.array([True]))
    art._tensor_rows(members, np.array([ENCODING_TENSOR, ENCODING_TENSOR]), np.array([1, 1]), np.array([True, True]))


# ---------------------------------------------------------------------------
# the loader needs no change: the factored view lowers like the dense chunk
# ---------------------------------------------------------------------------
def test_lowering_the_factored_view_of_a_paired_chunk_matches_the_dense_chunk(symmetric):
    from drbx.stencils.loader import FactoredChunk, LoaderGrid, lower_point_chunks
    from tests.test_stencils_tensor_loader import (
        TOL, _apply, _assert_same_layout, _fields, _relative, _weight_scales)

    chunk, factors, context = symmetric["chunk"], symmetric["factors"], symmetric["context"]
    data = encode_chunk("cells", chunk, factors=factors)
    item = FactoredChunk(*art.decode_chunk_factored("cells", data))
    dense = decode_chunk("cells", data)
    assert item.tensor_rows is not None and item.is_tensor.any()
    grid = LoaderGrid.from_context(context)
    plan_f = lower_point_chunks([item], grid=grid)
    plan_d = lower_point_chunks([dense], grid=grid)
    _assert_same_layout(plan_f, plan_d)
    assert plan_f.payload.tensor_batches and not plan_d.payload.tensor_batches
    fields = _fields(context)
    v_f, g_f = (np.asarray(x) for x in _apply(plan_f, fields))
    v_d, g_d = (np.asarray(x) for x in _apply(plan_d, fields))
    scale_v, scale_g = _weight_scales(dense)
    magnitude = np.abs(fields).max()
    assert _relative(v_f - v_d, np.maximum(scale_v, 1e-300)[:, None] * magnitude) <= TOL
    rows = np.flatnonzero(plan_d.targets.gradient_slot >= 0)
    slots = plan_d.targets.gradient_slot[rows]
    assert _relative(g_f[slots] - g_d[slots], np.maximum(scale_g[rows], 1e-300)[:, :, None] * magnitude) <= TOL
