"""Tests for ``scripts/p_shared/replay_units.py`` -- the unit-parallel
replay refactor (P08 step-1 campaign task). Fully synthetic: no real
geometry, no row artifact, no oracle data -- exercises the generic pieces
every campaign's per-unit compute function shares (sparse owner-space
scatter, the owner accumulator, the pack/unpack roundtrip a unit's on-disk
chunk uses, and the build-plan-driven unit list / manifest-entry lookup a
unit's chunk-file read depends on).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from p_shared import replay_units as ru  # noqa: E402


# ---------------------------------------------------------------------------
# Sparse owner-space scatter -- the additivity trick every campaign's unit
# compute relies on (module docstring, "Owner-space additivity").
# ---------------------------------------------------------------------------
def test_sparse_scatter_matches_dense_add_at():
    term = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weight = np.array([1.0, 2.0, 0.5])
    owner_ids = np.array([2, 0, 2])
    uniq, values = ru._sparse_scatter(term, weight, owner_ids)

    dense = np.zeros((3, 2))
    np.add.at(dense, owner_ids, weight[:, None] * term)
    reconstructed = np.zeros((3, 2))
    reconstructed[uniq] = values
    np.testing.assert_allclose(reconstructed, dense)


def test_sparse_scatter_no_weight_is_plain_sum():
    term = np.array([[1.0], [2.0], [3.0]])
    owner_ids = np.array([0, 0, 1])
    uniq, values = ru._sparse_scatter(term, None, owner_ids)
    dense = np.zeros((2, 1))
    dense[uniq] = values
    np.testing.assert_allclose(dense, [[3.0], [3.0]])


def test_sparse_scatter_summed_across_two_calls_matches_one_call():
    """The core additivity claim: splitting a domain into two "units" and
    summing their sparse scatters reproduces one call over the whole
    domain -- exactly what a reduction over many artifact-chunk units
    does."""
    rng = np.random.default_rng(0)
    term = rng.normal(size=(10, 3))
    owner_ids = rng.integers(0, 4, size=10)
    weight = rng.uniform(0.5, 2.0, size=10)

    uniq_all, values_all = ru._sparse_scatter(term, weight, owner_ids)
    dense_all = np.zeros((4, 3))
    dense_all[uniq_all] = values_all

    dense_split = np.zeros((4, 3))
    for lo, hi in ((0, 4), (4, 10)):
        uniq, values = ru._sparse_scatter(term[lo:hi], weight[lo:hi], owner_ids[lo:hi])
        np.add.at(dense_split, uniq, values)

    np.testing.assert_allclose(dense_split, dense_all)


def test_sparse_scatter_signed_matches_lower_plus_upper_minus():
    term = np.array([[1.0], [2.0]])
    lower_owner = np.array([0, 1])
    upper_owner = np.array([1, -1])  # -1: no upper side (dropped, matching every campaign's own convention)
    uniq, values = ru._sparse_scatter_signed(term, lower_owner, upper_owner, +1.0, -1.0)
    dense = np.zeros((2, 1))
    dense[uniq] = values
    # owner 0: +term[0] = 1; owner 1: +term[1] (lower) - term[0] (upper) = 2 - 1 = 1
    np.testing.assert_allclose(dense, [[1.0], [1.0]])


def test_sparse_scatter_both_sides_adds_independently():
    lower = np.array([[1.0], [2.0]])
    upper = np.array([[10.0], [20.0]])
    lower_owner = np.array([0, 1])
    upper_owner = np.array([0, 1])
    uniq, values = ru._sparse_scatter_both_sides(lower, upper, lower_owner, upper_owner)
    dense = np.zeros((2, 1))
    dense[uniq] = values
    # both sides ADD (not an antisymmetric jump): owner 0 gets 1+10, owner 1 gets 2+20.
    np.testing.assert_allclose(dense, [[11.0], [22.0]])


def test_owner_accumulator_add_is_additive_and_ignores_empty():
    acc = ru.OwnerAccumulator(3, (2,))
    acc.add(np.array([], dtype=np.int64), np.zeros((0, 2)))
    acc.add(np.array([0, 2]), np.array([[1.0, 1.0], [2.0, 2.0]]))
    acc.add(np.array([0]), np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(acc.total, [[2.0, 2.0], [0.0, 0.0], [2.0, 2.0]])


# ---------------------------------------------------------------------------
# Pack/unpack roundtrip -- what a unit's on-disk chunk file actually stores.
# ---------------------------------------------------------------------------
def test_pack_unpack_roundtrip_pair_list_dict_scalar_array():
    payload = {
        "a_pair": (np.array([0, 2], dtype=np.int64), np.array([[1.0, 2.0], [3.0, 4.0]])),
        "a_list": [(np.array([1], dtype=np.int64), np.array([[9.0]])),
                  (np.array([0, 1], dtype=np.int64), np.array([[1.0], [2.0]]))],
        "a_dict": {"foo": (np.array([0], dtype=np.int64), np.array([[5.0]])),
                  "bar": (np.array([1], dtype=np.int64), np.array([[6.0]]))},
        "a_scalar": 3.5,
        "an_array": np.array([1, 2, 3]),
    }
    arrays = ru._pack_unit_arrays(payload)
    restored = ru._unpack_unit_arrays(arrays)

    np.testing.assert_array_equal(restored["a_pair"][0], payload["a_pair"][0])
    np.testing.assert_array_equal(restored["a_pair"][1], payload["a_pair"][1])
    assert len(restored["a_list"]) == 2
    for (o1, v1), (o2, v2) in zip(restored["a_list"], payload["a_list"]):
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(v1, v2)
    assert set(restored["a_dict"]) == {"foo", "bar"}
    for key in ("foo", "bar"):
        np.testing.assert_array_equal(restored["a_dict"][key][0], payload["a_dict"][key][0])
        np.testing.assert_array_equal(restored["a_dict"][key][1], payload["a_dict"][key][1])
    assert restored["a_scalar"] == pytest.approx(3.5)
    np.testing.assert_array_equal(restored["an_array"], payload["an_array"])


def test_pack_unit_arrays_rejects_unsupported_type():
    with pytest.raises(TypeError):
        ru._pack_unit_arrays({"bad": object()})


def test_pack_unpack_roundtrips_through_actual_npz(tmp_path):
    payload = {"term": (np.array([0, 1], dtype=np.int64), np.array([[1.0], [2.0]]))}
    arrays = ru._pack_unit_arrays(payload)
    path = tmp_path / "unit.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as z:
        loaded_arrays = {name: z[name] for name in z.files}
    restored = ru._unpack_unit_arrays(loaded_arrays)
    np.testing.assert_array_equal(restored["term"][0], payload["term"][0])
    np.testing.assert_array_equal(restored["term"][1], payload["term"][1])


# ---------------------------------------------------------------------------
# artifact_plan_units / manifest-entry lookup -- the build-plan-driven unit
# boundaries this module's "align with the artifact's build units" design
# depends on, checked against a small synthetic plan.json/manifest.json
# (the same on-disk shape build_artifact.py writes), never a real build.
# ---------------------------------------------------------------------------
def test_artifact_plan_units_reads_plan_json_with_chunk_index(tmp_path):
    grid_dir = tmp_path / "N8"
    grid_dir.mkdir()
    plan = {
        "identity": {"x": 1},
        "plan": {
            "cells": [{"stage": "cells", "n": 8, "start": 0, "stop": 4},
                     {"stage": "cells", "n": 8, "start": 4, "stop": 8}],
            "faces": [{"stage": "faces", "n": 8, "start": 0, "stop": 2}],
            "p07": [],
        },
    }
    (grid_dir / "plan.json").write_text(json.dumps(plan))
    units = ru.artifact_plan_units(tmp_path, 8)
    assert [u["chunk_index"] for u in units["cells"]] == [0, 1]
    assert units["cells"][1]["start"] == 4 and units["cells"][1]["stop"] == 8
    assert [u["chunk_index"] for u in units["faces"]] == [0]
    assert units["p07"] == []


def test_manifest_entry_lookup_by_group_stage_index():
    manifest = {"chunks": {"cells": [{"file": "rows/cells_cells_00000.npz", "sha256": "a"},
                                     {"file": "rows/cells_cells_00001.npz", "sha256": "b"}]}}
    entry = ru._manifest_entry(manifest, "cells", "cells", 1)
    assert entry["file"] == "rows/cells_cells_00001.npz"
    with pytest.raises(KeyError):
        ru._manifest_entry(manifest, "cells", "cells", 5)


def test_face_row_selection_excludes_collapsed_and_alias(monkeypatch):
    from types import SimpleNamespace
    census = SimpleNamespace(collapsed_r0=np.array([True, False, False, False]),
                             legacy_alias_slots=np.array([False, False, True, False]))
    selected = ru.face_row_selection(census)
    np.testing.assert_array_equal(selected, [1, 3])


# ---------------------------------------------------------------------------
# Grid-generic oracle paths (task: "make every oracle path grid-generic") --
# ``_load_oracle_owner_values`` must read N{n}, not a hardcoded N32, for
# whatever grid its ``Environment`` says -- fully synthetic (tmp_path-based
# fake oracle npz files, never real geometry/oracle data).
# ---------------------------------------------------------------------------
def _fake_oracle_tree(root: Path, n: int) -> dict:
    """A minimal synthetic oracle tree, one tiny npz per campaign at grid
    ``n`` only (never at any other grid), matching exactly the file layout
    ``_load_oracle_owner_values`` reads."""
    p05 = root / "p05"; (p05 / "reuse_inputs").mkdir(parents=True)
    np.savez(p05 / "reuse_inputs" / f"N{n}.reuse.npz", observations=np.array([1.0, 2.0]))

    p05n_frozen = root / "p05n_frozen"; p05n_frozen.mkdir()
    np.savez(p05n_frozen / f"N{n}.owner_values.npz", values=np.array([3.0]))

    upwind_root = root / "upwind"
    (upwind_root / "p05n_upwind").mkdir(parents=True)
    np.savez(upwind_root / "p05n_upwind" / f"N{n}.owner_values.npz", values=np.array([4.0]))
    (upwind_root / "p06n").mkdir(parents=True)
    np.savez(upwind_root / "p06n" / f"N{n}.owner_values.npz", values=np.array([5.0]))

    p06_legacy = root / "p06_legacy"; p06_legacy.mkdir()
    np.savez(p06_legacy / f"N{n}.prepare.npz", owner_values=np.array([6.0]))

    p07 = root / "p07"; p07.mkdir()
    np.savez(p07 / "owner_values.npz", **{f"N{n}": np.array([7.0])})

    p07n = root / "p07n"; p07n.mkdir()
    np.savez(p07n / f"N{n}.owner_values.npz", values=np.array([8.0]))

    return {"p05": p05, "p05n_frozen": p05n_frozen, "p05n_p06n_upwind": upwind_root,
           "p06_legacy": p06_legacy, "p07": p07, "p07n": p07n}


def test_load_oracle_owner_values_is_grid_generic_not_hardcoded_to_n32(tmp_path):
    from types import SimpleNamespace
    from p_shared.replay_support import CAMPAIGN_FUNCS

    paths_48 = _fake_oracle_tree(tmp_path / "grid48", 48)
    env_48 = SimpleNamespace(n=48)
    out_48 = ru._load_oracle_owner_values(env_48, paths_48, CAMPAIGN_FUNCS)
    assert out_48["p05"]["owner_values"].tolist() == [1.0, 2.0]
    assert out_48["p05n_frozen"]["owner_values"].tolist() == [3.0]
    assert out_48["p05n_upwind"]["owner_values"].tolist() == [4.0]
    assert out_48["p06n"]["owner_values"].tolist() == [5.0]
    assert out_48["p06_legacy"]["owner_values"].tolist() == [6.0]
    assert out_48["p07"]["owner_values"].tolist() == [7.0]
    assert out_48["p07n"]["owner_values"].tolist() == [8.0]

    # A tree that only has N32 files must NOT be silently read for n=48 --
    # confirms this is genuinely grid-keyed, not still hardcoded.
    paths_32_only = _fake_oracle_tree(tmp_path / "grid32", 32)
    with pytest.raises(FileNotFoundError):
        ru._load_oracle_owner_values(env_48, paths_32_only, CAMPAIGN_FUNCS)
