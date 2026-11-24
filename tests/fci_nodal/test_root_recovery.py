import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from drbx.fci_nodal.case._mms_stages import _with_root_recovery
from drbx.fci_nodal.operators.parallel.owner_runtime import coupled_records


@pytest.fixture
def records():
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("records",))
    sharding = NamedSharding(mesh, P("records"))
    return mesh, sharding


def _inputs(sharding, *, needs_recovery):
    neutral = [1., 1., 1., 0., 0., 0.]
    difficult = [.6, .225, 2., 95., -30., 0.]
    faces = np.array([neutral, neutral, difficult if needs_recovery else neutral])
    jumps = np.full((3, 6), .125)
    ids = np.arange(3, dtype=np.int64)
    return tuple(jax.device_put(value, sharding) for value in (faces, jumps, ids))


def test_branchless_face_scan_lowers_a_smaller_program(records):
    mesh, sharding = records
    args = _inputs(sharding, needs_recovery=False)
    full = coupled_records(mesh, (1., 1836.), 2).lower(*args).as_text()
    explicit = coupled_records(mesh, (1., 1836.), 2, robust=True).lower(*args).as_text()
    optimistic = coupled_records(mesh, (1., 1836.), 2, robust=False).lower(*args).as_text()
    assert full == explicit
    assert "stablehlo.case" in full and "stablehlo.case" not in optimistic
    assert len(optimistic) < len(full)


@pytest.mark.parametrize("needs_recovery", [False, True])
def test_optimistic_scan_flag_and_host_retry(records, needs_recovery):
    mesh, sharding = records
    optimistic = coupled_records(mesh, (1., 1836.), 2, robust=False)
    args = _inputs(sharding, needs_recovery=needs_recovery)
    calls = []

    def try_analytic(*args):
        value, status, flag = optimistic(*args)
        return value, {"records": status, "needs_root_recovery": flag}

    def recover(*args):
        full = coupled_records(mesh, (1., 1836.), 2)
        value, status = full(*args)
        calls.append((value, status))
        return value, {"records": status}

    _, analytic = try_analytic(*args)
    assert bool(analytic["needs_root_recovery"]) == needs_recovery
    value, aux = _with_root_recovery(try_analytic, recover)(*args)
    assert int(aux["records"][0]) == 0
    assert len(calls) == int(needs_recovery)
    if needs_recovery:
        expected, expected_status = calls[0]
        assert "needs_root_recovery" not in aux
        assert np.asarray(value).tobytes() == np.asarray(expected).tobytes()
        for actual, expected in zip(aux["records"], expected_status, strict=True):
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
