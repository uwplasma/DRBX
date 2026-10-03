"""Portable harness gates; no GPUs or original campaign data needed."""
from collections import namedtuple
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import campaign
from boundary_cache import BoundaryCache

BC = namedtuple('BC', 'node query tangent normal')


def sample(bank, case):
    result = []
    nr = len(bank.raw)
    for fields in (6, 6, None):
        shapes = [(nr, 35), (nr, 3), (nr, 3, 2), (nr, 35)]
        if fields:
            shapes = [(fields, *shape) for shape in shapes]
        arrays = [np.full(shape, -.0) for shape in shapes]
        for a in arrays:
            view = np.moveaxis(a, 1 if fields else 0, 0)
            view[bank.wall_index] = case + np.arange(len(bank.wall_index))[:, None] if view.ndim == 2 else case + .125
        result.append(BC(*arrays))
    return tuple(result)


class ContractTests(unittest.TestCase):
    def bank(self, wall=(1, 3)):
        return SimpleNamespace(identity='test', raw=np.arange(5), wall_index=np.asarray(wall, dtype=int))

    def test_source_manifest(self):
        identity, manifest = campaign.check_source()
        self.assertEqual(len(identity), 64)
        self.assertEqual(manifest['method'], 'polynomial')
        self.assertEqual(manifest['repeats'], 7)

    def test_cache_exact_signed_zero(self):
        cache = BoundaryCache(sample)
        bank = self.bank()
        first = cache(bank, 2)
        again = cache(bank, 2)
        for a, b in zip(first, again):
            for x, y in zip(a, b):
                self.assertEqual(x.tobytes(), y.tobytes())
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
        self.assertLess(cache.bytes, sum(x.nbytes for g in first for x in g))

    def test_cache_protects_retained_values(self):
        cache = BoundaryCache(sample)
        bank = self.bank()
        first = cache(bank, 1)
        first[0].node[:] = 123
        again = cache(bank, 1)
        expected = sample(bank, 1)
        self.assertEqual(again[0].node.tobytes(), expected[0].node.tobytes())

    def test_cache_rejects_nonuniform_nonwall(self):
        def bad(bank, case):
            result = sample(bank, case)
            result[0].node[:, 0] = 4
            return result
        with self.assertRaisesRegex(ValueError, 'nonuniform'):
            BoundaryCache(bad)(self.bank(), 1)

    def test_cache_budget(self):
        with self.assertRaisesRegex(MemoryError, 'budget'):
            BoundaryCache(sample, max_bytes=1)(self.bank(), 0)

    def test_cache_all_or_no_wall(self):
        for wall in ((), tuple(range(5))):
            cache = BoundaryCache(sample)
            a, b = cache(self.bank(wall), 0), cache(self.bank(wall), 0)
            for aa, bb in zip(a, b):
                for x, y in zip(aa, bb):
                    self.assertEqual(x.tobytes(), y.tobytes())

    def test_identity_refuses_output_relabel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign.bind_paths(root / 'new', root / 'old', root / 'src', 'first')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                campaign.bind_paths(root / 'new', root / 'old', root / 'src', 'second')

    def test_old_run_cannot_be_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for new in (root, root / 'nested'):
                with self.assertRaisesRegex(ValueError, 'separate'):
                    campaign.bind_paths(new, root, root / 'src', 'identity')


if __name__ == '__main__':
    unittest.main()
