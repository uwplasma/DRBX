from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree

PATH = Path(__file__).resolve().parents[1] / 'scripts/hsx_remote_qualification/selection.py'
spec = importlib.util.spec_from_file_location('clean_selection_test', PATH)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def context(points, ids):
    return SimpleNamespace(arrays={'owner_centroid_xy':np.asarray(points, float),
                                  'owner_flat_ids':np.asarray(ids)},
        plane_indices=(np.arange(len(points)),), plane_trees=(cKDTree(points),), dr=.03)


def test_all_octant_boundaries_stable_under_ulp_perturbations():
    angles = np.arange(8) * np.pi / 4
    for offset in (-4*np.finfo(float).eps, 0, 4*np.finfo(float).eps):
        points = np.column_stack((np.cos(angles+offset), np.sin(angles+offset)))
        np.testing.assert_array_equal(selection.sectors(points), np.arange(8))
    angles += 1e-8
    np.testing.assert_array_equal(selection.sectors(np.column_stack((np.cos(angles),np.sin(angles)))), np.arange(8))


def test_candidate_cutoff_includes_nearly_tied_neighbors_before_id_sort():
    # Low canonical ID lies just beyond the nominal k-nearest cutoff.
    points = np.array([[.1,0], [1.,0], [np.nextafter(1.,np.inf),0], [2.,0]])
    c = context(points, [7,90,3,1])
    for exact in (False,True):
        pool,_ = selection.nearest_pool(c,0,np.array([[0.,0]]),exact=exact,pool_count=2)
        np.testing.assert_array_equal(pool, [[0,2]])


def test_pool_and_sectors_independent_of_storage_order():
    a = np.arange(16)*np.pi/8
    points = np.column_stack((np.cos(a),np.sin(a)))
    ids = np.arange(16)[::-1]
    expected = None
    for permutation in (np.arange(16),np.arange(16)[::-1],np.random.default_rng(4).permutation(16)):
        c=context(points[permutation],ids[permutation])
        pool,_=selection.nearest_pool(c,0,np.zeros((1,2)),pool_count=12)
        chosen=selection.sector_select(c,pool,np.zeros((1,2)),np.ones(1),8)
        donor_ids=c.arrays['owner_flat_ids'][chosen]
        if expected is None: expected=donor_ids
        else: np.testing.assert_array_equal(donor_ids,expected)


def test_tolerance_groups_do_not_chain_across_distant_values():
    # Pairwise approximate equality is non-transitive: use a fixed group anchor.
    order=selection.tied_order(np.array([0.,.75,1.5,3.]),np.array([9,2,1,0]),1.)
    np.testing.assert_array_equal(order,[1,0,2,3])
