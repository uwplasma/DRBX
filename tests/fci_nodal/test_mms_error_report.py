import numpy as np

from drbx.fci_nodal.mms.error_report import FIELDS, REGIONS, error_rows, owner_partials


def test_owner_error_reduction_matches_direct_weighted_errors():
    weights = np.array([1., 4., 2., 3., 7., 99.])
    active = np.array([True, True, True, True, True, False])
    delta = np.arange(6.)[:, None] - np.arange(len(FIELDS))[None, :]
    for touch in (np.array([True, False, True, False, False, True]),
                  np.array([False, False, False, False, False, True])):
        owners = (slice(0, 3), slice(3, 6))
        partials = [owner_partials(weights[owner], active[owner], touch[owner], delta[owner])
                    for owner in owners]
        assert partials[1]["wall_touch"] == (0, 0., None, None)
        assert sum(part["all"][0] for part in partials) == 5
        assert sum(part["all"][1] for part in partials) == 17.
        rows = error_rows(partials)
        assert [(row["region"], row["field"]) for row in rows] == [
            (region, field) for region in REGIONS for field in FIELDS]
        for region, mask in zip(REGIONS, (active & touch, active & ~touch, active), strict=True):
            selected = [row for row in rows if row["region"] == region]
            rms = [row["absolute_Hp_RMS"] for row in selected]
            maximum = [row["max_node_error"] for row in selected]
            if mask.any():
                h = weights[mask]
                np.testing.assert_allclose(rms, np.sqrt(np.sum(h[:, None]*delta[mask]**2, axis=0)/h.sum()))
                np.testing.assert_array_equal(maximum, np.max(np.abs(delta[mask]), axis=0))
            else:
                assert rms == maximum == [None]*len(FIELDS)
