"""Candidate-aware dispatch check for the C0-vs-candidate evaluation (numpy only; no jax, no geometry).

``dispatch_check`` compares the C0 (``inner_support="profile7"``) owner rows with the candidate's rows on identical
owners and reports whether every row that changed did so for a legitimate reason.  It replaces the C1-only rule
("anchor ring <= last") with candidate-independent invariants:

* point rows (R1 / R2 / R3): with ``coupled`` meaning ``diagnostics["family"] == "coupled_quartic"``
  - coupled in neither C0 nor the candidate: bitwise identical;
  - coupled in both: identical, unless the candidate's row carries ``diagnostics["donor_policy"] == "nearest28"``;
  - any changed row is ``coupled_quartic`` in the candidate;
* R4 (P07 integrated rows, keyed by an integer id; ``.family == 7`` stands for coupled): the same rules, where the
  nearest-28 exemption applies when the *candidate's inner support* is nearest-28 (R4 rows carry no diagnostics).

``n_violations`` is the full count; ``violations`` lists at most ``MAX_LISTED`` of them.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

MAX_LISTED = 20
COUPLED_FAMILY = 'coupled_quartic'
COUPLED_CODE = 7
NEAREST28 = 'nearest28'


def is_nearest28_support(inner_support) -> bool:
    return str(inner_support).endswith(NEAREST28)


def row_family(row):
    return str(row.diagnostics.get('family', ''))


def same_point(a, b):
    return (np.array_equal(a.donor_ids, b.donor_ids) and np.array_equal(a.value, b.value)
            and np.array_equal(a.gradient, b.gradient) and a.boundary_conditioned == b.boundary_conditioned
            and np.array_equal(a.trace_target_points, b.trace_target_points))


def same_integrated(a, b):
    return (np.array_equal(a.donor_ids, b.donor_ids) and np.array_equal(a.weights, b.weights)
            and a.family == b.family)


def judge(identical, coupled_a, coupled_b, nearest28):
    """``True`` if a row with this (identical, C0 coupled, candidate coupled) status obeys the invariants."""
    if identical:
        return True
    if not coupled_b:                                # changed rows must be coupled in the candidate
        return False
    return (not coupled_a) or nearest28              # coupled in both: only the nearest-28 donor policy may change it


def dispatch_check(rows_c0, rows_cand, *, candidate_support):
    """``rows_c0`` / ``rows_cand``: the ``row_index`` mappings of the two builds (keys ``(request, entity)`` for point
    rows, an integer P07 id for R4)."""
    k0, k1 = rows_c0, rows_cand
    cand_nearest28 = is_nearest28_support(candidate_support)
    out = dict(rows=len(k0), same_keys=bool(set(k0) == set(k1)), identical=defaultdict(int), changed=defaultdict(int),
               violations=[], n_violations=0, candidate_support=str(candidate_support))
    n_bad = 0
    for key in k0:
        if key not in k1:                            # recorded through ``same_keys``; counted as a violation as well
            n_bad += 1
            if len(out['violations']) < MAX_LISTED:
                out['violations'].append(dict(key=str(key), kind='missing', reason='row absent from the candidate'))
            continue
        a, b = k0[key], k1[key]
        if isinstance(key, (int, np.integer)):       # R4: P07 integrated row
            kind = 'R4'
            identical = same_integrated(a, b)
            coupled_a, coupled_b = a.family == COUPLED_CODE, b.family == COUPLED_CODE
            nearest28 = cand_nearest28
        else:
            kind = key[0]
            identical = same_point(a, b)
            coupled_a, coupled_b = row_family(a) == COUPLED_FAMILY, row_family(b) == COUPLED_FAMILY
            nearest28 = b.diagnostics.get('donor_policy') == NEAREST28
        ok = judge(identical, coupled_a, coupled_b, nearest28)
        (out['identical'] if identical else out['changed'])[kind] += 1
        if not ok:
            n_bad += 1
            if len(out['violations']) < MAX_LISTED:
                out['violations'].append(dict(key=str(key), kind=kind, identical=bool(identical),
                                              coupled_c0=bool(coupled_a), coupled_candidate=bool(coupled_b)))
    if not out['same_keys']:
        n_bad += len(set(k1) - set(k0))             # rows only the candidate built
    out['identical'], out['changed'] = dict(out['identical']), dict(out['changed'])
    out['n_violations'] = n_bad
    return out
