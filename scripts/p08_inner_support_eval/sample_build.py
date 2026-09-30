"""Owner sample for the C0-vs-candidate evaluation: the 29 September audit's roles and fixed-u tracks, C1's interface roles, and
a seeded per-ring sample over rings 0 .. last+3.  Geometry indices only; no error is computed here.

Every owner entry is a dict ``{owner, raw, radial, kind, role, phase, ring}``; one owner may carry several tags.
``kind`` is ``role`` (audit roles: Q's 36 owners, reused verbatim, plus two fresh phases each), ``c1role`` (C1 interface
roles), ``track`` (fixed-u tracks) or ``ring`` (the per-ring sample).  Groups are derived from the tags (``groups_of``).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Q_SAMPLE = HERE / 'sample.json'                                     # vendored from work/q_fci_coupled_ringwise_overlap_20260928

THETA0, ETA0 = 0.9, 1.7
TRACKS = [('near_axis', 0.02), ('u012', 0.12), ('u020', 0.20), ('u030', 0.30), ('u045', 0.45), ('u070', 0.70)]
AUDIT_ROLES = ('coupled_control', 'switch', 'switch_plus1', 'switch_plus2', 'last_aggregate', 'first_singleton')
C1_ROLES = (('c1_last_m1', -1), ('c1_last', 0), ('c1_single1', 1), ('c1_single2', 2))     # ring = last + offset
RING_EXTRA = 3                                                      # the ring sample covers rings 0 .. last + 3
SEAM_J = (0, -1)                                                    # theta index 0 and n-1
SEAM_K = (0, 1, -2, -1)                                             # eta indices 0, 1, n-2, n-1
SEED = 20260930


def _nearest_periodic(centers, target, period):
    d = np.abs((centers - target + period / 2) % period - period / 2)
    return int(np.argmin(d))


def _entry(t, n, i, j, k, **tags):
    raw = int((i * n + j) * n + k)
    return dict(n=n, owner=int(t.ro[raw]), raw=raw, radial=int(i), **tags)


def audit_role_entries(t, n):
    """Q's owners (reused verbatim) and two fresh phases per (role, grid): same radial layer, new theta and eta."""
    q = [z for z in json.loads(Q_SAMPLE.read_text()) if z['N'] == n]
    out, by_role = [], {}
    for z in q:
        for role in z['roles']:
            out.append(dict(n=n, owner=z['owner'], raw=z['selected_raw'], radial=z['radial'], kind='role', role=role,
                            phase=z['phase'], ring=z['radial']))
            by_role.setdefault(role, []).append(z)
    for role, entries in sorted(by_role.items()):
        i = entries[0]['radial']
        k_used = {int(np.unravel_index(e['selected_raw'], (n, n, n))[2]) for e in entries}
        k_new = (max(k_used) + n // 3) % n
        for tag, frac in (('fresh_a', 0.25), ('fresh_b', 0.625)):
            out.append(_entry(t, n, i, int(round(n * frac)) % n, k_new, kind='role', role=role, phase=tag, ring=i))
    return out


def c1_role_entries(t, n, last):
    """C1 interface roles: the last two agglomerated rings and the first two singleton rings, four phases each."""
    out = []
    j_ns, k_ns = int(round(0.11 * n)), int(round(0.2 * n))
    k_new = (k_ns + n // 3) % n
    for role, offset in C1_ROLES:
        i = last + offset
        for tag, (j, k) in (('nonseam', (j_ns, k_ns)), ('double_seam', (0, 0)),
                            ('fresh_a', (int(round(0.25 * n)) % n, k_new)), ('fresh_b', (int(round(0.625 * n)) % n, k_new))):
            out.append(_entry(t, n, i, j, k, kind='c1role', role=role, phase=tag, ring=i))
    return out


def track_entries(t, n):
    out = []
    for label, u0 in TRACKS:
        i = int(np.argmin(np.abs(t.centers[0] - u0)))
        j = _nearest_periodic(t.centers[1], THETA0, 2 * np.pi)
        k = _nearest_periodic(t.centers[2], ETA0, t.g.eta_period)
        out.append(_entry(t, n, i, j, k, kind='track', role=f'track_{label}', phase='track', ring=i, u0=u0))
    return out


def ring_entries(t, n, last, per_ring):
    """Seeded sample: ``per_ring`` owners per ring (rings 0 .. last+3) per phase.  ``nonseam``: random (j, k) away from
    the seams; ``double_seam``: random (j, k) among the theta-seam x eta-seam index combinations."""
    out = []
    lo, hi = n // 8, n - n // 8
    seam = [(j % n, k % n) for j in SEAM_J for k in SEAM_K]
    for i in range(0, last + RING_EXTRA + 1):
        for phase_idx, phase in enumerate(('nonseam', 'double_seam')):
            rng = np.random.default_rng([SEED, n, i, phase_idx])
            if phase == 'nonseam':
                jk = [(int(rng.integers(lo, hi)), int(rng.integers(lo, hi))) for _ in range(per_ring)]
            else:
                pick = rng.choice(len(seam), size=min(per_ring, len(seam)), replace=False)
                jk = [seam[int(p)] for p in pick]
            seen = set()
            for j, k in jk:
                e = _entry(t, n, i, j, k, kind='ring', role=f'ring_{i}', phase=phase, ring=i)
                if e['owner'] in seen:
                    continue
                seen.add(e['owner'])
                out.append(e)
    return out


def build_sample(t, n, last, per_ring=6):
    return (audit_role_entries(t, n) + c1_role_entries(t, n, last) + track_entries(t, n)
            + ring_entries(t, n, last, per_ring))


def owners_of(entries):
    return sorted({e['owner'] for e in entries})


def groups_of(entry, last):
    """Group ids an entry contributes to: ``role:<name>``, ``track:<name>``, ``ring:<i>``, ``pool:inner`` (ring sample,
    rings <= last), ``pool:extended`` (ring sample, rings <= last+3)."""
    kind = entry['kind']
    if kind in ('role', 'c1role'):
        return [f"role:{entry['role']}"]
    if kind == 'track':
        return [f"track:{entry['role']}"]
    groups = [f"ring:{entry['ring']}", 'pool:extended']
    if entry['ring'] <= last:
        groups.append('pool:inner')
    return groups
