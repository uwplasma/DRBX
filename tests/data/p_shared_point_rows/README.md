# Bounded real-HSX point-row regression fixtures

`N32/N48/N64.plan.npz` contain only the selected prepared point rows, donor
closure, boundary query coordinates, row family/eta-offset metadata, and cache
identities. `N32/N48/N64.heldout.npz` contain donor values for one smooth held-out
field pair at time `0.37`, prescribed boundary values and tangential gradients,
and the unchanged NumPy builder's expected values and logical gradients.
The six NPZ files total well below 1 MiB; no geometry mesh or global run array
is embedded. The test reconstructs a sparse owner-major input array from the
fixture-local donor values.

Regenerate from the workspace's validated P05/P06 inputs:

```bash
cd DRBX
conda run -n drb python ../work/p_shared_point_rows_extraction_20260926/fixture_replay.py --resolutions 32 48 64
XDG_CACHE_HOME=/tmp/drbx-p-shared-point-rows-cache conda run -n drb pytest -q tests/test_fci_perpendicular_point_rows.py
```

The work artifact `fixture_provenance.json` records the fixture hashes, while
`host_replay.json` records source/geometry identities, exact selected keys and
owners, original-versus-extracted row replay, P05/P06 action checks, and timings.
These are bounded regression fixtures, not new global qualification evidence.
