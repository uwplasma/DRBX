# Structured P05/P06 execution optimization, 25 September 2026

The numerical reconstruction, quadrature, fields, and primary references are
unchanged. This release removes unused P06 full-RHS geometry preparation,
shares exact-coordinate metric queries within bounded batches, caches exact
singleton interpolation weights, skips unused face-gradient contractions, and
loads P05 topology once per worker. It does not introduce a GPU kernel.

Local validation against 2ad730c82de8a669205d7d635fe450e1255b17cb covered N32
and N64 axis/first-ring/interior/seam/near-wall/wall samples. All 24 executable
old/new kernel comparisons were bitwise identical. Two additional N64 q7/q9
cell-reference comparisons exposed a pre-existing fixed radial curl stencil
leaving [0,1]. Only these otherwise-invalid stencil points now shrink their
radial step to 0.2 times boundary distance. The q3 candidate and q5 global
reference keep the fixed step on N32/N48/N64. The repaired controls are finite;
they are not claimed to be an independent reference-accuracy qualification.

Warmed local timings (small sampled batches, not whole-campaign estimates):
P06 q5 references approximately 2.8x faster; P06 candidate cells approximately
2.5–2.6x; face stages approximately 1.04–1.12x. P05 benefits vary by batch,
roughly 1.1–2x in the sampled cell/reference work. No total remote speedup is
promised. First-call JIT/geometry initialization timings are excluded from
these estimates. The focused suite has 34 passing tests.

## Explicit continuation of a prior campaign

Stop its controller and workers before updating the same checkout path. The
`adopt-optimization` command validates immutable inputs and an exact release
allowlist before creating `optimization_execution.json`. It does not rewrite
old manifests, numerical arrays, chunk hashes, or source identities. Their
original numerical lineage is retained and the new execution provenance is
recorded separately. Newly computed chunks carry the execution certificate.
The release allowlist is `optimization_release.json`; unrelated source changes,
changed inputs/configuration and incompatible baseline chains are rejected.

The P05 baseline is exactly 2ad730c8 (the nine-input repair). P06's sources at
551ebbd7 and 2ad730c8 are identical and both match the allowlist. A different
checkout path is not a supported continuation for P06's absolute-path source
identities. Use a fresh campaign or separately audit such a migration.

Each runner accepts:

```
python scripts/p05_structured_global/campaign.py adopt-optimization --input-root "$INPUT_ROOT" --output "$P05_OUTPUT"
python scripts/p06_structured_global/campaign.py adopt-optimization --input-root "$INPUT_ROOT" --output "$P06_OUTPUT"
```

Then run its usual preflight, run, and validate commands with allocation-sized
CPU workers and memory caps. Existing validation still verifies payload hashes,
exact coverage, preparation/runtime identities and one-writer locks. Input
verification is not bypassed. An invalid upgrade must return the obstacle;
never edit a manifest or receipt to make it pass.
