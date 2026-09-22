# Bounded implementation verification

Validated 21 September 2026 against the existing real-HSX direct cubic
prototype and qualified q9/q11 references. This verifies implementation
equivalence, not global convergence or diffusion structural properties.

| Resolution | Checked faces | Identical ordered donor lists | Max interior flux difference | Max complete action difference |
|---:|---:|:---:|---:|---:|
| 32 | 379 | yes | 1.41e-16 | 2.75e-12 |
| 48 | 523 | yes | 2.73e-16 | 3.82e-11 |
| 64 | 620 | yes | 5.02e-16 | 1.23e-10 |

The action differences include amplification of roundoff by small owner
volumes. They are far below the existing sampled spatial errors. Coefficients
differ by at most 4.62e-16; independent raw-cell moment evaluation differs by
at most 2.73e-12 in the scaled polynomial entries, and independent primal/dual
flux evaluation differs by less than 7.1e-17 on 25 spread/worst-condition faces
per resolution. Maximum relative cubic defect is 1.49e-12.

On those 25-face samples, optimized setup/action was approximately 2.45x,
4.53x and 7.96x faster than the original N32/N48/N64 prototype routine.
**The prototype also performs an extra primal verification solve**; these
numbers include removing that redundant execution-time check and are not
isolated tree-search speedups. Small concurrent benchmark timings are
indicative, not a guaranteed global speedup. At N32, indexed queries alone
were slower than exhaustive ranking restricted to already-indexed planes;
the intended savings are removal of whole-domain scans, reusable moments,
batching, and scaling with resolution.

The 620-face N64 sample ran in about 0.49 seconds after input preparation,
with peak serial RSS about 95 MiB. Two-worker actions were bitwise identical.
This includes the full N64 scalar owner pool but only bounded face inputs;
it is not a measured full-domain runtime/memory bound.

Fresh continuous q9/q11 references were rebuilt on all 379 selected N32 faces.
Serial and two-worker reference fluxes exactly match each other and the saved
qualified reference. Reference-worker peak RSS was about 940 MiB at N32.
Two-worker preparation was slower on this tiny sample because each process
loads geometry; it remains available for larger independent reference chunks.
N64 reference-worker memory and full-domain reference time remain unmeasured.

Focused verification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_q03_direct_campaign.py tests/test_q03_exchange_campaign.py
```

Result: **8 passed**. Coverage includes actual-HSX prototype replay, serial/
parallel identity, resume without rewriting valid chunks, duplicate locking,
corrupt inputs/chunks, missing coverage, and refusal to treat sample norms or
unqualified references as global convergence. Scalar order-acceptance tests
exercise bookkeeping only; they supply no HSX numerical evidence.

Detailed evidence and the independent benchmark script are in workspace
`work/parallel_q03_direct_runner_validation_20260921/`. The 924-KiB committed-
candidate regression fixture under `tests/data/q03_direct_hsx_excerpt.*`
records its geometry and prototype provenance. No global campaign or production operator change was performed during that validation.


## Portable remote preparation validation

The remote adapter exports frozen scalar observations, moments, canonical
face/grid data and targets once; only continuous q9/q11 reference generation
and direct reconstruction remain remote. It no longer loads historical research
modules or full simulation geometry. The exact analytical field class and
quadrature definition are frozen in `frozen_mms.py` and `continuous.py`.

Fresh portable references and serial/parallel reconstruction passed on all
379/523/620 bounded HSX faces at N32/N48/N64, including in a clean checkout of
base revision `257bd55f` with only the new campaign and tests added. Maximum
prototype action differences were `2.75e-12/3.81e-11/1.23e-10`. Both execution
paths used two process workers in this local implementation check; this does
not prescribe remote resources. The external metric and MAKEGRID data were
verified by SHA256. No local historical module was needed for the clean run.

The metric alias used by the prototype resolves to the corrected d58 cache,
whose SHA256 is `216dbcfb343fa23dd7ac9e60da6592f5757c7f5e45c8d96a5b642c6419dc10a1`.
The old Q01 catalogue records a different historical cache hash. Scalar states
are exported unchanged; fresh reference provenance pins the actual evaluator.
The clean portable preflight independently reproduces the previously qualified
q9/q11 fluxes with this corrected evaluator.

Focused tests now total **10 passed**, adding remote seed/source-identity
rejection and completion-validator reassembly/corruption checks. Sample
results cannot be promoted to global certification by the validator. These
are implementation checks, not new scientific acceptance requirements.

Detailed operational evidence is in workspace
`work/q03_direct_remote_preparation_20260922/`, including the clean-checkout
preflight and transferable seed archive. The archive contains about 333 MB
of uncompressed inputs; it omits the already available metric/MAKEGRID files.
