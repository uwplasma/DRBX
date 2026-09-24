# Cached endpoint reconstruction

The projected-FCI runner now wraps its prepared `OwnerMoments` geometry in
`CachedOwnerMoments`. Geometry arrays, owner moments and KD trees are reused;
they are not constructed twice. The wrapper is local to this campaign, so the
preceding single/four-seed return runners retain their existing implementation.

For each endpoint/eta-plane fit, the wrapper translates the moments of the
existing candidate-owner pool once, then gathers the same candidate supports
in their original order. Absolute moment gathers are also reused. Plane
membership uses the stored owner-eta label instead of searching a plane-wide
list; directional selection uses a boolean membership array while retaining
its original donor order. The cache is discarded after each call, including
failure. Models belong to individual workers and reject reentrant use.

The frozen candidate fits, SVDs, weights, rank/feasibility checks, scores,
fallbacks, quadrature, tracing, eta interpolation and manufactured fields are
unchanged. The standalone helper methods remain uncached outside endpoint_pair.
The geometry is immutable while a model is in use.

## Bounded evidence

The local actual-HSX prototype replay checked 9,175 endpoint points over
N32/N48/N64, including accepted wall/seam/interior traces, all N32 preflight
batches and canonical probes covering axis/aggregate support. Endpoint sparse
maps and selector metadata were bitwise identical. Optional quadratic controls,
natural and forced fallback branches, invalid-plane failure and cache cleanup
were checked. Full/half maps for six archived wall/seam patches were reproduced
bitwise after canonical CSR sorting. The integrated implementation repeats
this audit; the compact receipt is in validation/endpoint_optimization.json.

A separate uninstrumented, alternating local benchmark measured endpoint-map
speedups of 1.4243x / 1.4230x / 1.4208x at N32/N48/N64. A warmed eight-face N32
full-preparation benchmark, including fresh traces and all field/reference
channels, fell from median 3.6922 s to 3.2358 s: 1.1411x, or 12.36% less time.
All prepared arrays and field outputs matched bitwise. This sample covers
radial/eta faces and axis/aggregate support but not theta faces or span halving.
It excludes initial context/JIT and checkpoint I/O. These measurements are not
Perlmutter scaling results or a promised remote completion time.

The committed actual-HSX owner-plane fixtures reproduce the optimized versus
frozen fit, complete selector metadata, controls and fallbacks on all three
resolutions without external geometry inputs. They test equivalence, not
convergence. Run:

```bash
python -m pytest -q tests/test_q_fci_projected_endpoints.py \
  tests/test_q_fci_projected_campaign.py tests/test_q_fci_return_campaign.py
```

## Restart contract

The source identity changes. This version does not support checkpoint migration
from the earlier runner. Use one newly created campaign folder and run verify,
all-resolution preflight, N32, N48/N64, final validation and status as documented
in README.md. The old campaign remains a separate artifact set. Only immutable
input datasets and the established environment may be reused. Do not copy old
trace/map/result receipts or override identity checks.

The full/half/fourth numerical method and scientific gates are unchanged. This
is an implementation optimization, not a new scientific qualification result.
