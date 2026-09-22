# Local implementation validation

Validated in an isolated checkout of the committed centered dependencies,
with only the new P05 campaign added. Unrelated local research modifications
are excluded from the source manifest and eventual commit.

- Source/input verification: 2,431 centered input files pinned by content,
  alongside the existing immutable HSX geometry/reference input manifest.
- Actual-HSX preflight: N32 28 faces; N48 952 faces; N64 1,230 faces.
  Every reconstructed central donor hash and common flux replays against the
  completed centered campaign.
- Complete N48/N64 bounded jump/action replay: maximum complete-owner action
  differences from the worker's saved candidate are 1.78e-14 and 3.76e-14.
  The kernel stages took approximately 2.9 and 3.7 seconds respectively,
  excluding input verification/context startup. These are not global timing
  forecasts.
- A 48-face actual-HSX serial/two-process test passes; maximum jump-flux
  difference is 1.66e-23. Changing query-batch length changes floating reduction
  order, so justified floating tolerances replace a bitwise-equality demand.
- Resume preserves existing checkpoint timestamps. Corrupt output and changed
  identities are rejected. The bounded parallel check exposed and fixed a
  colliding `campaign` module import; multiprocessing now uses the qualified
  `p05_material_campaign` package name.
- Focused tests: `tests/test_p05_material_campaign.py` plus
  `tests/test_hsx_clean_remote_campaign.py`: **11 passed**. This includes actual
  HSX complete-owner incidence replay at both fine resolutions, periodic-endpoint
  bookkeeping, exact global chunk partition, corruption/identity handling, and
  per-field two-interval/reference-budget gate arithmetic. The arithmetic tests
  provide no independent numerical convergence evidence.

Detailed local receipts are under workspace
`work/p05_material_remote_preparation_20260921/release/`.
No global run was launched locally. Static accuracy, structural properties and
production suitability remain distinct scientific questions; this document
certifies bounded implementation checks only.
