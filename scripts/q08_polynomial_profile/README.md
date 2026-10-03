# Q08 device-native characteristic split benchmark

This bounded implementation/performance test replaces only the generic
nonsymmetric eigensolver in the Q characteristic correction. It does not
change the centered operator, diffusion, reconstruction, support, geometry,
span, wall conditions or production default. `characteristic_method="eig"`
remains the default; `"polynomial"` is an opt-in candidate.

The P06 closed-form optimization motivated this candidate. For positive
density/Te/mu and nonnegative Ti/tau, the Q characteristic polynomial factors
into `(lambda-Vi) * Q4(lambda-Ve)`. With `d=Vi-Ve`, `e=Te`, `r=tau*Ti`,
the monic quartic is

```
x^4 - d*x^3
+ (-132723*e*mu + 21300*e - 30000*r*mu)/45000 * x^2
+ d*(169146*e*mu + 30000*r*mu)/45000 * x
+ (-36423*d^2*e*mu + 15123*e^2*mu + 46505*e*r*mu)/45000.
```

The candidate scales speeds, calculates the three stationary points from a
cubic, and bisects the four root intervals in 64 fixed JAX iterations. Real
root brackets, eigen residuals and normalized eigenvector conditioning are
checked. Algebraic eigenvectors and a small device LU inverse construct the
same sign projectors. Invalid cases use the existing Frobenius/Rusanov action
on-device. There is no CPU callback or eigensolve in the candidate. Q's
stopped-projector/live-matrix AD contract is retained; it differs from P06's
Fréchet derivative contract and must not be replaced by P's derivative rule.

Physical-domain restriction and residual checks can reject additional extreme
states. This is not promoted as universally equivalent until those states are
audited. The existing eigensolver remains the baseline, and exact admission
flags plus the existing `1e-8 + 1e-11*abs(expected)` action gate are required
for every prescribed benchmark case.

## Immutable inputs and scope

Reuse the existing remote Q08 RUN:
`/pscratch/sd/y/yiqunx/q08-extraction-global-ff1bff86-20261002T233534Z`.
Its baseline source is `RUN/source/scripts/q08_extraction_global`, identity
`ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722`.
Read only `cpu_N32.json`, `data_N32.json`, the complete `cpu/N32` banks and
`data/N32` owner states/topology. Source/input hashes, chunk receipts, merge
coverage and memory guards are mandatory. No geometry/traces are regenerated.

The five-file source overlay is frozen in `overlay/` with `manifest.json`.
It loads over the unchanged baseline runtime in a fresh Python process, so
concurrent P/production changes are not imported. `freeze.py` is a local
maintainer packaging command, never a remote recovery step.

Run on one node with four actual A100 GPUs. All numerical reference and
candidate actions run on the GPUs; host input loading, validation, boundary
preparation and localization are not timed as GPU RHS work. Use the existing
JAX/JAXLIB 0.9.2 environment. One process, no CPU worker pool. The remote
setup skill selects allocation/CPU affinity/host memory and keeps the old
large campaign paused with no overlapping GPU workload.

The scope is N32, h/32 material + h/16 outer samples, h/16 diffusion, states
1 (smooth), 0 (constant), 21 (held-out short wave), all four D/N patterns.
Full numerical replay has 12 records. Warm timing uses only smooth all-D:
one untimed call plus three synchronized calls for eig/polynomial full RHS,
eig/polynomial split, and polynomial one/four-GPU sharding. Matrix/split
timings use the actual reconstructed centers and actual `b_eta`, not a unit
normal. No profiler trace, full 352-record matrix or N48/N64 global run.

## Commands

From the pinned checkout, with allocation-selected `PYTHON` and `HOST_GIB`:

```bash
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Keep scheduler CUDA_VISIBLE_DEVICES intact. PROFILE is a unique new folder.
mkdir -p "$PROFILE/logs" "$PROFILE/tmp" "$PROFILE/cache"
export TMPDIR="$PROFILE/tmp" XDG_CACHE_HOME="$PROFILE/cache"
RUN=/pscratch/sd/y/yiqunx/q08-extraction-global-ff1bff86-20261002T233534Z
"$PYTHON" -u scripts/q08_polynomial_profile/profile.py run \
  --baseline-source "$RUN/source/scripts/q08_extraction_global" \
  --run "$RUN" --output "$PROFILE/results" --host-gib "$HOST_GIB" \
  > "$PROFILE/logs/run.log" 2>&1
"$PYTHON" scripts/q08_polynomial_profile/profile.py validate \
  --output "$PROFILE/results" > "$PROFILE/logs/validation.json"
```

`results/` must be absent or empty. Never overwrite or append to old timing
records. Use an external 30-minute wall-clock limit; internal checkpoints and
25-minute boundary checks preserve partial results. A failed gate stops the
diagnostic. Return the exact evidence, without numerical changes, tolerance
changes, environment upgrades, CPU offload, extra experiments or restarting
the original campaign. After an operational interruption, preserve partial
results and seek local review before repeating this bounded diagnostic.

Return the single PROFILE folder with actual files: results, compiled text,
source overlay, source/input receipts, environment/allocation logs and an
operational receipt. Existing large immutable banks remain in RUN and are
identified by their receipts. GPU performance is not established by local
CPU tests or forced-host-device sharding controls.
