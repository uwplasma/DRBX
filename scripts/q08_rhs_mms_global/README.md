# Q08 global static six-field RHS MMS

This is the scientific follow-up to the completed Q08 full-domain extraction
and one/four-A100 implementation replay. It is **not another implementation
replay or performance sweep**. It does not change a solver or production default.

## Frozen scope

N32/N48/N64, all 22 existing smooth/wave/held-out states, four uniform/mixed
Dirichlet and physical-normal Neumann patterns, all complete owners and members.
Material inner span h/32, outer upwind samples h/16; diffusion h/16 and h/32.
The saved compact-C3 geometry, RK4-64 traces, gradient repair, donor support and
polynomial characteristic implementation are unchanged. There is no tracing,
reconstruction preparation, degree/support/span change or reference quadrature.

The selected differential block has six fields n, Te, Ti, Vi, Ve, omega, with
prescribed phi; characteristic material transport, scalar vorticity transport,
primitive-slot product current/phi drive, and six constant diffusion channels.
Coefficients are `.01*[1,2,3,4,5,6]`, tau=1, mu=1836, and the accepted omega
combination is retained. No collision/external-source physics, perpendicular
operators, polarization solve, physical sheath SAT, exterior wall crossing or
time integration is added.

## Scientific definitions

- **N**: the actual extracted six-field RHS on four A100 GPUs, using the
  qualified polynomial characteristic kernel. No candidate work moves to CPU.
- **O**: the same discrete formulas supplied with exact manufactured values
  at saved stencil points and exact gradients for diffusion. Its independent
  characteristic oracle retains the CPU eig implementation already replayed
  against the GPU polynomial implementation. O has no reconstruction error.
- **R**: analytic first field derivatives at raw centers, retaining div(b).
  Diffusion uses the previous independent fourth-order coordinate difference
  of J b (b.grad f), with reference steps 1e-4, 5e-5, 2.5e-5. The first is the
  fixed target; maximum per-term sensitivity to the other two is saved for
  every chunk and reduced globally. There is no assumption div(B)=0.

Each action is projected to complete owners with the same physical volume
weights. R is a weighted collection of raw-center targets, not a new integrated
cell target. O/R are computed once and reused for all BC representations.
Both representations are compatible with the same exact manufactured fields.
The steady manufactured source S=-R is added inside the same assembled GPU
stage. Its residual must reproduce N-R; this is not evolved MMS evidence.

All 31 outputs are retained separately: centered/correction/diffusion/combined
for six equations plus current, omega advection, omega current, phi force,
electron material, Ti compensation and generalized force. Report global and
nine overlapping regional physical-volume weighted RMS, relative RMS, maxima
with owner identities, signed integrals and both refinement orders for N-O,
O-R, N-R. Constants, smooth fields and short-wave stress controls stay distinct.
Scientific orders are results, not execution gates; do not tune on a rebound.

## Required immutable inputs

1. Completed original CPU bank run with identity
   `ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722`,
   its frozen source `source/scripts/q08_extraction_global`, inputs, eight
   canonical files, all CPU banks, owner state/phi/topology arrays and receipts.
2. Completed implementation run with manifest identity
   `1a953c60d7352b2a74e6a737d7c2be7dcca9ee194808db3154e5f840d4a9b087`.
   Its completion receipt and every referenced file are verified, not rerun.
3. This directory and sibling `q08_polynomial_global` from the pinned commit.
   The latter supplies the unchanged five-module runtime overlay and verified
   boundary cache/sharding helpers. The baseline supplies the frozen runtime.

Missing/corrupt input stops the assignment. Neither old output folder is edited.
Do not regenerate a missing bank, trace or reference to bypass verification.

## Execution and resources

Use one full four-A100 node, actual CUDA JAX x64, and the previously successful
JAX/JAXLIB 0.9.2 CUDA environment. One controller sequences CPU independent
reference stages and one four-GPU candidate process; no competing GPU workers.
CPU references use a spawn pool, single-threaded BLAS/JAX workers, at most the
effective affinity count. Choose `WORKERS` from allocation CPU/memory limits.
Reference resource contract is 4 GiB per worker plus 8 GiB parent allowance;
exceeding the observed worker limit fails without a completed receipt.

The full-grid GPU merge applies a new host guard of three times the old estimate
plus 8 GiB for scientific scratch, accounting for the previous underestimate.
Pass the allocation's conservative usable budget as HOST_GIB. Reference and GPU
stages run sequentially. Keep at least 12 GiB free disk. Existing inputs remain
external; generated outputs, caches, logs and provenance stay under one new RUN.
Reference chunks are checkpointed; O/R merge scratch is removed on GPU completion.
Expected returned scientific records are a few GiB, not a guarantee.

```bash
python run_all.py --run "$RUN" --baseline-run "$OLD" \
  --baseline-source "$OLD/source/scripts/q08_extraction_global" \
  --implementation-run "$IMPL" --workers "$WORKERS" --host-gib "$HOST_GIB" \
  --through pilot
python run_all.py --run "$RUN" --baseline-run "$OLD" \
  --baseline-source "$OLD/source/scripts/q08_extraction_global" \
  --implementation-run "$IMPL" --workers "$WORKERS" --host-gib "$HOST_GIB" \
  --through complete
```

The controller runs portable tests, input verification, persisted 21-owner
O/R replay, actual-GPU N replay (all 22 states, BCs and spans), and a stratified
three-chunk reference pilot. Use measured pilot cost to schedule the main run;
do not substitute the preceding verification campaign's runtime as an estimate.
The pilot's completed reference chunks are reused. The main run executes each
resolution's remaining references, one four-GPU action per state/BC/span,
deterministic postprocessing and independent completion reduction. There are
no seven-repeat timing loops, new literal-CPU N sweeps or one-GPU scaling runs.

Each stage can also be invoked with `campaign.py STAGE` and the same four path
arguments; `references` additionally requires n/workers/host-gib, and `gpu`
requires n/host-gib. See `--help`. A single controller/writer lock prevents
duplicates. Resume the same RUN and pinned source after confirming prior
processes ended; valid chunks/cases are reused with content hashes. Never
relabel old results. Replay limit 1e-8+1e-11*abs(expected), constant N-O limit
1e-7; no remote relaxation, repair, method change or promotion.

### Pilot import recovery

The initial publication at `19136040` passed portable tests, input verification
and CPU/GPU bounded replay, but its pilot stopped before computing a reference
chunk. Loading the polynomial overlay placed its unrelated `campaign.py` ahead
of this controller on the module search path. Tests had preloaded the MMS
controller under that bare name, masking the command-line failure.

MMS modules now use the explicit `scripts.q08_rhs_mms_global` package namespace.
The portable suite includes a cold-interpreter and spawn-worker regression with
the conflicting older controller deliberately present. This repair changes
module routing only; scientific formulas, inputs and all gates are unchanged.
Its manifest identity is new. Preserve the failed RUN and start a separate RUN
with this pinned source, reusing the same immutable baseline and implementation
inputs. Re-execute the prescribed gates; do not copy or relabel old preflight
receipts. There are no scientific chunks from that failed pilot to migrate.

## Return

`completion.json`, `analysis.json`, `totals.npz`, `orders.csv`, `report.md`,
all reference chunks/receipts, scientific reduction records/compiler proofs,
preflight/pilot/grid receipts, frozen source, provenance, logs and operational
receipt. `totals.npz` includes every norm/order/max/integral array; analysis.json
documents its axes and the reference sensitivity. An execution pass is not an
automatic scientific pass. Return any scientific exceptions unchanged for
local interpretation. Do not close full Q08 or promote production.
