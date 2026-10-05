# Q09 full-domain evolved MMS

This is the finite-duration prescribed-phi extension of the completed N32
pilot at `50b3a6d80bc33b123cc9d0df6e62d7721025103f`. Use this directory's
controller/manifest; the original pilot manifest remains an immutable record,
not the manifest for the extended harness. The numerical runtime must match
that pilot byte-for-byte. Concurrent production changes are excluded.

- Actual HSX N32/N48/N64 owner domains, all six fields and prescribed phi.
- Diffusion-only and complete six-field parallel RHS; DDDDDD/D, NNNNNN/N,
  DNDNDN/N and NDNDND/D field/phi boundary combinations.
- Common nondimensional end time 0.0001 (ten times the completed pilot). Coarse RK4 counts 100/225/400;
  corresponding coarse dt = 1e-6, 4.4444444444444444e-7, 2.5e-7.
  Each resolution has one fixed timestep, as requested. The conservative N-squared timestep
  scaling is predeclared; it is not a measured stability guarantee.
- 24 cases, 24 integrations, 5,800 accepted steps and 23,200 RK stage calls.
- Every stage checks finite/admissible/positive state. Invalid stages stop.
- Five full-state snapshots per level at 20/40/60/80/100% of end time;
  restart checkpoints every 25 accepted steps, plus every snapshot/final.
- C3, RK4-64 saved traces, unchanged accepted compact28 gradient guard,
  five eta planes; h/32 diffusion/inner material and h/16 outer upwind samples.
- Same independent continuum source and time-varying state, phi and physical
  D/N data at every RK stage. No retracing or numerical source cancellation.
- One first-visible actual A100 executes RK4, all cases sequentially. CPU
  input/reference preparation only. No CPU numerical integration fallback.
  Additional allocated GPUs are idle; this is not multi-GPU evolved validation.

`run_all.py --run RUN --baseline-run OLD --science-run SCIENCE
--canonical-root INPUTS --host-gib HOST_GIB` runs portable tests, verifies all
immutable input receipts, then per-grid bounded CPU/GPU replay and full-grid
RK4, followed by checkpoint validation and deterministic reductions. Run it
from the repository root. It bootstraps only the frozen runtime, establishes
x64/thread limits before JAX, and preserves scheduler CUDA visibility.
`source`, `tests`, `verify`, `preflight --n N`, `run --n N`, and `validate`
are also exposed by `campaign.py` for diagnosed recovery. `freeze` is local
packaging only; never run it remotely.

One parent controller and one numerical subprocess at a time; grids release
memory between processes. The setup worker chooses a usable host budget from
actual resources. Header-based host and measured-free-device admission gates
must pass before dense work. Prefer an A100 with enough free memory for N64;
never switch to CPU or change the numerical configuration to bypass a gate.
All outputs, reference-preparation caches and logs live inside RUN; immutable
inputs remain outside it. Verify enough scratch space for state checkpoints,
source and reference caches. A practical initial reserve is 20 GiB.

Repeat the identical controller command to resume the same RUN. Completed
case receipts are checked; partial levels resume their last accepted saved
state, losing at most 24 unsaved steps. Test/verification/preflight stages
replay on restart. Snapshot files are committed before a checkpoint can move
past them. Don't alter source, time settings, identity, tolerances or inputs.
A scientific rebound/order loss is a result, not an instruction to tune.

`analysis.json` contains global/regional errors, maxima/owners, signed
integrals, five time snapshots and the prescribed timestep/time grid. `orders.csv`
reports spatial orders at the prescribed dt (32–48, 48–64 and 32–64). `report.md`
is machine-generated. One timestep per resolution does not independently measure temporal error
or temporal order. Spatial trends retain that limitation.
All runs use the same compatible manufactured field family (six components),
not every field in the static MMS catalogue. The two operator modes isolate diffusion and then exercise the complete RHS.
Errors are evolved solution minus exact MMS, not static N-O-R quantities.
Small total-field relative errors must also be compared with exact temporal
change. RK integral bookkeeping is not exact spatial conservation.

`completion.json` binds all consumed solution/reference-observation files and
reports operational completion only. Scientific qualification remains local.
No long-time/perturbation stability, reconstructed phi, physical live sheath,
exterior wall crossings, perpendicular coupling, new hot-ion model, multiple
manufactured families or production promotion is established by this campaign.
