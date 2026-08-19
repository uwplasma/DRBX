# Axis-Regular Prototype Experiments

This document is the canonical log for axis-regular and angular-RLP
production prototypes. Each entry records the architecture under test, the
problem it addresses, and the observed gate status.

## Eta-only RLP sharding implementation

**Architecture.** The production driver decomposes the logical domain only
along eta, using `--shard-counts 1 1 Seta` for every topology. Toroidal angular
owner aggregates are confined to one eta plane, so owner prolongation and
physical-volume restriction remain local. Global owner-space means, norm,
compatibility, and GMRES reductions still span the eta shards. The line-u
preconditioner retains local radial trees and includes both eta-face diagonal
contributions at local slab interfaces. x/theta RLP decomposition and
fallback paths are intentionally unsupported.

**Problem addressed.** Enable production RLP on multiple devices without
splitting an angular owner aggregate across devices, while keeping the
fine-grid axis-regular operator and the owner-space phi solve mathematically
consistent with the global problem.

**Validation plan.** First compare single-shard and eta-sharded geometry,
operator, owner-space phi, and RHS results; then run the full 32^3 simulation
with an eta-sharded production configuration.

**Final 32^3 gate status: passed.** A two-device CPU run with
`--shard-counts 1 1 2`, RK4, 225 steps, and `t_f=0.15` completed the full time
range. The four phi solves per step averaged 62.30 GMRES iterations; the final
step averaged 59 iterations with maximum relative residual `9.645e-9`. No
solve reached the 500-iteration cap, and no positivity or finite-value gate
failed. The final materialized field ranges were density
`[0.9920619, 1.0127181]`, `Te=[0.9920726, 1.0047989]`,
`Ti=[0.9958022, 1.0031481]`, `Vi=[-0.0036129, 0.0046220]`,
`Ve=[-0.3370424, 0.3163146]`, phi `[-0.0030767, 0.0051019]`, and
vorticity `[-0.0882613, 0.0395847]`. The saved history is
`prototype_runs/rlp_eta_sharding_32_gate/hsx_rlp_eta_sharding_32_gate.npz`.
