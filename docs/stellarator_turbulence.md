# Stellarator FCI turbulence status

The earlier rotating-ellipse and island-divertor turbulence drivers are
historical context only; their reproduction scripts and validation gates were
removed as part of the sharded-only FCI refactor. The associated media remain
useful for visual context.

The supported non-axisymmetric path uses `FciGeometry3D` for host-side
geometry staging, then lowers local metrics, boundary data, and halo payloads
to a `jax.shard_map` kernel. Current runnable validation is provided by the
sharded two-field example and the shifted-torus full-EB MMS test.
