"""Extract the two metrics and assemble the disk-backed perpendicular nodal bundle."""

from __future__ import annotations

from ..._hashing import sha256_file
from pathlib import Path


from ...operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout
from ...operators.nodal.nodal_shape import ZERNIKE_CORE_RING_KIND

from .options import PerpendicularBuildOptions
from .owner_bundle import _plane_edges


def build_nodal_bundle(options: PerpendicularBuildOptions, *, owner_parts: int,
                       shared_table: Path | None = None) -> dict[str, Path]:
    """Produce the reference and SBP leaves directly by owner."""
    if options.layout_kind != ZERNIKE_CORE_RING_KIND:
        raise ValueError("the P nodal build supports layout Zernike core-ring layout")
    if options.symmetry not in ("nfp", "none"):
        raise ValueError("symmetry must be nfp or none")
    if options.field_periods < 1 or options.n_eta < 1 or options.chunk < 1:
        raise ValueError("field periods, eta planes and extraction chunk must be positive")
    _plane_edges(options.n_eta, owner_parts)
    map_periods, covered_periods = options.period_counts
    paths = {name: Path(path).resolve() for name, path in
             (("metric_cache", options.metric_cache), ("makegrid", options.makegrid))}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    inputs = {name: dict(path=str(path), size=path.stat().st_size, sha256=sha256_file(path))
              for name, path in paths.items()}
    inputs.update(makegrid_currents=[float(x) for x in options.makegrid_currents],
                  field_periods=options.field_periods, symmetry=options.symmetry,
                  table_grid=list(options.table_grid) if options.eta_filter is not None else None)

    if options.map_periods is not None or options.covered_periods is not None:
        inputs.update(map_periods=map_periods, covered_periods=covered_periods)

    n = options.resolution
    layout = build_zernike_core_ring_layout(n, n_eta=options.n_eta)
    out = Path(options.output_directory).resolve()
    from .build_by_owner import _build_owners

    return _build_owners(options, layout, out, inputs, owner_parts, shared_table=shared_table)
