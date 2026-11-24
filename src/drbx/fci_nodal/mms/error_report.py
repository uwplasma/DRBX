"""Common-Hp manufactured-solution error reductions by owner."""

from contextlib import closing

import jax
import jax.numpy as jnp
import numpy as np

from .evolved import PERIOD, scaled

FIELDS = ("n", "Te", "Ti", "Vi", "w", "Omega", "Ve", "j", "phi", "A")
REGIONS = ("wall_touch", "other", "all")
METRIC = "absolute common-Hp RMS error and maximum node error against the exact field"


def owner_masks(nodes, wall, q, q_products, indexed_geometry):
    """Find wall-touching columns for each addressable owner."""
    from ..operator_build.parallel.indexed_input import IndexedInput

    masks = {}
    with closing(IndexedInput(indexed_geometry)) as indexed:
        for owner, ids in nodes.ids_by_owner.items():
            begin, end = q.edges[owner:owner+2]
            start, stop = begin*q.width, end*q.width
            touch = np.zeros(len(ids), dtype=bool)
            _, hits, _ = q.hits(owner, q_products, indexed)
            launch = hits[:, 0].astype(np.int64)
            if np.any((launch < start) | (launch >= stop)):
                raise ValueError("Q wall launch nodes differ from the owner layout")
            touch[launch-start] = True
            # The trace is local. Conormal wall rows reach two periodic eta planes.
            planes = np.arange(begin-2, end+2) % q.edges[-1]
            neighbors = np.unique(np.searchsorted(q.edges[1:], planes, side="right"))
            for source in neighbors:
                path = wall / f"owner{source:04d}" / "wall_plan.npz"
                with np.load(path, allow_pickle=False) as archive:
                    for matrix in (("T", "F") if source == owner else ("F",)):
                        columns = archive[matrix+"_indices"][archive[matrix+"_data"] != 0]
                        keep = (columns >= start) & (columns < stop)
                        touch[columns[keep]-start] = True
            masks[owner] = (ids >= 0, touch)
    return masks


def owner_partials(weights, active, touch, delta=None):
    """Reduce one owner's active rows into JSON-compatible region statistics."""
    partials = {}
    masks = (active & touch, active & ~touch, active)
    for region, mask in zip(REGIONS, masks, strict=True):
        h = weights[mask]
        squared = maximum = None
        if delta is not None and h.size:
            error = delta[mask]
            squared = np.sum(h[:, None]*error**2, axis=0).tolist()
            maximum = np.max(np.abs(error), axis=0).tolist()
        partials[region] = (int(h.size), float(h.sum()), squared, maximum)
    return partials


def _shards(array, mesh):
    by_device = {shard.device: shard for shard in array.addressable_shards}
    return {owner: np.asarray(by_device[device].data)
            for owner, device in enumerate(mesh.devices.flat) if device in by_device}


def owner_weights(resolution):
    return _shards(resolution.plan.wall_ops.H, resolution.q.mesh)


@jax.jit
def _exact_primitives(data, time):
    amplitude = data.get("amplitude", 1.)
    phase = 2*jnp.pi*time/PERIOD
    a = amplitude*(1+.2*jnp.sin(phase))
    ap = amplitude*.2*2*jnp.pi/PERIOD*jnp.cos(phase)
    exact = scaled(data["nodal"], a, ap,
                   nonlinear_phi="uses_nonlinear_manufactured_potential" in data,
                   grounded_phi="uses_grounded_wall_potential" in data).value
    return exact, a*data["A"].value


def target_partials(resolution, weights, masks, state, stage, data, time, aux):
    mesh = resolution.q.mesh
    if len(mesh.local_devices) != mesh.size:
        from ..drb.owner_route_store import replicate
        time = replicate(mesh, time)
    exact, exact_a = _exact_primitives(data, time)
    u, canonical, v, a_exact, j, psi, a_numeric = (
        _shards(array, mesh) for array in
        (state, stage.exact, exact, exact_a, aux["current"], aux["psi"], aux["A"]))
    tau = resolution.plan.params.tau
    tau = float(tau if tau.is_fully_addressable else tau.addressable_data(0))
    partials = []
    for owner, h in weights.items():
        active, touch = masks[owner]
        numeric, primitive = u[owner], v[owner]
        ve = numeric[:, 3]-j[owner]/numeric[:, 0]
        phi = psi[owner]-tau*numeric[:, 0]*numeric[:, 2]
        exact_j = primitive[:, 0]*(primitive[:, 3]-primitive[:, 4])
        delta = np.column_stack((numeric-canonical[owner], ve-primitive[:, 4],
                                 j[owner]-exact_j, phi-primitive[:, 6], a_numeric[owner]-a_exact[owner]))
        partials.append(owner_partials(h, active, touch, delta))
    return partials


def error_rows(partials):
    rows = []
    for region in REGIONS:
        entries = [partial[region] for partial in partials if partial[region][0]]
        mass = sum(entry[1] for entry in entries)
        rms = np.sqrt(np.sum([entry[2] for entry in entries], axis=0)/mass).tolist() if mass else [None]*len(FIELDS)
        maximum = np.max([entry[3] for entry in entries], axis=0).tolist() if entries else [None]*len(FIELDS)
        for field, average, peak in zip(FIELDS, rms, maximum, strict=True):
            rows.append(dict(field=field, region=region, absolute_Hp_RMS=average, max_node_error=peak))
    return rows


def merge_reports(reports, *, dt, steps):
    geometry = [partial for report in reports for partial in report["geometry"]]
    regions = {region: sum(partial[region][0] for partial in geometry) for region in REGIONS}
    wall_mass = sum(partial["wall_touch"][1] for partial in geometry)
    total_mass = sum(partial["all"][1] for partial in geometry)
    targets = []
    for group in zip(*(report["targets"] for report in reports), strict=True):
        time, step = group[0]["time"], group[0]["steps"]
        if any((entry["time"], entry["steps"]) != (time, step) for entry in group):
            raise ValueError("Manufactured error target differs across owners")
        rows = error_rows([partial for entry in group for partial in entry["partials"]])
        targets.append(dict(time=time, steps=step, rows=rows))
    seconds = [max(durations) for durations in zip(*(report["seconds"] for report in reports), strict=True)]
    if len(seconds) != steps:
        raise ValueError("Manufactured error report timing count differs from the run")
    return dict(metric=METRIC, fields=FIELDS, regions=regions, targets=targets, dt=dt, steps=steps,
                wall_touch_Hp_share=wall_mass/total_mass if total_mass else None,
                first_step_seconds=seconds[0] if seconds else None,
                mean_step_seconds_after_first=sum(seconds[1:])/(steps-1) if steps > 1 else None)
