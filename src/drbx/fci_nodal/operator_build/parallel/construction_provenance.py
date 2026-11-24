"""Layout and canonical moment-reduced face quadrature leaf identity for owner-only Q construction."""

import hashlib
import json

import numpy as np

from .indexed_input import ROWS
from ..._hashing import sha256_file as sha
from .job_storage import read


def indexed_layout(geometry):
    n, planes, width, nodes = (int(geometry.info[k]) for k in ('N', 'E', 'P', 'NN'))
    source = json.loads((geometry.handoff.parent/'geometry.json').read_text())
    symmetry, periods = source['symmetry'], source['field_periods']
    period_planes = planes // periods if symmetry == 'nfp' else planes
    return dict(N=n, eta_planes=planes, symmetry=symmetry, field_periods=periods,
                plane_nodes=width, physical_nodes=nodes,
                period_nodes=period_planes*width)


def owner_plane_paths(runtime, manifest):
    """Resolve the ordered plane paths within the construction root."""
    paths = []
    for owner, item in enumerate(manifest['owners']):
        if item['owner'] != owner or item['begin'] != len(paths) or item['begin'] >= item['end']:
            raise ValueError('Noncanonical Q construction owner layout')
        directory = (runtime/item['directory']).resolve()
        if not directory.is_relative_to(runtime):
            raise ValueError('Q owner construction directory escaped its root')
        paths.extend(directory/f'plane_{p:05d}' for p in range(item['begin'], item['end']))
    return paths


def face_quadrature_provenance(planes, geometry):
    """Hash typed leaves after bounded ordering and geometric coverage checks."""
    width, physical = (int(geometry.info[k]) for k in ('P', 'NN'))
    face_count = geometry.shape('faces')[0]
    reg = []
    for plane, folder in enumerate(planes):
        face_file, weight_file = folder/'moment_reduced_face_quadrature.npy', folder/'mu.npy'
        ids, weights = read(face_file), read(weight_file)
        if ids.ndim != 1 or ids.dtype != np.int64 or weights.shape != ids.shape or weights.dtype != np.float64:
            raise ValueError('Invalid owner moment-reduced face quadrature recipe leaves')
        last = -1
        for lo in range(0, len(ids), ROWS):
            block = ids[lo:lo+ROWS]
            mu = weights[lo:lo+ROWS]
            if block[0] <= last or block[-1] >= face_count or np.any(block[1:] <= block[:-1]):
                raise ValueError('Invalid canonical moment-reduced face quadrature face order or bounds')
            if np.any(~np.isfinite(mu) | (mu <= 0)):
                raise ValueError('Invalid canonical moment-reduced face quadrature weight')
            nodes = geometry.take('faces', block)[:, :2]
            # Strict ordering on each leaf and disjoint left planes rule out duplicates globally.
            if np.any((nodes < 0) | (nodes >= physical)) or np.any(nodes[:, 0]//width != plane):
                raise ValueError('moment-reduced face quadrature face does not belong to its physical owner plane')
            last = int(block[-1])
        reg.append(dict(plane=plane, count=len(ids), indices_sha256=sha(face_file), mu_sha256=sha(weight_file)))
    count = sum(leaf['count'] for leaf in reg)
    if count >= 2**31:
        raise ValueError('moment-reduced face quadrature recipe exceeds signed int32 capacity')
    recipe = dict(format='drbx.parallel.moment_reduced_face_quadrature_leaves.v1', degree=2, count=count,
                  index_dtype='int64', weight_dtype='float64', leaves=reg)
    encoded = json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return dict(recipe, sha256=hashlib.sha256(encoded).hexdigest())


def verify_parallel_construction(manifest, geometry, planes):
    """Verify the owner-only layout, indexed binding and moment-reduced face quadrature identity."""
    layout = indexed_layout(geometry)
    E, P = layout['eta_planes'], layout['plane_nodes']
    parts = len(manifest['owners'])
    edges = [plane*P for plane in range(E+1)]
    if not parts or E//parts < 4:
        raise ValueError('Owner-only Q construction requires four planes per owner')
    cuts = [(owner*E+parts-1)//parts for owner in range(parts+1)]
    if (manifest['format'] != 'drbx.parallel.owner_construction.v1' or manifest['index_binding'] != geometry.identity()
            or manifest['layout'] != layout or manifest['plane_edges'] != edges or len(planes) != E
            or [(item['begin'], item['end']) for item in manifest['owners']] != list(zip(cuts, cuts[1:]))):
        raise ValueError('Owner-only Q layout or index binding differs from the indexed geometry')
    recipe = face_quadrature_provenance(planes, geometry)
    if recipe != manifest['moment_reduced_face_quadrature']:
        raise ValueError('Canonical moment-reduced face quadrature leaves differ from the Q construction manifest')
    return recipe
