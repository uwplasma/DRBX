"""repaired polygon and merged wall rows row streams and the optional reference-geometry ID crosswalk keyed by source."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from ..options import boundary_bisection_count
from ..operator_build.parallel._case import CASE_SCHEMA, PRODUCT_ROLES
from .storage import arrays, load
from .._hashing import sha256_file as sha


def check_match(name, recorded, requested):
    if recorded != requested:
        raise ValueError(f'repaired polygon and merged wall rows {name}: recorded {recorded!r}, requested {requested!r}')


def validate_polygon_case(config, receipt):
    """Bind repaired polygon and merged wall rows case metadata and verify the two streams before splicing."""
    if receipt is None or not Path(receipt).is_file():
        raise ValueError(f'Production repaired polygon and merged wall rows requires a polygon case record, got {receipt!r}')
    receipt = Path(receipt)
    result = load(receipt)
    check_match('case record schema', result.get('schema'), CASE_SCHEMA)
    settings = result.get('route_options', {})
    for field, option, requested in (('N', 'resolution', config['N']),
                                     ('rings', 'rings', config['polygon_rings']),
                                     ('field_periods', 'field_periods', config['field_periods']),
                                     ('symmetry', 'symmetry', config['symmetry'])):
        check_match(field, result.get(field), requested)
        check_match(f'route option {option}', settings.get(option), requested)
    check_match('route filter spec', settings.get('field_filter'), config['filter_spec'].as_dict())
    check_match('boundary bisections', boundary_bisection_count(config['N'], settings.get('boundary_bisections')),
                config['clip_bisections'])
    check_match('metric cache SHA-256', result.get('metric_cache_sha256'), sha(config['files']['metric_cache']))
    products = result.get('products', {})
    check_match('product roles', sorted(products), sorted(PRODUCT_ROLES))
    for role, stage in (('geometry', 'geometry'), ('merged', 'merge')):
        path = config['files'][f'polygon_{role}'].resolve()
        product = products[role]
        name = product.get('file')
        recorded = (receipt.parent / name).resolve() if name and Path(name).name == name else None
        check_match(f'{role} product', recorded, path)
        check_match(f'{role} SHA-256', product.get('sha256'), sha(path))
        data = load(path.parent.parent/f"{stage}_N{config['N']}.json")
        check_match(f'{stage} resolution', data.get('N'), config['N'])
    check_match('field table SHA-256', result.get('field_table_sha256'), sha(config['files']['field_table']))


KEY_WIDTH = {'face': 7, 'wall': 7}
ARCHIVED_CROSSWALK_MEMBERS = {
    'face': ('face_source_key', 'face_c14_reference_id'),
    'wall': ('wall_source_key', 'wall_c14_reference_id'),
}


def source_band(xy, triangles, resolution, rings, angles):
    # Same source disk and radial predicate as polysplice.source_census.
    band = np.min(np.linalg.norm(xy[triangles], axis=2), axis=1) >= (resolution - rings) / resolution - 1e-12
    if np.count_nonzero(band) != 2 * rings * angles:
        raise ValueError('Production source disk does not match the polygon band')
    return band


def key_records(keys):
    rows = np.ascontiguousarray(keys, dtype=np.int64)
    return rows.view(np.dtype([('', np.int64)] * rows.shape[1])).ravel()


class ReferenceKeys:
    """Read source keys and reference IDs from the archived NPZ member table."""

    def __init__(self, path):
        self.tables = {}
        with np.load(path, allow_pickle=False) as source:
            for kind, width in KEY_WIDTH.items():
                key_name, id_name = ARCHIVED_CROSSWALK_MEMBERS[kind]
                if key_name not in source and id_name not in source:
                    continue
                keys, ids = source[key_name], source[id_name]
                if (keys.ndim != 2 or keys.shape[1] != width or ids.shape != (len(keys),)
                        or keys.dtype.kind != 'i' or ids.dtype.kind != 'i' or np.any(ids < 0)):
                    raise ValueError(f'Invalid {kind} reference geometry table')
                records = key_records(keys)
                order = np.argsort(records)
                records = records[order]
                if np.any(records[1:] == records[:-1]) or len(np.unique(ids)) != len(ids):
                    raise ValueError(f'Duplicate {kind} reference geometry key or ID')
                self.tables[kind] = records, ids[order]

    def match(self, kind, keys):
        result = np.full(len(keys), -1, np.int64)
        if kind in self.tables:
            records, ids = self.tables[kind]
            query = key_records(keys)
            positions = np.searchsorted(records, query)
            available = np.flatnonzero(positions < len(records))
            matched = available[records[positions[available]] == query[available]]
            result[matched] = ids[positions[matched]]
        return result


def stream_key(rows):
    # Parent ordinals bind only to the original stream that defines them.
    values = np.ascontiguousarray(rows)
    h = sha256(repr((values.dtype.str, values.shape)).encode())
    h.update(values.view(np.uint8).ravel())
    return np.frombuffer(h.digest(), dtype='<i8')


def polygon_keys(parent, identity):
    return np.column_stack((np.zeros(len(parent), np.int64), parent,
                            np.broadcast_to(identity, (len(parent), len(identity))),
                            np.full(len(parent), -1, np.int64)))


def retained_keys(actual, quad_pair):
    # The quadrature rows of a triangle stay in one source leaf. Keep the
    # original order among repeated (triangle, subdivision, left, right) keys.
    records = key_records(quad_pair)
    order = np.argsort(records, kind='stable')
    ranks = np.arange(len(quad_pair), dtype=np.int64)
    start = np.r_[True, records[order[1:]] != records[order[:-1]]][:len(ranks)]
    ordinal = np.empty_like(ranks)
    ordinal[order] = ranks - np.maximum.accumulate(np.where(start, ranks, 0))
    return np.column_stack((np.ones(len(ranks), np.int64), np.full(len(ranks), actual), quad_pair, ordinal))


class ProductionSplice:
    def __init__(self, files, physical_nodes, reference_keys):
        self.faces = arrays(files['polygon_geometry'], ['faces'])['faces']
        self.wall = arrays(files['polygon_merged'], ['wall'])
        wall = self.wall['wall']
        if self.faces.ndim != 2 or self.faces.shape[1] != 11 or wall.ndim != 2 or wall.shape[1] != 10:
            raise ValueError('Expected repaired polygon and merged wall rows face and merged wall columns')
        if np.any(~((wall[:, 0] >= 0) & (wall[:, 0] < physical_nodes))):
            raise ValueError('The production repaired polygon and merged wall rows merged stream must contain physical walls')
        self.face_stream = stream_key(self.faces)
        self.wall_stream = stream_key(wall)
        self.reference = ReferenceKeys(reference_keys) if reference_keys is not None else None

    def metadata(self, kind, dense, keys):
        values = dict(mode_dense_ordinal=dense, source_key=keys)
        if self.reference is not None:
            values['reference_geometry_id'] = self.reference.match(kind, keys)
        return values
