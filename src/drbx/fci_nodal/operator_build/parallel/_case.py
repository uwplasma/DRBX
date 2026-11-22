"""File identities for the four final repaired polygon and merged wall rows products."""

import json
from pathlib import Path

from ..._hashing import sha256_file as file_sha256
from ._io import dump

CASE_NAME = 'polygon_case.json'
CASE_SCHEMA = 'drbx.geometry.polygon_case.v3'
PRODUCT_ROLES = ('geometry', 'merged', 'face_flux', 'wall_rows')

REACH_CLASSES = ('both_planes_reached', 'forward_wall_hit', 'backward_wall_hit', 'neither_plane_reached')
REACH_OUTPUT_STEMS = {
    'both_planes_reached': ('class_both_planes_reached', 'joint_both_planes_reached'),
    'forward_wall_hit': ('class_forward_wall_hit', 'joint_forward_wall_hit'),
    'backward_wall_hit': ('class_backward_wall_hit', 'joint_backward_wall_hit'),
    'neither_plane_reached': ('class_neither_plane_reached', 'joint_neither_plane_reached'),
}
MIDPOINT_WALL_CLASSES = {'forward_wall_hit': 'midpoint_forward_wall_hit', 'backward_wall_hit': 'midpoint_backward_wall_hit'}
PLANE_WALL_STEM = 'plane_seeded_wall_rows'


def case_inputs(options, field_table_sha256, metric_cache_sha256):
    return dict(schema=CASE_SCHEMA, N=options['resolution'], rings=options['rings'],
                field_periods=options['field_periods'], symmetry=options['symmetry'],
                field_table_sha256=field_table_sha256, metric_cache_sha256=metric_cache_sha256,
                route_options=options)


def open_case_record(root, inputs):
    """Claim a fresh root or check its original inputs before using any checkpoints."""
    root = Path(root)
    receipt = root/'polygon_repair'/'arrays'/CASE_NAME
    if receipt.is_file():
        record = json.loads(receipt.read_text())
        for key, value in inputs.items():
            if record.get(key) != value:
                raise ValueError(f'repaired polygon and merged wall rows root inputs differ for {key}: {receipt}')
        if record['products']:
            files = {role:receipt.parent/item['file'] for role,item in record['products'].items()}
            if record['products'] != _product_entries(receipt.parent, files):
                raise ValueError(f'repaired polygon and merged wall rows completed products changed: {receipt}')
        return receipt
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'repaired polygon and merged wall rows root contains outputs without an input record: {root}')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as target:
        json.dump(dict(inputs, products={}, backfilled=False), target, indent=2, allow_nan=False)
        target.write('\n')
    return receipt


def _product_entries(directory, products):
    directory = directory.resolve()
    if set(products) != set(PRODUCT_ROLES):
        raise ValueError('repaired polygon and merged wall rows case record requires all four final products')
    entries = {}
    for role in PRODUCT_ROLES:
        path = Path(products[role]).resolve()
        if path.parent != directory:
            raise ValueError(f'repaired polygon and merged wall rows {role} must be beside its case record')
        entries[role] = dict(file=path.name, sha256=file_sha256(path))
    return entries


def finish_case_record(receipt, inputs, products):
    record = json.loads(receipt.read_text())
    if any(record.get(key) != value for key,value in inputs.items()):
        raise ValueError(f'repaired polygon and merged wall rows root inputs changed before finalization: {receipt}')
    entries = _product_entries(receipt.parent, products)
    if record['products'] and record['products'] != entries:
        raise ValueError(f'repaired polygon and merged wall rows completed products changed: {receipt}')
    dump(receipt, dict(record, products=entries))
    return receipt


def write_case_record(directory, options, field_table_sha256, products, *,
                      metric_cache_sha256, backfilled=False):
    """Write a completed case record from separately verified legacy inputs."""
    directory = Path(directory).resolve()
    receipt = directory / CASE_NAME
    inputs = case_inputs(options, field_table_sha256, metric_cache_sha256)
    dump(receipt, dict(inputs, products=_product_entries(directory, products), backfilled=backfilled))
    return receipt
