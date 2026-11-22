"""Opt-in one-owner rebuild against a case produced by the same backend."""

import json
from pathlib import Path
import re

import numpy as np
import pytest

from tests.fci_nodal.support.backend import backend_module
from tests.fci_nodal.support.cold_case import (
    cold_options, current_case_directory, mms_profile, read_stage, stage_products,
)
from tests.fci_nodal.support.identity import array_difference, compare_array_files, compare_indexed, require_bytes

pytestmark = pytest.mark.rebuild_case

FIELD_TABLE_METADATA = 'field_table/field_table/meta'
NODAL_FILTER_METADATA = 'owner0000/nodal_meta_eta_filter.npy'
NODAL_FILTER_ROW = f'perpendicular_operators/owner_perpendicular_bundle/{NODAL_FILTER_METADATA}'


def without_seconds(value):
    if isinstance(value, dict):
        return {key: without_seconds(child) for key, child in value.items() if key != 'seconds'}
    if isinstance(value, list):
        return [without_seconds(child) for child in value]
    return value


def polygon_history_key(relative):
    if relative.parent.as_posix() == 'history':
        match = re.fullmatch(r'(base_polygon_pre_refine_geometry_[0-9]+)_[0-9]{16,}\.npz',
                             relative.name)
        if match:
            return relative.with_name(match[1]+'.npz')
    return relative


def compare_case_products(case, reference):
    differences = {}

    def record(name, actual, expected):
        if name == 'metric/metric_cache/metric_evaluator_exact_plane_eta_diagnostics_json':
            left, right = (without_seconds(json.loads(value.item())) for value in (actual, expected))
            row = dict(bytes_equal=left == right, max_abs=None, max_abs_reference=None)
        elif name in (FIELD_TABLE_METADATA, NODAL_FILTER_ROW):
            left, right = (json.loads(value.item()) for value in (actual, expected))
            left.pop('arm_sha256', None)
            right.pop('arm_sha256', None)
            row = dict(bytes_equal=left == right, max_abs=None, max_abs_reference=None)
        else:
            row = array_difference(actual, expected)
        differences[name] = row

    actual, expected = (stage_products(root, 'geometry')['indexed_geometry']
                        for root in (case, reference))
    compare_indexed(actual, expected, record)

    for stage, role in (
        ('metric', 'metric_cache'),
        ('field_table', 'field_table'),
        ('polygon', 'polygon_route'),
        ('parallel_operators', 'parallel_construction'),
        ('perpendicular_operators', 'owner_perpendicular_bundle'),
        ('owner_inputs', 'owner_parallel'),
        ('wall_data', 'wall_data'),
        ('mms_prepare', 'prepared'),
        ('mms_run', 'owner_state'),
    ):
        actual, expected = (stage_products(root, stage)[role] for root in (case, reference))
        path_key = polygon_history_key if stage == 'polygon' else None
        compare_array_files(actual, expected, record, prefix=f'{stage}/{role}/', path_key=path_key)

    for stage, role in (('owner_inputs', 'manufactured_ion_flow_fit'), ('mms_run', 'summary')):
        actual, expected = (stage_products(root, stage)[role] for root in (case, reference))
        record(f'{stage}/{role}', actual.read_bytes(), expected.read_bytes())

    return differences


def require_case_arm_identity(case):
    receipts = backend_module('case._receipts')
    sha256_file = backend_module('_hashing').sha256_file
    arm_identity = backend_module('operator_build.perp.eta_filter').arm_identity
    metric_receipt = read_stage(case, 'metric')
    binding = receipts.binding(metric_receipt)
    makegrid = Path(binding['options']['makegrid_path']).resolve()
    metric_cache = receipts.receipt_products(metric_receipt)['metric_cache']
    provenance = {
        'metric_cache': {'sha256': sha256_file(metric_cache)},
        'makegrid': {'sha256': binding['input_files'][str(makegrid)]},
    }
    expected = arm_identity(provenance, cold_options(case).perpendicular)
    table = stage_products(case, 'field_table')['field_table']
    perpendicular = stage_products(case, 'perpendicular_operators')['owner_perpendicular_bundle']
    with np.load(table, allow_pickle=False) as saved:
        table_hash = json.loads(saved['meta'].item()).get('arm_sha256')
    nodal_metadata = np.load(perpendicular/NODAL_FILTER_METADATA, allow_pickle=False)
    nodal_hash = json.loads(nodal_metadata.item()).get('arm_sha256')
    assert table_hash == nodal_hash == expected, (case, table_hash, nodal_hash, expected)
    return dict(field_table=table_hash, nodal=nodal_hash, reconstructed=expected)


def test_cold_rebuild_matches_reference(tmp_path, runtime_module):
    """Compare both metadata members without the arm hash because cold metric files contain wall-clock timings.

    On each side, both members must carry the hash reconstructed from that case's own metric file.
    """
    reference = current_case_directory()
    options = cold_options(reference)
    driver = runtime_module('case')
    case = tmp_path/'cold'
    driver.build_case(options, case)
    driver.run_mms(case, mms_profile(options, reference))
    differences = compare_case_products(case, reference)
    (tmp_path/'owner_array_differences.json').write_text(json.dumps(differences, indent=2)+'\n')
    for root in (case, reference):
        require_case_arm_identity(root)
    require_bytes(differences)
