"""Compact host bank for two checked frozen Q span views.

This is storage extraction, not a new reconstruction policy. Runtime payload
contains five scalar positions and precontracted diffusion; coordinate gradients
are optional host diagnostics. Logical coordinates are not physical positions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import numpy as np

from .q_parallel import PreparedQ, _ARRAYS, sha256_array
from .query_tables import ExactQueryTable

SCHEMA = 'drbx.q-paired-bank.v1'
SPANS = (1 / 16, 1 / 32)
SLOT_NAMES = ('outer_minus', 'inner_minus', 'center', 'inner_plus', 'outer_plus')
SPAN_SLOTS = ((0, 4, 2), (1, 3, 2))
COMMON = ('owners', 'raw', 'raw_to_owner', 'raw_weight', 'owner_raw',
          'owner_weight', 'donor', 'mask', 'row_count')
SCALAR = ('row_value_D', 'row_value_N', 'boundary_value_D_trace',
          'boundary_value_N_normal', 'slot_points', 'magnetic_b', 'boundary_wall_queries')
GRADIENTS = ('row_gradient_D', 'row_gradient_N', 'boundary_gradient_D_trace',
             'boundary_gradient_N_normal')
REQUIRED = COMMON + ('wall_index', 'row_value_D', 'row_value_N_wall',
    'boundary_value_D_trace', 'boundary_value_N_normal', 'diffusion_D',
    'diffusion_N_wall', 'boundary_D_node', 'boundary_D_tangent',
    'boundary_N_normal', 'query_table', 'wall_node_query', 'wall_slot_query',
    'boundary_wall_normal', 'magnetic_b', 'magnetic_L')
IDENTITY_KEYS = ('campaign_identity', 'source_identity', 'geometry_identity',
                 'topology_hash', 'trace_hash', 'support', 'n', 'n_owner',
                 'trace_steps', 'trace_method')


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      default=lambda x: x.item() if isinstance(x, np.generic) else x)


def content_identity(metadata, arrays, diagnostics):
    """Hash complete metadata and every content buffer, never file timestamps."""
    h = hashlib.sha256(_json(metadata).encode())
    for group, values in (('arrays', arrays), ('diagnostics', diagnostics)):
        for name, array in sorted(values.items()):
            h.update(f'{group}:{name}:'.encode())
            h.update(sha256_array(array).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class QBank:
    metadata: dict
    arrays: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]

    def __getattr__(self, name):
        arrays = object.__getattribute__(self, 'arrays')
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)

    @property
    def identity(self):
        return content_identity(self.metadata, self.arrays, self.diagnostics)

    def footprint(self):
        """Report host buffers and padding; NPZ bytes are a separate disk measure."""
        runtime = sum(a.nbytes for a in self.arrays.values())
        host_diagnostics = sum(a.nbytes for a in self.diagnostics.values())
        roots = {}
        for a in (*self.arrays.values(), *self.diagnostics.values()):
            root = a
            while isinstance(root.base, np.ndarray):
                root = root.base
            roots[id(root)] = root.nbytes
        # donor, mask, five scalar D rows, and two diffusion D rows.
        padding = int(np.count_nonzero(~self.mask)) * (
            self.donor.dtype.itemsize + self.mask.dtype.itemsize +
            5 * self.row_value_D.dtype.itemsize + 2 * self.diffusion_D.dtype.itemsize)
        wall_padding = int(np.count_nonzero(~self.mask[self.wall_index])) * (
            5 * self.row_value_N_wall.dtype.itemsize + 2 * self.diffusion_N_wall.dtype.itemsize)
        return dict(runtime_logical_bytes=runtime, host_diagnostic_bytes=host_diagnostics,
                    host_logical_bytes=runtime + host_diagnostics,
                    unique_buffer_bytes=sum(roots.values()), padded_bytes=padding + wall_padding,
                    shared_topology_donor_projection_bytes=sum(self.arrays[k].nbytes for k in COMMON),
                    scalar_row_bytes=self.row_value_D.nbytes + self.row_value_N_wall.nbytes,
                    diffusion_row_bytes=self.diffusion_D.nbytes + self.diffusion_N_wall.nbytes,
                    boundary_runtime_bytes=sum(a.nbytes for k, a in self.arrays.items()
                        if k.startswith(('boundary_', 'wall_', 'query_'))),
                    gradient_diagnostic_bytes=sum(self.diagnostics[k].nbytes for k in GRADIENTS
                                                  if k in self.diagnostics),
                    metadata_json_bytes=len(_json(self.metadata).encode()))

    def validate(self):
        m = self.metadata
        if (m.get('schema') != SCHEMA or tuple(m.get('spans', ())) != SPANS or
            tuple(m.get('slot_names', ())) != SLOT_NAMES):
            raise ValueError('Q bank schema/span/slot identity mismatch')
        if any(not m.get(k) for k in IDENTITY_KEYS):
            raise ValueError('incomplete Q bank identity')
        legacy = m.get('span_metadata', [])
        if len(legacy) != 2:
            raise ValueError('Q bank span provenance incomplete')
        for span, record in zip(SPANS, legacy):
            if record.get('span') != span or any(record.get(k) != m[k] for k in IDENTITY_KEYS):
                raise ValueError('Q bank span provenance identity mismatch')
        if (m['trace_steps'] != 64 or m['trace_method'] != 'RK4' or
            m['n'] not in (32, 48, 64) or m['n_owner'] <= 0):
            raise ValueError('invalid frozen Q bank identity')
        if set(self.arrays) != set(REQUIRED):
            raise ValueError('Q bank array manifest incomplete')
        for key in ('raw','owners','raw_to_owner','owner_raw','wall_index','row_count','donor'):
            if self.arrays[key].dtype.kind not in 'iu':
                raise ValueError(f'Q bank {key} requires integer identity dtype')
        trace_parts = str(m['trace_hash']).split(':')
        if (len(trace_parts)!=2 or any(re.fullmatch('[0-9a-f]{64}',part) is None for part in trace_parts) or
            trace_parts[0]!=sha256_array(self.raw)):
            raise ValueError('Q bank raw content/trace receipt mismatch')
        coefficients = ('raw_weight','owner_weight','row_value_D','row_value_N_wall',
            'boundary_value_D_trace','boundary_value_N_normal','diffusion_D','diffusion_N_wall',
            'boundary_D_node','boundary_D_tangent','boundary_N_normal','boundary_wall_normal',
            'magnetic_b','magnetic_L','query_table')
        for key in coefficients:
            if self.arrays[key].dtype.kind not in 'iuf':
                raise ValueError(f'Q bank {key} requires real numeric dtype')
        for key in GRADIENTS:
            if key in self.diagnostics and self.diagnostics[key].dtype.kind not in 'iuf':
                raise ValueError(f'Q bank {key} requires real numeric dtype')
        nr, nd = self.donor.shape
        nw = len(self.wall_index)
        if self.owners.ndim != 1 or self.wall_index.ndim != 1:
            raise ValueError('Q bank owner/wall identity shape mismatch')
        if nr == 0 or len(self.owners) == 0:
            raise ValueError('empty Q bank')
        shapes = dict(raw=(nr,), raw_to_owner=(nr,), raw_weight=(nr,), mask=(nr, nd),
            row_count=(nr,), row_value_D=(nr, 5, nd), row_value_N_wall=(nw, 5, nd),
            boundary_value_D_trace=(nw, 5, 35), boundary_value_N_normal=(nw, 5, 35),
            diffusion_D=(2, nr, nd), diffusion_N_wall=(2, nw, nd),
            boundary_D_node=(2, nw, 35), boundary_D_tangent=(2, nw, 3, 2),
            boundary_N_normal=(2, nw, 35), wall_node_query=(nw, 35),
            wall_slot_query=(nw, 5), boundary_wall_normal=(nw, 35, 3),
            magnetic_b=(nr, 5, 3), magnetic_L=(2, nr, 3))
        for k, shape in shapes.items():
            if self.arrays[k].shape != shape:
                raise ValueError(f'Q bank {k} shape mismatch')
        if self.query_table.ndim != 2 or self.query_table.shape[1] != 3:
            raise ValueError('Q bank query table shape')
        for ids in (self.wall_node_query, self.wall_slot_query):
            if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(self.query_table)):
                raise ValueError('Q bank query index')
        if not np.array_equal(self.wall_index, np.flatnonzero(self.raw // m['n']**2 >= m['n'] - 2)):
            raise ValueError('Q bank wall partition mismatch')
        if (self.donor.dtype.kind not in 'iu' or self.mask.dtype.kind != 'b' or
            np.any(self.donor[~self.mask] != 0) or
            np.any(self.donor[self.mask] >= m['n_owner']) or np.any(self.donor < 0) or
            not np.array_equal(self.row_count, self.mask.sum(1)) or
            not np.array_equal(self.mask,np.arange(nd)[None,:]<self.row_count[:,None])):
            raise ValueError('Q bank donor bounds/padding mismatch')
        # Padded safe donor ids still gather a live value. All channels must
        # have zero coefficients there, including wall replacements and
        # optional coordinate-gradient diagnostics.
        for key, mask, order in (
            ('row_value_D',self.mask,(0,2,1)),
            ('diffusion_D',self.mask,(1,2,0)),
            ('row_value_N_wall',self.mask[self.wall_index],(0,2,1)),
            ('diffusion_N_wall',self.mask[self.wall_index],(1,2,0))):
            if np.any(self.arrays[key].transpose(order)[~mask] != 0):
                raise ValueError(f'Q bank {key} nonzero padded coefficient')
        if m.get('has_gradients'):
            for key in ('row_gradient_D','row_gradient_N'):
                if key in self.diagnostics:
                    if self.diagnostics[key].shape!=(nr,5,3,nd):
                        raise ValueError('Q bank coordinate gradient shape mismatch')
                    if np.any(self.diagnostics[key].transpose(0,3,1,2)[~self.mask]!=0):
                        raise ValueError(f'Q bank {key} nonzero padded coefficient')
        if (len(np.unique(self.raw)) != nr or len(np.unique(self.owners)) != len(self.owners) or
            np.any(self.owners < 0) or np.any(self.owners >= m['n_owner']) or
            np.any(self.raw < 0) or np.any(self.raw >= m['n']**3) or
            np.any(self.raw_to_owner < 0) or np.any(self.raw_to_owner >= len(self.owners)) or
            np.any(np.diff(self.raw_to_owner) < 0)):
            raise ValueError('Q bank source identities mismatch')
        for row, count in zip(self.donor, self.row_count):
            if len(np.unique(row[:count])) != count:
                raise ValueError('Q bank duplicate donor identity')
        if self.owner_raw.shape != self.owner_weight.shape or self.owner_raw.shape[0] != len(self.owners):
            raise ValueError('Q bank projection shape')
        if (np.any(self.owner_raw < 0) or np.any(self.owner_raw >= nr) or
            np.any(self.owner_weight < 0) or np.any(self.raw_weight <= 0)):
            raise ValueError('Q bank projection index')
        used = self.owner_raw[self.owner_weight != 0]
        if (not np.array_equal(np.sort(used), np.arange(nr)) or
            not np.allclose(self.owner_weight.sum(1), 1, rtol=0, atol=1e-12)):
            raise ValueError('Q bank raw-member coverage/projection incomplete')
        for i in range(len(self.owners)):
            active = self.owner_weight[i] != 0
            rows = self.owner_raw[i, active]
            if (np.any(self.raw_to_owner[rows] != i) or
                not _bits_equal(self.owner_weight[i, active], self.raw_weight[rows])):
                raise ValueError('Q bank projection weights mismatch')
        for a in (*self.arrays.values(), *self.diagnostics.values()):
            if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():
                raise ValueError('Q bank nonfinite/unsupported payload')
        if m.get('has_gradients') != all(k in self.diagnostics for k in GRADIENTS):
            raise ValueError('Q bank gradient diagnostic manifest incomplete')
        expected_diagnostics = {'choice', 'indicator', 'noise', 'slot_points'}
        if m['has_gradients']:
            expected_diagnostics.update(GRADIENTS)
            for k in GRADIENTS:
                width = 35 if k.startswith('boundary') else nd
                if self.diagnostics[k].shape != (nr, 5, 3, width):
                    raise ValueError('Q bank coordinate gradient shape mismatch')
        if set(self.diagnostics) != expected_diagnostics:
            raise ValueError('Q bank diagnostic manifest incomplete')
        if (self.diagnostics['slot_points'].shape != (nr, 5, 3) or
            self.diagnostics['choice'].shape != (nr,) or
            self.diagnostics['indicator'].shape != (nr, 3, 2) or
            self.diagnostics['noise'].shape != (nr, 3)):
            raise ValueError('Q bank diagnostic shape mismatch')
        return self


def _five(outer, inner, name):
    a, b = getattr(outer, name), getattr(inner, name)
    if not _bits_equal(a[:, 2], b[:, 2]):
        raise ValueError(f'paired center {name} differs')
    return np.stack((a[:, 0], b[:, 0], a[:, 2], b[:, 1], a[:, 1]), axis=1)


def build_q_bank(outer, inner, *, identity=None, include_gradients=False):
    """Compact checked h/16,h/32 preparations without changing coefficient order.

    ``identity`` may add evaluator/normalization/choice provenance; it cannot
    replace any checked legacy identity. Physical coordinate maps, if needed,
    belong to a producer adapter, since PreparedQ only supplies logical points.
    """
    for q, span in zip((outer, inner), SPANS):
        q.validate(q.metadata['campaign_identity'])
        if q.metadata['span'] != span:
            raise ValueError('paired Q spans must be outer h/16, inner h/32')
    # Timers and the selected span are observational; every remaining legacy
    # policy/diagnostic identity must agree before any center is deduplicated.
    pair_keys = (set(outer.metadata) | set(inner.metadata)) - {
        'span', 'magnetic_seconds', 'rows_seconds', 'packing_seconds'}
    for k in sorted(pair_keys):
        if outer.metadata.get(k) != inner.metadata.get(k):
            raise ValueError(f'paired {k} identity mismatch')
    for name in COMMON + ('choice', 'indicator', 'noise', 'boundary_wall_nodes', 'boundary_wall_normal'):
        if not _bits_equal(getattr(outer, name), getattr(inner, name)):
            raise ValueError(f'paired {name} mismatch')
    nr = len(outer.raw)
    wall = np.flatnonzero(outer.raw // outer.metadata['n']**2 >= outer.metadata['n'] - 2).astype(np.int32)
    interior = np.ones(nr, bool); interior[wall] = False
    for q in (outer, inner):
        for d, n in (('row_value_D', 'row_value_N'), ('diffusion_D', 'diffusion_N')):
            if not _bits_equal(getattr(q, d)[interior], getattr(q, n)[interior]):
                raise ValueError(f'interior D/N {d} differs')
        for k in ('boundary_value_D_trace', 'boundary_value_N_normal'):
            if np.any(getattr(q, k)[interior] != 0):
                raise ValueError('non-wall scalar boundary response')
    five = {k: _five(outer, inner, k) for k in SCALAR}
    arrays = {k: getattr(outer, k).copy() for k in COMMON}
    arrays.update(wall_index=wall, row_value_D=five['row_value_D'],
        row_value_N_wall=five['row_value_N'][wall],
        boundary_value_D_trace=five['boundary_value_D_trace'][wall],
        boundary_value_N_normal=five['boundary_value_N_normal'][wall],
        diffusion_D=np.stack((outer.diffusion_D, inner.diffusion_D)),
        diffusion_N_wall=np.stack((outer.diffusion_N[wall], inner.diffusion_N[wall])),
        magnetic_b=five['magnetic_b'], magnetic_L=np.stack((outer.magnetic_L, inner.magnetic_L)),
        boundary_wall_normal=outer.boundary_wall_normal[wall].copy())
    for k in ('boundary_D_node', 'boundary_D_tangent', 'boundary_N_normal'):
        arrays[k] = np.stack((getattr(outer, k)[wall], getattr(inner, k)[wall]))
    table = ExactQueryTable()
    arrays['wall_node_query'] = np.array([table.add_many(p) for p in outer.boundary_wall_nodes[wall]],
                                        np.int32).reshape(len(wall), 35)
    arrays['wall_slot_query'] = np.array([table.add_many(p) for p in five['boundary_wall_queries'][wall]],
                                        np.int32).reshape(len(wall), 5)
    arrays['query_table'] = table.array()
    diagnostics = {k: getattr(outer, k).copy() for k in ('choice', 'indicator', 'noise')}
    diagnostics['slot_points'] = five['slot_points']
    if include_gradients:
        diagnostics.update({k: _five(outer, inner, k) for k in GRADIENTS})
    metadata = {k: outer.metadata[k] for k in IDENTITY_KEYS}
    metadata.update(schema=SCHEMA, spans=list(SPANS), slot_names=list(SLOT_NAMES),
        has_gradients=bool(include_gradients), query_coordinates='logical (r,theta,eta)',
        boundary_roles={'wall_node_query': ['D_value', 'N_physical_normal'],
                        'wall_slot_query': ['D_query_value', 'D_theta_eta_derivatives']},
        span_metadata=[dict(outer.metadata), dict(inner.metadata)],
        donor_storage='shared padded dense; bucket descriptors only')
    families = outer.metadata.get('row_diagnostics', [])
    buckets = {}
    for r in range(nr):
        family = families[r]['family'] if len(families) == nr else ('wall' if not interior[r] else 'interior')
        key = (family, int(outer.row_count[r]))
        buckets.setdefault(key, []).append(r)
    metadata['buckets'] = [dict(family=k[0], width=k[1], raw_indices=v) for k, v in buckets.items()]
    if identity:
        if any(k in metadata and metadata[k] != v for k, v in identity.items()):
            raise ValueError('additional Q identity overrides checked identity')
        metadata['additional_identity'] = identity
    return QBank(metadata, arrays, diagnostics).validate()


def decode_span(bank, span):
    """Exact PreparedQ decode, available only when full gradients were captured."""
    bank.validate()
    if not bank.metadata['has_gradients']:
        raise ValueError('PreparedQ decode requires captured coordinate gradients')
    if span not in SPANS:
        raise ValueError('unsupported Q bank span')
    ai = SPANS.index(span); slots = np.array(SPAN_SLOTS[ai]); nr = len(bank.raw)
    a = {k: bank.arrays[k].copy() for k in COMMON}
    a.update({k: bank.diagnostics[k].copy() for k in ('choice', 'indicator', 'noise')})
    for k in ('slot_points', *GRADIENTS):
        a[k] = bank.diagnostics[k][:, slots].copy()
    a['row_value_D'] = bank.row_value_D[:, slots].copy()
    a['row_value_N'] = a['row_value_D'].copy()
    a['row_value_N'][bank.wall_index] = bank.row_value_N_wall[:, slots]
    for k in ('boundary_value_D_trace', 'boundary_value_N_normal'):
        a[k] = np.zeros((nr, 3, 35), dtype=bank.arrays[k].dtype)
        a[k][bank.wall_index] = bank.arrays[k][:, slots]
    a['diffusion_D'] = bank.diffusion_D[ai].copy()
    a['diffusion_N'] = a['diffusion_D'].copy()
    a['diffusion_N'][bank.wall_index] = bank.diffusion_N_wall[ai]
    for k in ('boundary_D_node', 'boundary_D_tangent', 'boundary_N_normal'):
        source = bank.arrays[k][ai]
        a[k] = np.zeros((nr, *source.shape[1:]), dtype=source.dtype)
        a[k][bank.wall_index] = source
    for k, shape in (('boundary_wall_nodes', (35, 3)), ('boundary_wall_queries', (3, 3)),
                     ('boundary_wall_normal', (35, 3))):
        a[k] = np.zeros((nr, *shape))
    a['boundary_wall_nodes'][bank.wall_index] = bank.query_table[bank.wall_node_query]
    a['boundary_wall_queries'][bank.wall_index] = bank.query_table[bank.wall_slot_query[:, slots]]
    a['boundary_wall_normal'][bank.wall_index] = bank.boundary_wall_normal
    a['magnetic_b'] = bank.magnetic_b[:, slots].copy()
    a['magnetic_L'] = bank.magnetic_L[ai].copy()
    assert set(a) == set(_ARRAYS)
    metadata = dict(bank.metadata['span_metadata'][ai])
    return PreparedQ(metadata, **a).validate(metadata['campaign_identity'])
