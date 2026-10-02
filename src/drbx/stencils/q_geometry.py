"""Host-only geometry adapter for saved-trace Q preparation.

The adapter records evaluator and coordinate-map identities separately. It
does not change batching, normalize geometry, trace, clip queries or import
research campaigns. Existing bare callback preparation remains supported.
"""
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Callable, Mapping


@dataclass(frozen=True, init=False)
class QGeometryProvider:
    """Callbacks plus a content-addressed, field-independent geometry contract.

    ``geometry(points)`` returns ``(J, b_contravariant, B_magnitude)``;
    ``coordinate_jacobian(points)`` returns the embedding Jacobian. Pass
    ``provider.geom`` and ``provider.jacobian`` to ``prepare_paired_chunks``.
    Input and output objects/batch shapes pass through untouched. Callbacks
    remain responsible for rejecting unsupported queries, without clipping.
    """
    geometry: Callable
    coordinate_jacobian: Callable
    _canonical_provenance: str

    def __init__(self, geometry, coordinate_jacobian, provenance: Mapping):
        if not callable(geometry) or not callable(coordinate_jacobian):
            raise TypeError('geometry and coordinate Jacobian must be callable')
        required = ('magnetic_evaluator', 'coordinate_map', 'normalization',
                    'validity', 'batching', 'input_hashes', 'source_hashes')
        # Freeze a canonical JSON value, so later mutation of caller dictionaries
        # cannot change the identity attached to already-prepared rows.
        value = json.loads(json.dumps(dict(provenance), sort_keys=True, allow_nan=False))
        if any(not value.get(k) for k in required):
            raise ValueError('incomplete Q geometry provenance')
        for group in ('input_hashes', 'source_hashes'):
            hashes = value[group]
            if not isinstance(hashes, dict) or any(
                not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                for h in hashes.values()
            ):
                raise ValueError(f'{group} requires SHA256 content hashes')
        object.__setattr__(self, '_canonical_provenance', json.dumps(value, sort_keys=True, separators=(',', ':')))
        object.__setattr__(self, 'geometry', geometry)
        object.__setattr__(self, 'coordinate_jacobian', coordinate_jacobian)

    @property
    def provenance(self):
        return json.loads(self._canonical_provenance)

    @property
    def identity(self):
        return hashlib.sha256(self._canonical_provenance.encode()).hexdigest()

    def identity_record(self):
        """Independent receipt for QBank.additional_identity; safe to mutate."""
        return {'geometry_provider': json.loads(self._canonical_provenance),
                'geometry_provider_identity': self.identity}

    def geom(self, points):
        return self.geometry(points)

    def jacobian(self, points):
        return self.coordinate_jacobian(points)
