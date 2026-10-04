"""Checked prepared-input boundary; no retracing or global campaign imports."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import jax
import numpy as np
from drbx.stencils.q_plan import lower_q_plan
from drbx.native.q_plan import stage_q_plan
from scripts.q08_extraction_global.common import digest, atomic_npz
from .mms import Manufactured, ContinuumReference

FROZEN_IDENTITY = 'ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722'


def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


SOURCE_DEPENDENCIES = (
    'scripts/q08_extraction_global/common.py', 'scripts/q08_rhs_mms_global/science.py',
    'scripts/q08_extraction_global/gpu.py', 'scripts/q08_extraction_global/design.json',
    'scripts/q08_extraction_global/input_manifest.json', 'src/drbx/stencils/q_artifact.py',
    'src/drbx/stencils/query_tables.py',
    'src/drbx/native/q_sharding.py', 'src/drbx/geometry/_reconstruction_primitives.py',
    'src/drbx/native/fci_time_integrator.py', 'src/drbx/native/fci_model.py',
    'src/drbx/native/q_plan.py', 'src/drbx/stencils/q_plan.py', 'src/drbx/stencils/q_bank.py',
    'src/drbx/stencils/q_parallel.py', 'src/drbx/stencils/q_parallel_channels.py',
    'src/drbx/native/q_parallel.py', 'src/drbx/native/q_parallel_rhs.py',
    'src/drbx/native/q_parallel_material.py', 'src/drbx/native/q_parallel_current_phi.py',
    'src/drbx/native/q_parallel_vorticity.py', 'src/drbx/native/q_parallel_characteristic.py',
    'src/drbx/native/q_characteristic_polynomial.py', 'src/drbx/native/fci_parallel_production_flux.py',
    'src/drbx/native/characteristic_wall_residual.py', 'src/drbx/native/q_parallel_divergence.py',
    'src/drbx/native/q_parallel_gradient.py', 'src/drbx/native/q_parallel_channels.py',
)


def source_hash():
    """Explicit time/source/application dependency closure plus harness sources."""
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob('*.py'))+[root/p for p in SOURCE_DEPENDENCIES]
    paths += list((Path(__file__).parent/'runtime').rglob('*.py'))
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def validate_frozen_policy(bank, geometry):
    """Check actual bank provenance; receipt identity is a separate input."""
    m = bank.metadata
    if not str(m.get('geometry_identity', '')).startswith(('C3:', 'compact_c3:')):
        raise ValueError('Q09 frozen C3 magnetic identity mismatch')
    if m.get('trace_steps') != 64 or m.get('trace_method') != 'RK4':
        raise ValueError('Q09 frozen trace identity mismatch')
    if m.get('support') != 'selective compact28/gradient guard; structured outer; quartic physical wall':
        raise ValueError('Q09 frozen support policy mismatch')
    if not np.allclose(np.asarray(geometry['eta_step']), np.pi/m['n']/32, rtol=2e-13, atol=0):
        raise ValueError('Q09 h/32 inner cap identity mismatch')
    return True


@dataclass(frozen=True)
class PreparedProvider:
    bank: object
    geometry: dict
    manufactured: Manufactured
    reference: ContinuumReference
    volume: object
    regions: dict
    input_hashes: dict

    def validate(self):
        self.bank.validate()
        b = self.bank; n = int(b.metadata['n']); no = int(b.metadata['n_owner'])
        validate_frozen_policy(b, self.geometry)
        if not np.array_equal(b.owners, np.arange(no)) or len(b.raw) != n**3 or not np.array_equal(np.sort(b.raw), np.arange(n**3)):
            raise ValueError('scientific evolution requires full-domain owner/raw closure; bounded fixtures are replay-only')
        if np.any(b.donor < 0) or np.any(b.donor >= no):
            raise ValueError('incomplete donor closure')
        # The accepted bank span_metadata carries inner/outer support/trace
        # policy provenance and is validated by the bank authority.
        plan = lower_q_plan(b, diffusion_span=1/32, **self.geometry)
        if not np.allclose(np.asarray(plan.eta_step), np.pi/n/32, rtol=2e-13, atol=0):
            raise ValueError('Q09 h/32 inner cap identity mismatch')
        vol = np.asarray(self.volume)
        if vol.shape != (no,) or not np.isfinite(vol).all() or np.any(vol <= 0):
            raise ValueError('invalid complete owner volumes')
        m = self.manufactured; r = self.reference
        if np.shape(m.owner_initial) != (6, no) or np.shape(m.phi_initial) != (no,):
            raise ValueError('MMS owner coverage mismatch')
        if any(np.shape(getattr(r, k)) != shape for k, shape in (('values', (n**3, 6)), ('gradients', (n**3, 6)), ('phi_gradient', (n**3,)), ('kappa', (n**3,)), ('bmag', (n**3,)), ('diffusion', (no, 6)))):
            raise ValueError('reference coverage mismatch')
        if not np.array_equal(r.owner_raw, b.owner_raw) or not np.array_equal(r.owner_weight, b.owner_weight):
            raise ValueError('reference projection identity mismatch')
        v = np.zeros((no, 6)); p = np.zeros(no)
        from .mms import smooth_fields
        values, gradients, phi, phi_gradients = smooth_fields(b.diagnostics['slot_points'][:, 2], 0.)
        np.testing.assert_allclose(r.values, values, rtol=2e-13, atol=2e-14)
        direction = np.asarray(b.magnetic_b)[:, 2]
        np.testing.assert_allclose(r.gradients, np.einsum('ra,rfa->rf', direction, gradients), rtol=2e-11, atol=1e-10)
        np.testing.assert_allclose(r.phi_gradient, np.einsum('ra,ra->r', direction, phi_gradients), rtol=2e-11, atol=1e-10)
        np.testing.assert_allclose(r.bmag, self.geometry['bmag'], rtol=0, atol=1e-10)
        values, phi = np.asarray(values), np.asarray(phi)
        for i in range(no):
            active = b.owner_weight[i] != 0
            rr, ww = b.owner_raw[i, active], b.owner_weight[i, active]
            v[i] = np.sum(values[rr]*ww[:, None], axis=0)
            p[i] = np.sum(phi[rr]*ww)
        np.testing.assert_allclose(m.owner_initial, v.T, atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(m.phi_initial, p, atol=2e-14, rtol=2e-13)
        from scripts.q08_extraction_global.common import boundaries
        for actual, expected in ((m.boundary_initial, boundaries(b, 1)), (m.boundary_constant, boundaries(b, 0))):
            for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                np.testing.assert_array_equal(a, e)
        if self.input_hashes.get('q08_receipt_identity') != FROZEN_IDENTITY or self.input_hashes.get('prepared_bank_identity') != b.identity:
            raise ValueError('Q09 prepared input/accepted receipt lineage mismatch')
        if not self.input_hashes or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in self.input_hashes.values()):
            raise ValueError('explicit content input hashes required')
        for a in jax.tree.leaves((m.owner_initial, m.phi_initial, m.boundary_initial,
                                  r.values, r.gradients, r.diffusion, r.kappa, r.bmag, r.phi_gradient)):
            if not np.isfinite(a).all():
                raise ValueError('nonfinite analytic/reference payload')
        self._validate_regions(no)
        return self

    def _validate_regions(self, no):
        for name, mask in self.regions.items():
            if name == 'global' or np.shape(mask) != (no,) or np.asarray(mask).dtype.kind != 'b':
                raise ValueError('invalid regional mask')

    @property
    def plan(self):
        return stage_q_plan(lower_q_plan(self.bank, diffusion_span=1/32, **self.geometry))

    @property
    def identity(self):
        reference = {k: array_hash(getattr(self.reference, k)) for k in
            ('values', 'gradients', 'phi_gradient', 'diffusion', 'kappa', 'bmag', 'owner_raw', 'owner_weight')}
        return digest(dict(bank=self.bank.identity, inputs=self.input_hashes, reference=reference,
            reference_diagnostics=self.reference.diagnostics, geometry={k: array_hash(v) for k, v in self.geometry.items()}, volume=array_hash(self.volume),
            regions={k: array_hash(v) for k, v in self.regions.items()}, source=source_hash()))


def save_checkpoint(path, identity, payload):
    arrays = dict(state=np.asarray(payload['state']), accepted_steps=np.asarray(payload['accepted_steps']),
        time=np.asarray(payload['time']), integral_rhs=np.asarray(payload['integral_rhs']),
        stage_times=np.asarray(payload['stage_times']))
    receipt = digest({k: array_hash(a) for k, a in arrays.items()})
    atomic_npz(path, signature=np.array(identity), content_digest=np.array(receipt), **arrays)


def load_checkpoint(path, identity):
    with np.load(path, allow_pickle=False) as z:
        if str(z['signature']) != identity:
            raise ValueError('stale checkpoint input/source/run identity')
        keys = ('state', 'accepted_steps', 'time', 'integral_rhs', 'stage_times')
        if set(z.files) != set(keys) | {'signature', 'content_digest'}:
            raise ValueError('checkpoint array manifest mismatch')
        arrays = {k: z[k].copy() for k in keys}
        if str(z['content_digest']) != digest({k: array_hash(a) for k, a in arrays.items()}):
            raise ValueError('checkpoint content hash mismatch')
        if arrays['accepted_steps'].shape != () or arrays['accepted_steps'].dtype.kind not in 'iu':
            raise ValueError('checkpoint step dtype/shape mismatch')
        if arrays['time'].shape != () or arrays['integral_rhs'].shape != (6,) or arrays['stage_times'].ndim != 1:
            raise ValueError('checkpoint diagnostic shape mismatch')
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('nonfinite checkpoint content')
        return dict(state=arrays['state'], accepted_steps=int(arrays['accepted_steps']), time=float(arrays['time']),
            integral_rhs=arrays['integral_rhs'], stage_times=arrays['stage_times'].tolist())
