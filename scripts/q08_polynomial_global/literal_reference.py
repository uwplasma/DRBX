"""Independent eager chunk actions, with validated host plans reused per span."""
import time
import numpy as np
import jax
from scripts.q08_extraction_global.gpu import lower_q_plan, apply_q_plan, _sync


class LiteralReference:
    """Reuse only lowering/validation; retain all original CPU eig actions.

    QPlan contains NumPy views of the immutable checked bank/geometry, so this
    cache adds no dense field/action storage. A new object is used per span.
    """
    def __init__(self, chunks, span, api):
        self.chunks, self.span, self.api = chunks, span, api
        self.plans = {}
        self.stats = dict(lowerings=0, actions=0, lowering_seconds=0.,
                          boundary_seconds=0., action_seconds=0.)

    def __call__(self, x, phi, case, kinds, pk):
        outputs, definition = [], None
        with jax.default_device(jax.devices('cpu')[0]):
            for index, chunk in enumerate(self.chunks):
                if index not in self.plans:
                    start = time.perf_counter()
                    self.plans[index] = lower_q_plan(chunk.bank, diffusion_span=self.span, **chunk.geometry)
                    self.stats['lowering_seconds'] += time.perf_counter()-start
                    self.stats['lowerings'] += 1
                start = time.perf_counter()
                bi, bo, pb = self.api.boundaries(chunk.bank, case)
                self.stats['boundary_seconds'] += time.perf_counter()-start
                start = time.perf_counter()
                result = _sync(apply_q_plan(self.plans[index], x, bi, bo, phi, pb, self.api.COEFF,
                    kinds=tuple(kinds), phi_kind=pk, tau=self.api.TAU, mu=self.api.MU))
                leaves, treedef = jax.tree.flatten(result)
                if definition is None:
                    definition = treedef
                elif definition != treedef:
                    raise ValueError('CPU output structure changed across chunks')
                outputs.append(tuple(np.asarray(a) for a in leaves))
                self.stats['action_seconds'] += time.perf_counter()-start
                self.stats['actions'] += 1
        if not outputs:
            raise ValueError('empty literal chunk reference')
        return jax.tree.unflatten(definition, [np.concatenate([part[i] for part in outputs], axis=0)
                                              for i in range(len(outputs[0]))])
