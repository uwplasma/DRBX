"""Package-backed continuous evaluator for the saved field contract."""
import jax
import jax.numpy as jnp
import numpy as np
from ..evaluators.metric_evaluator import MetricEvaluator
from ..evaluators.eta_filtered_field import EtaFilterTable
from ..evaluators.jax_metric_evaluator import JaxMetricEvaluator


class Evaluator:
    capacity = 512

    def __init__(self, metric_cache, field_table, options):
        with np.load(metric_cache, allow_pickle=False) as payload:
            metric = MetricEvaluator.from_cache_payload(payload, prefix='metric_evaluator_')
        self.metric = JaxMetricEvaluator.from_metric_evaluator(metric)
        table = EtaFilterTable.load(field_table)
        self._bind(self.metric, table, options)

    @classmethod
    def from_objects(cls, metric, table, options):
        """Use a package JAX metric evaluator and ETA table supplied by the caller."""
        evaluator = cls.__new__(cls)
        evaluator._bind(metric, table, options)
        return evaluator

    def _bind(self, metric, table, options):
        self.metric = metric
        if table.spec.nfp != options.field_periods:
            raise ValueError('The table and trace options give different physical field periods')
        if options.field_filter is not None and table.spec != options.field_filter:
            raise ValueError('The table and trace options give different filter specifications')
        self.table_max = table.u_max
        self.table = table.to_jax()

        def evaluate(q):
            pos, A = self.metric.position_and_jacobian(q)
            B = self.table.flux_density(q)/jnp.linalg.det(A)[:,None]
            return pos, A, B, jnp.linalg.norm(jnp.einsum('pij,pj->pi', A, B), axis=1)

        self._evaluate = jax.jit(evaluate)
        self.beta_min = float('inf')
        self.beta_max = -float('inf')
        self.max_u = 0.

    def __call__(self, points):
        out = [[] for _ in range(4)]
        for start in range(0, len(points), self.capacity):
            q = np.asarray(points[start:start+self.capacity], float).copy()
            q[:,1:] %= 2*np.pi
            length = len(q)
            padded = np.concatenate((q, np.repeat(q[-1:], self.capacity-length, axis=0)))
            ans = self._evaluate(padded)
            for dest, value in zip(out, ans):
                dest.append(np.asarray(value)[:length])
        pos, A, B, mag = (np.concatenate(v) for v in out)
        if not all(np.isfinite(v).all() for v in (pos,A,B,mag)) or np.any(mag <= 0):
            raise ValueError('Invalid continuous geometry/field')
        if np.any(np.linalg.det(A) <= 0):
            raise ValueError('Nonpositive chart Jacobian')
        beta = B[:,2]
        self.beta_min = min(self.beta_min, float(beta.min()))
        self.beta_max = max(self.beta_max, float(beta.max()))
        self.max_u = max(self.max_u, float(np.max(points[:,0])))
        if np.any(beta == 0) or self.beta_min*self.beta_max <= 0:
            raise ValueError('B^eta zero or sign change')
        return B, mag, A, pos
