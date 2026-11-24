"""Frozen field and metric evaluation from explicit cache paths."""
import jax.numpy as jnp
import numpy as np


def evaluate_frozen_field(metric,table,q):
 pos,A=metric.position_and_jacobian(q);B=table.flux_density(q)/jnp.linalg.det(A)[:,None];return pos,A,B,jnp.linalg.norm(jnp.einsum('pij,pj->pi',A,B),axis=1)

class FrozenFieldEvaluator:
 def __call__(self,points):
  out=[[],[],[],[]]
  for lo in range(0,len(points),self.capacity):
   q=np.asarray(points[lo:lo+self.capacity],float).copy();q[:,1:]%=2*np.pi;n=len(q);q=np.r_[q,np.repeat(q[-1:],self.capacity-n,axis=0)]
   ans=self.fn(q)
   for a,b in zip(out,ans):a.append(np.asarray(b)[:n])
  pos,A,B,mag=[np.concatenate(x) for x in out]
  if not all(np.isfinite(x).all() for x in [pos,A,B,mag]) or np.any(np.linalg.det(A)<=0) or np.any(mag<=0):raise ValueError('Invalid frozen field/metric')
  return B,mag,A,pos


_MISSING = object()


class Evaluator(FrozenFieldEvaluator):
    capacity = 512

    def __init__(self, metric_cache, field_table, *, filter_spec=None, field_periods=None,
                 cache_derivative_batches=False, table_first=False, require_periods=False):
        """Load a frozen metric and table with the caller's validation and cache policy."""
        import jax
        from .metric_evaluator import MetricEvaluator
        from .jax_metric_evaluator import JaxMetricEvaluator
        from .eta_filtered_field import EtaFilterTable

        def read_table():
            source = EtaFilterTable.load(field_table)
            if filter_spec is _MISSING:
                raise KeyError('filter_spec')
            if (table_first or filter_spec is not None) and source.spec != filter_spec:
                if table_first:
                    raise ValueError(f"Geometry field table spec {source.spec!r} differs from requested {filter_spec!r}")
                raise ValueError('Saved field table spec differs from options')
            if field_periods is _MISSING:
                raise KeyError('field_periods')
            if (table_first or require_periods or field_periods is not None) and source.spec.nfp != field_periods:
                if table_first:
                    raise ValueError(f"Geometry field table periods {source.spec.nfp} differ from requested {field_periods}")
                raise ValueError('Saved field table nfp differs from geometry options')
            return source

        # Geometry has always validated the table before opening the metric.
        if table_first:
            source = read_table()
        if metric_cache is _MISSING:
            raise KeyError('metric_cache')
        with np.load(metric_cache, allow_pickle=False) as archive:
            metric = MetricEvaluator.from_cache_payload(archive, prefix='metric_evaluator_')
        metric = JaxMetricEvaluator.from_metric_evaluator(metric)
        if not table_first:
            source = read_table()
        self.filter_spec = source.spec
        table = source.to_jax()
        self.metric, self.table = metric, table
        self.tables = (metric, table)
        if cache_derivative_batches:
            self.derivative_batches = {}
        self.fn = jax.jit(lambda q:evaluate_frozen_field(metric,table,q))

    def derivatives(self, points):
        from .metric_derivatives import evaluate

        return evaluate(self, points, cache=getattr(self, 'derivative_batches', None))

    def volume_derivatives(self, points):
        from .metric_derivatives import evaluate

        return evaluate(self, points, True, cache=getattr(self, 'derivative_batches', None))
