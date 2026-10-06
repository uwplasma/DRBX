"""Portable adapter for the immutable P m<=3 table and frozen HSX chart."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
from . import common as c


def setup(run,n, *, physical=True, tracing=True):
    from scripts.q08_extraction_global.common import load_geometry_model
    model = load_geometry_model(Path(run)/'inputs')
    root = Path(c.read(Path(run)/'run_config.json')['canonical_root'])
    ctx,t = model.context(n,root,physical=physical,magnetic=physical and tracing)
    if not physical: return t
    name = model.__package__+'.vendor.geometry.eta_filtered_field'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,c.HERE/'eta_filtered_field.py')
        mod = importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    mod = sys.modules[name]
    if tracing:
        jm=ctx['tracer'].metric
    else:
        # CPU preparation consumes the filtered table, not the raw MAKEGRID
        # spline. Avoid rebuilding/replicating its large coefficients per worker.
        cls=importlib.import_module(model.__package__+'.vendor.geometry.jax_metric_evaluator').JaxMetricEvaluator
        jm=cls.from_metric_evaluator(ctx['evaluator'])
    metric = mod.JaxMetricView(jm,block=2048)
    table = mod.EtaFilterTable.load(Path(run)/'inputs'/c.TABLE,metric_evaluator=metric)
    if table.meta['arm_sha256'] != c.ARM or table.spec.as_dict() != dict(quantity='J*B^i',max_harmonic_per_period=3,nfp=4,samples_per_period=64):
        raise ValueError('filtered magnetic arm identity')
    def geom(points):
        p=np.asarray(points); shape=p.shape[:-1]; flat=p.reshape(-1,3)
        if np.any(flat[:,0]<=0) or np.any(flat[:,0]>table.u_max): raise ValueError('filtered evaluator domain')
        m=metric.evaluate(flat); J=m.signed_J
        if not np.isfinite(J).all() or np.any(J<=0): raise ValueError('invalid geometry Jacobian')
        B=table.flux_density(flat)/J[:,None]
        cart=np.einsum('pij,pj->pi',m.jacobian_matrix,B); mag=np.linalg.norm(cart,axis=1)
        if not np.isfinite(mag).all() or np.any(mag<=0): raise ValueError('invalid filtered field')
        return J.reshape(shape),(B/mag[:,None]).reshape(*shape,3),mag.reshape(shape)
    jac=lambda p: metric.evaluate(np.asarray(p)).jacobian_matrix
    return dict(t=t,ctx=ctx,table=table,jtable=table.to_jax(),metric=metric,geom=geom,jac=jac)
