"""Intact cubic units from sharded geometry scheduling and canonical collection tasks/owner_jobs, using the placed kernels."""
from pathlib import Path

import numpy as np

from ._geometry_worker import geometry_rays
from . import owner_reductions as owners
from .build import clip_leaf, clipping_context
from .compiled_polygons import Polygons
from .context import context
from .indexed_output import Writer
from .numerics import fit_task, integrate, metric_task, observe, source_plan
from .production_splice import ProductionSplice
from .storage import FrameWriter, arrays, load


class Jobs:
    def __init__(self,root,options,config):
        self.root = root
        self.options = options
        self.m = config
        self.files = config['files']
        self.ns, self.ev = context(config)
        self.rays = {}
        self.polygons = Polygons() if options.compiled_clipping else None
        self.observation = None
        self.clip_data = None
        self.boundary = None
        self.plan = None

    def ray_request(self,plane):
        if plane not in self.rays:
            self.rays[plane] = geometry_rays(self.options,self.ev)
        return self.rays[plane]

    def production(self):
        if self.boundary is None:
            self.boundary = ProductionSplice(self.files,self.ns['NN'],self.options.geometry_reference_key_table)
        return self.boundary

    def execute(self,task,private,target):
        kind = task['kind']
        if kind in {'plane','polygon','physical_wall'}:
            result,files = self.owner(task,private,target)
            return result,files,0
        before = sum(rays.traced for rays in self.rays.values())
        if kind=='census':
            values,meta = owners.census(self.m,self.ns,task['plane'],Path(task['source']))
        elif kind=='physical_count':
            values,meta = owners.physical_count(*task['bounds'])
        else:
            values,meta = self.source(task)
        FrameWriter(private).write('.',values,meta)
        return meta,['complete.json','geometry.gds'],sum(rays.traced for rays in self.rays.values())-before

    def source(self,task):
        m,ns,ev = self.m,self.ns,self.ev
        kind = task['kind']
        if kind=='fit':
            return fit_task(m,task['keys']),dict(stage='fits')
        k = task['plane']
        if kind=='source_plan':
            sign,order = task['sign'],task['order']
            cuts = None
            if order==8:
                base = self.root/task['base']/f'q3_{sign}'
                cut = np.zeros(ns['P'],bool)
                for leaf in sorted(base.glob('leaf_*')):
                    z = arrays(leaf,['owner','hit'])
                    cut[z['owner'][z['hit']]] = True
                cuts = np.flatnonzero(cut)
            plan = source_plan(ns,k,order,cuts)
            if cuts is not None:
                plan['cuts'] = cuts
            return plan,dict(order=order,sign=sign)
        if kind=='integrate':
            sign,order = task['sign'],task['order']
            bounds = task['bounds']
            values = integrate(ns,ev,self.ray_request(k),arrays(self.root/task['plan']),sign,order,bounds)
            return values,dict(bounds=bounds)
        if kind=='observe':
            return observe(ns,self.ray_request(k),k,m['angular_subdivisions'],m['clip_bisections']),dict(plane=k)
        bounds = task['bounds']
        if kind=='clip':
            path = self.root/task['observation']
            if path!=self.observation:
                self.clip_data = clipping_context(arrays(path),self.polygons)
                self.observation = path
            return clip_leaf(self.clip_data,bounds),dict(bounds=bounds)
        if kind=='metric':
            return metric_task(ns,ev,k,arrays(self.root/task['clip'])),dict(bounds=bounds)
        raise ValueError(f'Unknown geometry task: {kind}')

    def owner(self,task,private,target):
        m,b = self.m,self.files
        kind = task['kind']
        if self.plan is None:
            self.plan = load(self.root/'assembly_plan.json')
        part = target.name
        production = self.production()
        writer = Writer(private,m,self.plan,published_root=target)
        try:
            writer.part_start(part)
            if kind=='plane':
                plane = task['plane']
                census = arrays(self.root/'census'/f'p{plane:03d}')
                owners.plane(m,self.plan,writer,Path(task['source']),plane,census,production)
            elif kind=='polygon':
                owners.polygon(m,self.plan,writer,*task['bounds'],production)
            else:
                owners.physical_wall(m,self.plan,writer,*task['bounds'],task['prefix'],production)
            writer.db.commit()
            writer.q.commit()
            spec = writer.spec
        finally:
            writer.close()
        files = ['index.sqlite','parallel_geometry_spans.sqlite']
        if spec:
            files.append('geometry.gds')
        return spec,files
