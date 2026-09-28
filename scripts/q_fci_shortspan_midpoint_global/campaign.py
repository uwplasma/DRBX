"""Stable, explicit node-local stages for the Q numerical campaign."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from . import numerical_runner as nr
from .reuse import EXPECTED

STAGES=('preflight','pilot','global')
def write(path,obj):nr.write(path,obj)
def load(path):return json.loads(Path(path).read_text())
def plan(sample,stage,n):
    roles=[r for r in sample['roles'] if r['N']==n]
    if stage=='global':owners=list(range(EXPECTED[n][0]));size=64
    elif stage=='preflight':owners=sorted({int(r['owner']) for r in roles});size=4
    else:
        chosen=('ordinary_interior','fresh_strong_wall')
        owners=sorted({int(r['owner']) for r in roles if r['role'] in chosen});size=2
    if not owners:raise RuntimeError(f'empty {stage} N{n} stage')
    return [owners[j:j+size] for j in range(0,len(owners),size)]
def freeze(campaign,input_root,screen,preflight,rk4_check_cache=None,trace_mode='cpu_inline',rk4_steps=256):
    root=Path(campaign);root.mkdir(parents=True,exist_ok=True)
    sample=load(Path(preflight)/'sample.json');manifest={}
    for stage in STAGES:
        fresh=[]
        if stage=='pilot':fresh=[int(r['members'][0]['raw_id']) for r in sample['roles'] if r['role'] in ('ordinary_interior','fresh_strong_wall')]
        nr.freeze(input_root,root/stage,screen,preflight,fresh,rk4_check_cache,trace_mode,rk4_steps)
        manifest[stage]={str(n):plan(sample,stage,n) for n in EXPECTED}
    write(root/'dispatch.json',dict(schema='q-campaign-dispatch-v1',stages=manifest,source_design_sha256={stage:nr.sha(root/stage/'design.json') for stage in STAGES},exact_reuse_sha256=nr.sha(root/'global/exact_reuse_manifest.json'),full_owner_counts={str(n):EXPECTED[n][0] for n in EXPECTED},full_raw_counts={str(n):EXPECTED[n][1] for n in EXPECTED},pilot_selection='frozen preflight ordinary interior and strong wall at each N; fresh trace/fit',chunk_owner_max=64))
    validate_dispatch(root)
def validate_dispatch(root):
    root=Path(root);d=load(root/'dispatch.json')
    if d['schema']!='q-campaign-dispatch-v1':raise RuntimeError('wrong dispatch schema')
    for stage in STAGES:
        if nr.sha(root/stage/'design.json')!=d['source_design_sha256'][stage]:raise RuntimeError('stage design differs')
        for n in EXPECTED:
            groups=d['stages'][stage][str(n)];flat=[x for group in groups for x in group]
            if not groups or any(not group or len(group)>64 or group!=sorted(group) for group in groups) or flat!=sorted(set(flat)):raise RuntimeError('unstable or overlapping owner chunks')
            if stage=='global' and flat!=list(range(EXPECTED[n][0])):raise RuntimeError(f'N{n} global owner coverage differs')
    return d
def run(campaign,stage,n,input_root,screen,workers,host_memory_gib,worker_memory_gib):
    if stage not in STAGES:raise ValueError('explicit stage required')
    root=Path(campaign);d=validate_dispatch(root)
    groups=d['stages'][stage][str(n)]
    return nr.run_jobs(n,groups,input_root,root/stage,screen,workers,host_memory_gib,worker_memory_gib)
def validate_outputs(campaign,stage,n,require_complete=False):
    root=Path(campaign);d=validate_dispatch(root);out=root/stage
    dh=nr.sha(out/'design.json');groups=d['stages'][stage][str(n)]
    owners=[];raw=0;bytes_total=0;cpu=0.;checks=[];rss=[]
    for group in groups:
        path=nr.chunk_path(out,n,group)
        if not nr.completed(path,dh,group):raise RuntimeError(f'missing required chunk: {path}')
        r=load(path.with_suffix('.json'))
        if r['schema']!='q-numerical-owner-chunk-v2' or r['N']!=n or r['raw_count']<len(group):raise RuntimeError('chunk schema/count differs')
        owners.extend(group);raw+=r['raw_count'];bytes_total+=path.stat().st_size;cpu+=r['cpu_s'];rss.append(r['maxrss_gib']);checks.extend(x['raw'] for x in r['rk4_checks'])
    expected_checks=set(load(out/'design.json')['rk4_512_raw_ids'][str(n)])
    if stage=='global' or require_complete:
        if owners!=list(range(EXPECTED[n][0])) or raw!=EXPECTED[n][1]:raise RuntimeError('incomplete global owners/raw targets')
        if set(checks)!=expected_checks or len(checks)!=len(expected_checks):raise RuntimeError('bounded RK4-512 schedule differs')
    result=dict(stage=stage,N=n,owners=len(owners),raw_targets=raw,chunks=len(groups),bytes=bytes_total,chunk_cpu_s=cpu,max_worker_rss_gib=max(rss),rk4_512_count=len(checks),rk4_512_expected_full=len(expected_checks),complete=stage=='global' or require_complete)
    write(root/f'{stage}/N{n}/validation.json',result);return result
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','verify','run','validate','reduce'));p.add_argument('--campaign',required=True);p.add_argument('--input-root');p.add_argument('--exact-screen');p.add_argument('--preflight-cache');p.add_argument('--rk4-check-cache');p.add_argument('--stage',choices=STAGES);p.add_argument('--N',type=int,choices=(32,48,64));p.add_argument('--workers',type=int);p.add_argument('--host-memory-gib',type=float);p.add_argument('--worker-memory-gib',type=float)
    p.add_argument('--trace-mode',choices=('cpu_inline','gpu_cache'),default='cpu_inline',help='freeze only: GPU mode requires a completed gpu_trace stage before run')
    p.add_argument('--rk4-steps',type=int,choices=(64,256),default=256,help='freeze only: new numerical identity; legacy cached maps require CPU RK4-256')
    a=p.parse_args()
    if a.command=='freeze':
        if not all((a.input_root,a.exact_screen,a.preflight_cache)):p.error('freeze requires input root, exact screen and preflight cache')
        freeze(a.campaign,a.input_root,a.exact_screen,a.preflight_cache,a.rk4_check_cache,a.trace_mode,a.rk4_steps);result={'frozen':True}
    elif a.command=='verify':
        if not a.input_root or not a.exact_screen:p.error('verify requires input root and exact screen')
        d=validate_dispatch(a.campaign)
        result={stage:nr.validate(a.input_root,Path(a.campaign)/stage,a.exact_screen)['schema'] for stage in STAGES}
    elif a.command=='run':
        if not all((a.stage,a.N,a.input_root,a.exact_screen,a.workers,a.host_memory_gib,a.worker_memory_gib)):p.error('run requires stage, N, paths, worker count and memory budgets')
        result=run(a.campaign,a.stage,a.N,a.input_root,a.exact_screen,a.workers,a.host_memory_gib,a.worker_memory_gib)
    elif a.command=='validate':
        if not a.stage or not a.N:p.error('validate requires stage and N')
        result=validate_outputs(a.campaign,a.stage,a.N)
    else:
        if a.stage!='global' or not a.input_root or not a.exact_screen:p.error('reduce requires --stage global, input root and exact screen')
        from .reduce_numerical import reduce_campaign
        result=reduce_campaign(a.campaign,a.input_root,a.exact_screen)
    print(json.dumps(result,default=str))
if __name__=='__main__':main()
