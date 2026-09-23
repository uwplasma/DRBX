"""Bounded actual-HSX RK4 check; archived DOP853 observations are immutable controls."""
from pathlib import Path
import argparse,importlib.util,sys,time,json,numpy as np
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from scripts.q_fci_return_campaign import campaign as c

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();W=a.workspace;out=a.output;c.unpack(out)
 spec=importlib.util.spec_from_file_location('scripts.q_fci_return_campaign.dop853_reference',out/'dop853_reference.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 hist=W/'work/parallel_fci_local_return_refinement_20260923';parent=W/'work/parallel_fci_local_return_20260923';report={'resolutions':{},'scope':'N32 three archived complete owners; N64 one ordinary complete owner; bounded axis/wall/seam trace comparisons. No full preflight or global run.'}
 for n in (32,64):
  c.initialize(n,str(W),str(out),trace_capacity=96);m=c.NUM;ctx=c.CTX
  summary=c.read(hist/f'summary_N{n}.json');groups=c.read(parent/'selection.json' if n==32 else hist/f'selection_N{n}.json')['groups'];regions=('ordinary','agglomerated','transition') if n==32 else ('ordinary',)
  rows={r:[] for r in regions};catalogue={};size=2*n**3
  for group in groups:
   if group['region'] not in rows:continue
   path=parent/'rows'/f"{group['id']}.npz" if n==32 else hist/'rows'/f'N{n}'/f"{group['id']}.npz"
   with np.load(path) as z:
    for j,ij in enumerate(z['ij']):
     rid=int(m.row_ids(n,[[int(ij[0]),int(ij[1]),group['k'],group['d']]])[0]);rows[group['region']].append(rid)
     catalogue[rid]={key:z[name][j] for key,name in {'source':'source_points','endpoint':'endpoint_points','F':'source_F','Z':'Z','g_sec':'g_sec','numerical':'g_num'}.items()}
  data={'valid':np.zeros(size,bool)}
  for key,shape in {'source':(4,3),'endpoint':(4,3),'F':(4,),'Z':(),'g_sec':(4,),'numerical':(4,)}.items():data[key]=np.zeros((size,)+shape)
  for rid,row in catalogue.items():
   data['valid'][rid]=True
   for key,value in row.items():data[key][rid]=value
  ids=np.array(sorted(catalogue));plan=c.trace_plan(ctx,ids)
  refs=np.load(hist/f'parent_endpoint_N{n}.npz');diags=summary['support_diagnostics']['geometry_96'];fi=[i for i,d in enumerate(diags) if d['region'] in regions]
  owners=summary['selection']['owners'];oi=[owners.index(summary['selection']['targets'][region]['owner']) for region in regions];B=refs['incidence'][np.ix_(oi,fi)];volume=refs['volume'][oi];reference=refs['reference'][oi];oldaction=refs['numerical_action'][oi]
  def fluxes(module):
   flux=[];selected=[]
   for i in fi:
    d=diags[i];rr=np.array([rows[d['region']][j] for j in d['row_indices']]);f,g,_,_,_=module.return_map(ctx,d['face'],data,frozen_ids=rr);flux.append(f);selected.append(g)
   return np.array(flux),np.array(selected)
  oldflux,_=fluxes(old);fastflux,_=fluxes(m);np.testing.assert_allclose(fastflux,oldflux,atol=1e-14,rtol=1e-9)
  rec={'rows':len(ids),'owners':len(oi),'faces':len(fi),'optimized_map_replay_max':float(abs(fastflux-oldflux).max()),'steps':{}}
  # The lean geometry primitive must reproduce all consumed full-evaluator quantities.
  points=data['source'][ids[:8]].reshape(-1,3);slow=old.base(ctx,points);fast=m.base(ctx,points)
  for x,y in zip(slow,fast):np.testing.assert_allclose(x,y,atol=1e-13,rtol=1e-12)
  actions={};rms=lambda v:np.sqrt(np.sum(volume[:,None]*v*v,axis=0)/volume.sum())
  for steps in (64,128):
   ctx['config']['policy']['trace_substeps']=steps;t=time.monotonic();endpoint_error=0.;fit_seconds=0.;numerical_error=0.
   for unit in plan:
    z=m.trace_rows(ctx,unit);assert np.all(z['valid']);fit_seconds+=float(z['fit_seconds'])
    oldend=np.array([catalogue[i]['endpoint'] for i in unit]);endpoint_error=max(endpoint_error,float(abs(m.regular(z['endpoint'].reshape(-1,3),0)-m.regular(oldend.reshape(-1,3),0)).max()))
    numerical_error=max(numerical_error,float(abs(z['numerical']-np.array([catalogue[i]['numerical'] for i in unit])).max()))
    for key in data:
     data[key][unit]=z[key]
   flux,g=fluxes(m);action=B@flux/volume[:,None];actions[steps]=action
   rec['steps'][str(steps)]={'seconds':time.monotonic()-t,'fit_seconds':fit_seconds,'endpoint_regular_max_vs_dop853':endpoint_error,'observation_max_vs_dop853':numerical_error,'action_rms':rms(action-reference),'action_difference_rms_vs_dop853':rms(action-oldaction),'difference_fraction_vs_spatial_error':rms(action-oldaction)[:3]/rms(oldaction-reference)[:3]}
   c.save(out/f'N{n}_rk4_{steps}_actions.npz',action=action,flux=flux,g_sec_flux=g,reference=reference,volume=volume,incidence=B)
   print('N',n,'steps',steps,rec['steps'][str(steps)],flush=True)
  rec['step_doubling_fraction']=rms(actions[64]-actions[128])[:3]/rms(oldaction-reference)[:3]
  assert np.max(rec['step_doubling_fraction'])<.01
  assert max(rec['steps'][str(s)]['difference_fraction_vs_spatial_error'].max() for s in (64,128))<.01
  if n==64:
   edge={};ctx['trace_capacity']=96
   for d in (-1,1):
    keys=[[i,j,0,d] for i in (0,1,n-2,n-1) for j in (0,n-1)];rr=m.row_ids(n,keys);results=[]
    for steps in (64,128):
     ctx['config']['policy']['trace_substeps']=steps;results.append(m.trace_rows(ctx,rr))
    x,y=results;np.testing.assert_array_equal(x['valid'],y['valid']);v=x['valid'];err=float(abs(m.regular(x['endpoint'][v].reshape(-1,3),0)-m.regular(y['endpoint'][v].reshape(-1,3),0)).max())
    edge[str(d)]={'rows':len(rr),'valid':int(v.sum()),'endpoint_regular_difference':err};assert err<1e-6
   rec['axis_wall_seam_step_doubling']=edge
  report['resolutions'][str(n)]=rec;c.write(out/'validation.json',report)
 report['passed']=True;report['source']=c.source_identity();c.write(out/'validation.json',report)
 print('Bounded RK4 qualification checks passed',flush=True)
if __name__=='__main__':main()
