"""Deterministic physical-volume reduction and independent completion replay."""
import csv,json
import numpy as np
import campaign as c


def reduction():
    data=[]
    for n in (32,48,64):
        c.validate(n);total=c.empty_stats()
        for b,owners in enumerate(c.plans(n)['global_']):
            path=c.OUT/'chunks'/f'N{n}'/f'block_{b:06d}.npz'
            with np.load(path) as z:
                for key in ('sum2','signed_sum','reference_sum2','volume','count'):total[key]+=z[key]
                improve=(z['maximum']>total['maximum'])|(total['max_owner']<0)
                total['max_owner']=np.where(improve,z['max_owner'],total['max_owner'])
                total['maximum']=np.maximum(total['maximum'],z['maximum'])
        if total['count'][0]!=c.design()['owner_counts'][str(n)] or np.any(total['volume']<=0):raise ValueError('reduced coverage')
        data.append(total)
    return {key:np.stack([x[key] for x in data]) for key in data[0]}


def main(validate_only=False):
    d=c.design();identity=c.digest(d);totals=reduction();path=c.OUT/'totals.npz'
    if validate_only:
        with np.load(path) as z:
            if set(z.files)!=set(totals):raise ValueError('totals schema')
            for key,value in totals.items():
                if not np.array_equal(z[key],value):raise ValueError('independent chunk reduction '+key)
        receipt=json.loads((c.OUT/'analysis_receipt.json').read_text())
        if receipt['identity']!=identity:raise ValueError('analysis identity')
        for name,h in receipt['artifacts'].items():
            if c.sha(c.OUT/name)!=h:raise ValueError('analysis artifact '+name)
        c.write_json(c.OUT/'completion.json',dict(identity=identity,passed=True,complete_coverage=True,independent_chunk_reduction=True,artifacts=receipt['artifacts']))
        return
    # Paired baseline must replay the completed, unchanged original campaign.
    oldpath=c.HERE/'inputs/evidence/q07_material_global_20260930/totals.npz'
    cols=np.array([0,1,2,5,6,7,10,11,12]); replay={}
    with np.load(oldpath) as old:
        if not np.array_equal(totals['count'],old['count']):raise ValueError('original global count replay')
        np.testing.assert_allclose(totals['volume'],old['volume'],rtol=1e-14,atol=1e-14)
        vol=totals['volume'][:,:,None,None,None,None]
        for key in ('sum2','maximum','signed_sum'):
            a=totals[key][...,:9];b=np.take(old[key],cols,axis=-1)
            if key=='sum2':a=np.sqrt(a/vol);b=np.sqrt(b/vol)
            if key=='signed_sum':a=a/vol;b=b/vol
            replay[key]=float(abs(a-b).max())
            if replay[key]>1e-8:raise ValueError('original global replay '+key+' '+str(replay[key]))
    c.write_json(c.OUT/'original_global_replay.json',dict(passed=True,identity=identity,maximum_absolute_norm_difference=replay,tolerance=1e-8))
    c.save_arrays(path,totals)
    volume=totals['volume'][:,:,None,None,None,None]
    rms=np.sqrt(totals['sum2']/volume)
    ref=np.sqrt(totals['reference_sum2']/totals['volume'][:,:,None,None])
    orders=np.full((2,)+rms.shape[1:],np.nan)
    for i in range(2):
        ok=(rms[i]>0)&(rms[i+1]>0)
        orders[i][ok]=np.log(rms[i][ok]/rms[i+1][ok])/np.log((48,64)[i]/(32,48)[i])
    def clean(x):
        a=np.asarray(x,dtype=object);a[~np.isfinite(np.asarray(x))]=None;return a.tolist()
    analysis=dict(identity=identity,resolutions=[32,48,64],regions=c.REGIONS,metrics=c.METRICS,
        kinds=d['kinds'],cases=d['cases'],terms=d['terms'],
        rms_axes='resolution,region,metric,boundary,case,term',rms=rms.tolist(),
        maximum=totals['maximum'].tolist(),maximum_owner=totals['max_owner'].tolist(),
        signed_integral=totals['signed_sum'].tolist(),continuum_rms=ref.tolist(),relative_rms=clean(np.divide(rms,ref[:,:,None,None,:,:],out=np.full_like(rms,np.nan),where=ref[:,:,None,None,:,:]>0)),
        orders_axes='interval,region,metric,boundary,case,term',orders=clean(orders),
        norm='physical-volume-weighted complete-owner RMS; maxima retain owner IDs',
        scope=d['scientific_gate'],case_class=['constant','smooth']+['wave_stress']*(c.m.NF-2))
    c.write_json(c.OUT/'analysis.json',analysis)
    count=0
    with (c.OUT/'orders.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['region','metric','bc','case','term','RMS32','RMS48','RMS64','p32_48','p48_64','max32','max48','max64'])
        for ri,region in enumerate(c.REGIONS):
            for ei,error in enumerate(c.METRICS):
                for bi,(bc,pk) in enumerate(c.m.KINDS):
                    for fi,case in enumerate(c.m.DESIGNS):
                        for ti,term in enumerate(c.kernel.TERMS):
                            row=(ri,ei,bi,fi,ti);po=orders[(slice(None),)+row]
                            w.writerow([region,error,''.join(bc)+'/phi'+pk,case['name'],term,*rms[(slice(None),)+row],*[float(v) if np.isfinite(v) else '' for v in po],*totals['maximum'][(slice(None),)+row]])
                            count+=1
    lines=['# Q07 global static qualification','',
        'Complete N32/N48/N64, 18 state sets, four boundary combinations, original/balanced density and temperature terms. Physical-volume complete-owner norms; raw members are never sampled away. See analysis.json and orders.csv for all regions, maxima, signed integrals and orders.','',
        'Centered/correction/combined outputs remain separate. Smooth controls and short-wave stress controls are not pooled into one gate. Finite-value/replay/coverage success is distinct from scientific order acceptance. Current/phi/SAT integration, exterior crossings, evolved stability and production remain unqualified. Velocity and diffusion are unchanged; neither is recomputed.','',
        '## Smooth global combined original and balanced RMS','',
        '| term | BC | N-O32 | N-O48 | N-O64 | p32-48 | p48-64 | N-R32 | N-R48 | N-R64 | p32-48 | p48-64 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ti in (6,7,8,15,16,17):
        for bi in (0,1):
            cells=[]
            for ei in (0,2):cells.extend([*rms[:,0,ei,bi,1,ti],*orders[:,0,ei,bi,1,ti]])
            lines.append('| '+c.kernel.TERMS[ti]+' | '+('D' if bi==0 else 'N')+' | '+' | '.join(f'{x:.6g}' if np.isfinite(x) else '—' for x in cells)+' |')
    (c.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    names=('totals.npz','analysis.json','orders.csv','report.md','original_global_replay.json')
    c.write_json(c.OUT/'analysis_receipt.json',dict(identity=identity,orders_rows=count,artifacts={p:c.sha(c.OUT/p) for p in names},passed=True))
