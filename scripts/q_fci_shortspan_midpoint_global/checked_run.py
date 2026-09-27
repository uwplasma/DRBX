"""Strict input/method gate for resuming exact-owner chunks."""
import argparse,json
from pathlib import Path
from . import exact_screen as x

def validate(input_root,output):
    out=Path(output);design=x.check_design(out,input_root)
    expected=dict(fields=list(x.FIELDS),outer_spans=['h/8','h/4'],seed_order='span,axis,negative-positive',reference_step=x.REF,reference_half=x.HALF,chunk_points=x.CHUNK_POINTS,owner_batch=x.OWNER_BATCH,projection='complete owner frozen raw-volume')
    if design['schema']!='q-shortspan-pr-v1' or design['method']!=expected:raise RuntimeError('incompatible method identity')
    for entry in design['inputs']['files']:
        p=Path(input_root)/entry['path']
        if x.sha(p)!=entry['sha256']:raise RuntimeError(f'input hash mismatch: {p}')
    return design
def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--output',required=True);p.add_argument('--N',type=int,choices=(32,48,64),required=True);p.add_argument('--workers',type=int,choices=(1,2),default=1);a=p.parse_args()
    validate(a.input_root,a.output);x.run(a.N,a.input_root,a.output,a.workers)
if __name__=='__main__':main()
