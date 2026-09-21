"""Bounded actual-HSX serial/parallel/recovery correctness check, not scaling."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import campaign


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=campaign.REPO.parent)
    parser.add_argument('--campaign-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    prepare=(args.campaign_output/'N32.prepare.npz').resolve()
    plans={}
    for name,workers in (('serial',1),('parallel',2)):
        root=args.output/name
        base=argparse.Namespace(input_root=args.input_root,output=root)
        kw=campaign.namespace(base,resolution=32,coverage='test',plan=root/'plan.json')
        kw.prepare=prepare
        campaign.runner.command_plan(kw);plans[name]=kw.plan
        kw.workers=workers;kw.max_tasks_per_worker=64;kw.fail_unit=None
        campaign.runner.command_execute(kw)
        campaign.runner.command_validate(kw)
    compare=campaign.namespace(argparse.Namespace(input_root=args.input_root,output=args.output))
    compare.plan=plans['serial'];compare.prepare=prepare
    compare.left=args.output/'serial';compare.right=args.output/'parallel';compare.report=args.output/'comparison.json'
    if campaign.runner.command_compare(compare): raise RuntimeError('serial/parallel mismatch')
    # Delete one owned test checkpoint to exercise partial resume.
    plan=json.loads(plans['parallel'].read_text())
    victim=campaign.runner._chunk_path(compare.right,32,plan['units'][-1])
    victim.unlink()
    campaign.runner.command_execute(kw)
    receipt=json.loads((compare.right/'N32.parallel-receipt.json').read_text())
    computed=sum(x['status']=='computed' for x in receipt['results'])
    if computed!=1: raise RuntimeError('partial resume failed')
    campaign.runner.command_validate(kw)
    compare.report=args.output/'recovery-comparison.json'
    if campaign.runner.command_compare(compare): raise RuntimeError('recovery mismatch')
    campaign.runner._atomic_json(args.output/'summary.json',{
        'passed':True,'scientific_accuracy_claim':False,'scaling_study':False,
        'resolution':32,'units':len(plan['units']),'recomputed_on_resume':computed,
        'arrays_and_complete_sample_cell_actions_equal':True})
    return 0


if __name__=='__main__': raise SystemExit(main())
