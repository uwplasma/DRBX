"""One node, one controller; same arguments resume checked stages."""
import shlex,subprocess,sys,time
from . import common as c

def main():
    args=c.parser(stages=False).parse_args();c.cpu_env()
    flags=sys.argv[1:];filtered=[];i=0
    while i<len(flags):
        if flags[i]=='--through':i+=2;continue
        if flags[i]=='--dry-run':i+=1;continue
        filtered.append(flags[i]);i+=1
    commands=[[sys.executable,'-m',__package__+'.campaign',stage,*filtered] for stage in c.STAGES[:c.STAGES.index(args.through)+1]]
    if args.dry_run:
        for cmd in commands:print(shlex.join(cmd))
        return
    with c.lock(args.run,'controller'):
        for cmd in commands:
            stage=cmd[3];log=args.run/'logs'/f'stage_{stage}.log';log.parent.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
            with log.open('a') as f:
                process=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=c.cpu_env())
            c.write(args.run/'receipts'/f'controller_{stage}.json',dict(command=cmd,exit_code=process.returncode,seconds=time.perf_counter()-tick))
            if process.returncode:raise SystemExit(f'{stage} failed; see {log}')
            print(f'{stage} command finished; inspect its convergence receipt',flush=True)
if __name__=='__main__':main()
