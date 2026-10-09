"""Node-local subprocess scheduler: exclusive core slots and independent writers."""
import os,signal,subprocess,sys,time
from . import common as c

def key(job):return '_'.join(str(job.get(k) if job.get(k) is not None else 'all') for k in ('action','operator','bc','seed'))
def run(args,jobs,*,finished=None,deadline=None):
    admission=c.admission(args);pending=list(jobs);live={};free=list(range(admission['effective_concurrency']));records=[];failures=[];skipped=[]
    def launch(job,slot):
        name=key(job);path=args.run/'jobs'/f'{name}.json';payload=vars(args).copy();payload.update(job,core_index=slot)
        remaining=deadline-time.monotonic() if deadline is not None else args.stage_seconds
        payload['stage_seconds']=min(args.stage_seconds,remaining)
        c.write(path,payload);log=args.run/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        stream=log.open('a');command=[sys.executable,'-m',__package__+'.worker','--job',str(path)]
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=c.cpu_env(),start_new_session=True)
        live[process.pid]=dict(process=process,stream=stream,job=job,name=name,core_index=slot,started=time.time(),monotonic=time.monotonic(),limit=payload['stage_seconds'],command=command,log=str(log.relative_to(args.run)),killed=False,budget_timeout=False)
    while pending or live:
        while pending and free:
            if deadline is not None and deadline-time.monotonic()<=.1:
                skipped.extend(pending);pending=[];break
            launch(pending.pop(0),free.pop(0))
        for pid,item in list(live.items()):
            proc=item['process'];elapsed=time.monotonic()-item['monotonic'];marker=args.run/'recovery'/f'killed_{item["name"]}.json'
            if getattr(args,'test_kill_job',None)==item['name'] and not marker.exists() and elapsed>.3 and proc.poll() is None:
                # Signal only a process launched and tracked by this scheduler.
                proc.kill();item['killed']=True;c.write(marker,dict(pid=pid,scope='bounded scheduler failure control'))
            if elapsed>item['limit']+5 and proc.poll() is None:
                proc.kill();item['killed']=True;item['budget_timeout']=True
            code=proc.poll()
            if code is None:continue
            item['stream'].close();free.append(item['core_index']);free.sort();del live[pid]
            record=dict(job=item['job'],core_index=item['core_index'],affinity_supported=hasattr(os,'sched_setaffinity'),pid=pid,
                started=item['started'],finished=time.time(),seconds=elapsed,exit_code=code,killed=item['killed'],budget_timeout=item['budget_timeout'],command=item['command'],log=item['log'])
            ack=args.run/'receipts'/f'job_ack_{item["name"]}.json'
            if ack.exists():
                acknowledged=c.read(ack)
                if acknowledged.get('pid')==pid:record['worker']=acknowledged
            c.write(args.run/'receipts'/f'process_{item["name"]}.json',record);records.append(record)
            budget_end=item['budget_timeout'] and item['job']['action'] in {'spectrum','analyses'}
            if code and not budget_end:failures.append(record)
            if finished is not None and (code==0 or budget_end):pending.extend(finished(item['job'],records) or [])
        if live:time.sleep(.05)
    result=dict(**admission,jobs=records,skipped=skipped,execution_errors=failures)
    c.write(args.run/'receipts'/f'scheduler_{getattr(args,"stage","jobs")}.json',result)
    if failures:raise RuntimeError(f'{len(failures)} independent child execution errors; other checkpoints retained')
    return result
