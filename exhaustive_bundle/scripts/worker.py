import argparse,json,time,os,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import engine as e

def load_rows(path,bundle,allowed):
    rows={};p=e.Path(path)
    if not p.exists():return rows
    data=p.read_bytes();lines=data.splitlines(keepends=True);good=0
    for i,line in enumerate(lines):
        try:
            r=json.loads(line);e.validate_record(r,bundle);assert r['task_id'] in allowed
        except Exception:
            if i==len(lines)-1 and not line.endswith(b'\n'):
                # Recover only an interrupted, unterminated final write, never an invalid complete row.
                with p.open('r+b') as f:f.truncate(good)
                break
            raise
        if r['task_id'] in rows and e.scientific_record(rows[r['task_id']])!=e.scientific_record(r):raise RuntimeError('Conflicting resumed result')
        rows[r['task_id']]=r;good+=len(line)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assignment',required=True);ap.add_argument('--output-root');ap.add_argument('--smoke-report',required=True)
    ap.add_argument('--max-tasks',type=int,help='Bounded validation run; resume using the same command without this option')
    args=ap.parse_args();bundle=e.verify();assignment=e.read(args.assignment);smoke=e.read(args.smoke_report)
    assert assignment['bundle_id']==smoke['bundle_id']==bundle and smoke['passed']
    node=e.node_name(assignment['node']);assert node==smoke['node']
    workers=assignment['workers'];assert workers==smoke['recommended_workers'] and workers<=e.resources()['worker_cap']
    base=e.Path(args.output_root or assignment['output_root_hint'])/node;out=base/'results';out.mkdir(parents=True,exist_ok=True)
    lock=base/'worker.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Worker lock exists: {lock}. Confirm no worker is running before removing a stale lock.')
    os.write(fd,str(os.getpid()).encode());os.close(fd);remaining=args.max_tasks
    try:
        for pack in assignment['packs']:
            pairs=pack['tasks'];allowed={e.task_id(f,m) for f,m in pairs};path=out/f"{pack['pack_id']}.jsonl"
            done=load_rows(path,bundle,allowed);pending=[e.make_task(f,m) for f,m in pairs if e.task_id(f,m) not in done]
            if remaining is not None:pending=pending[:max(0,remaining)]
            failures=[];start=time.perf_counter()
            with path.open('a',encoding='utf-8',newline='\n') as log:
                with ProcessPoolExecutor(max_workers=workers,initializer=e.initialize) as pool:
                    fs={pool.submit(e.evaluate,t):t for t in pending}
                    for future in as_completed(fs):
                        task=fs[future]
                        try:
                            r=future.result();e.validate_record(r,bundle)
                            log.write(json.dumps(r,separators=(',',':'))+'\n');log.flush();os.fsync(log.fileno());done[r['task_id']]=r
                        except Exception as exc:
                            error={'task':task,'node':node,'error':repr(exc),'traceback':traceback.format_exc()};failures.append(error)
                            with (base/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(error)+'\n')
                        if len(done)%100==0:print(pack['pack_id'],len(done),'/',len(pairs),flush=True)
            if remaining is not None:remaining-=len(pending)
            complete=set(done)==allowed and not failures
            e.write(out/f"{pack['pack_id']}.status.json",{'bundle_id':bundle,'node':node,'pack_id':pack['pack_id'],'complete':complete,'completed_tasks':len(done),'expected_tasks':len(pairs),'result_sha256':e.digest(path),'wall_seconds_this_attempt':time.perf_counter()-start,'failures_this_attempt':len(failures)})
            if failures:raise RuntimeError('Task failures recorded; correct environment and resume. Failed tasks are not marked complete.')
            if remaining is not None and remaining<=0:break
    finally:lock.unlink(missing_ok=True)
    print('WORKER_FINISHED',node,flush=True)
if __name__=='__main__':main()
