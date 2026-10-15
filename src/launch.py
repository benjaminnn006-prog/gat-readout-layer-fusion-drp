from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from src.common import ROOT, read_json, write_json, file_lock
from src.protocol import config, build_tasks, protocol_signature, task_root
from src.prepare import verify_cache
from src.preflight import check, gpu_status

STOP = False


def request_stop(signum,frame):
    global STOP
    STOP = True
    print('Scheduler stopping: send graceful stop to our own children only.',flush=True)


def estimate_seconds(task,records):
    same = [r['wall_seconds'] for r in records if r['task']['variant']==task['variant']
            and r['state']=='complete']
    if same:
        return max(same)*1.2
    return 3000.0  # conservative dispatch allowance; not a measured promise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours',type=float,default=15)
    parser.add_argument('--reserve-minutes',type=float,default=45)
    parser.add_argument('--gpus',default='0,1,2,3,4,5')
    parser.add_argument('--busy-memory-mib',type=int,default=2048)
    args = parser.parse_args()
    cfg = config()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus)!=len(set(gpus)) or not gpus or not set(gpus)<=set(cfg['allowed_physical_gpus']):
        raise ValueError('Only a unique subset of physical GPUs 0..5 is allowed; 6,7 always reserved')
    if args.hours<=0 or args.reserve_minutes<0 or args.reserve_minutes>=args.hours*60:
        raise ValueError('Invalid time budget')
    check(True)
    verify_cache()
    smoke = read_json(ROOT/'logs'/'smoke_pass.json')
    sig = protocol_signature()
    if not smoke['passed'] or smoke['protocol_signature']!=sig:
        raise ValueError('Pass smoke tests on this frozen package before launching formal runs')
    tasks = build_tasks()
    with file_lock(ROOT/'logs'/'.scheduler.lock'):
        active, records, completed = {}, [], []
        pending = []
        for task in tasks:
            location = task_root(task)/'metrics.json'
            if location.exists():
                result = read_json(location)
                if result['protocol_signature']!=sig or not result['publication_eligible']:
                    raise ValueError('Existing formal result is not compatible')
                completed.append(task['id'])
            else:
                pending.append(task)
        end = time.time()+args.hours*3600
        dispatch_end = end-args.reserve_minutes*60
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        hard_stop_sent = False
        while pending or active:
            now = time.time()
            if STOP or now>=end:
                if not hard_stop_sent:
                    for item in active.values():
                        item['process'].terminate()
                    hard_stop_sent = True
            for gpu,item in list(active.items()):
                rc = item['process'].poll()
                if rc is None:
                    continue
                item['log'].close()
                task = item['task']
                location = task_root(task)/'metrics.json'
                success = rc==0 and location.exists()
                records.append({'task':task,'gpu':gpu,'returncode':rc,
                                'state':'complete' if success else 'failed_or_interrupted',
                                'wall_seconds':time.time()-item['start']})
                if success:
                    completed.append(task['id'])
                print(f'FINISH GPU={gpu} task={task["id"]} rc={rc}',flush=True)
                del active[gpu]
                # Never auto-repeat a scientifically broken task; rerun launch after fixing runtime issues.
            if not STOP and now<dispatch_end and pending:
                status = {g['index']:g for g in gpu_status()}
                for gpu in gpus:
                    if gpu in active or not pending:
                        continue
                    # Do not kill or take over other users' workloads. Desktop graphics fit below threshold.
                    if status[gpu]['used_mib']>args.busy_memory_mib or status[gpu]['utilization']>20:
                        continue
                    task = pending[0]
                    if now+estimate_seconds(task,records)>dispatch_end:
                        continue
                    pending.pop(0)
                    env = os.environ.copy()
                    env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4',
                                'MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
                    folder = ROOT/'logs'/'formal'/task['group']
                    folder.mkdir(parents=True,exist_ok=True)
                    log = (folder/f'{task["id"]}.log').open('a',encoding='utf-8')
                    command = [sys.executable,'-m','src.train','--task',task['id'],'--device','cuda:0']
                    proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    active[gpu] = {'task':task,'process':proc,'log':log,'start':time.time()}
                    print(f'START physical GPU={gpu}, logical cuda:0, {task["id"]}',flush=True)
            state = {'protocol_signature':sig,'total':80,'completed':len(completed),
                     'completed_ids':completed,'pending_ids':[t['id'] for t in pending],
                     'active':{str(g):{'task':a['task']['id'],'pid':a['process'].pid} for g,a in active.items()},
                     'finished_this_invocation':records,'reserved_physical_gpus':[6,7],
                     'end_utc':datetime.fromtimestamp(end,timezone.utc).isoformat(),
                     'updated_utc':datetime.now(timezone.utc).isoformat()}
            write_json(ROOT/'logs'/'queue_status.json',state)
            if (STOP or now>=dispatch_end) and not active:
                break
            if not pending and not active:
                break
            time.sleep(10)
        collected = subprocess.run([sys.executable,'-m','src.collect'],cwd=ROOT)
        if (ROOT/'summary'/'audit.json').exists():
            subprocess.run([sys.executable,'-m','src.pack_results'],cwd=ROOT,check=True)
        print(f'FINISHED: {len(completed)}/80 complete. Incomplete experiment is clearly marked.',flush=True)
        if len(completed)!=80 or collected.returncode!=0:
            sys.exit(2)


if __name__=='__main__':
    main()
