import argparse
import os
import subprocess
import sys
from datetime import datetime,timezone
from src.common import ROOT, write_json, read_json
from src.protocol import build_tasks, task_root, protocol_signature
from src.model import VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    if args.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') not in [str(x) for x in range(6)]:
        raise ValueError('Smoke must see one permitted physical GPU, e.g. CUDA_VISIBLE_DEVICES=0')
    tasks = [next(t for t in build_tasks() if t['variant']==v and t['regime']=='warm'
                  and t['feature']=='compact' and t['repeat']==0) for v in VARIANTS]
    tasks += [next(t for t in build_tasks() if t['variant']=='full' and t['regime']==r
                   and t['feature']==f and t['repeat']==0)
              for r,f in [('warm','raw'),('cold_cell','compact')]]
    for task in tasks:
        subprocess.run([sys.executable,'-m','src.train','--task',task['id'],'--mode','smoke',
                        '--device',args.device,'--epochs','2','--max-samples','256'],cwd=ROOT,check=True)
        result = read_json(task_root(task,'smoke')/'metrics.json')
        if result['publication_eligible'] or result['completed_epochs']!=2:
            raise RuntimeError('Smoke isolation failure')
    write_json(ROOT/'logs'/'smoke_pass.json',{'passed':True,'protocol_signature':protocol_signature(),
               'tests':len(tasks),'publication_eligible':False,
               'finished_utc':datetime.now(timezone.utc).isoformat()})
    print('SMOKE PASS: all 10 architectures + raw input + cold-cell path; not publication results.',flush=True)


if __name__=='__main__':
    main()
