from src.common import ROOT, read_json
from src.protocol import build_tasks,task_root

def main():
    tasks = build_tasks()
    print('Formal completion:',sum((task_root(t)/'metrics.json').exists() for t in tasks),'/80')
    path = ROOT/'logs'/'queue_status.json'
    if path.exists():
        status = read_json(path)
        print('Updated UTC:',status['updated_utc'])
        print('Deadline UTC:',status['end_utc'])
        print('Active:',status['active'])
        print('Pending:',len(status['pending_ids']))
        print('Reserved physical GPUs: 6,7')

if __name__=='__main__':
    main()
