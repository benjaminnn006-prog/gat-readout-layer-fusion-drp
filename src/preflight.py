import argparse
import shutil
import subprocess
import sys
import torch
from src.common import ROOT, write_json, environment
from src.protocol import config, verify_data, build_tasks
from src.model import MedicalGAT, VARIANTS


def gpu_status():
    command = ['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
               '--format=csv,noheader,nounits']
    out = subprocess.check_output(command,text=True)
    records = []
    for line in out.strip().splitlines():
        index,uuid,name,used,total,util = [s.strip() for s in line.split(',')]
        records.append({'index':int(index),'uuid':uuid,'name':name,
                        'used_mib':int(used),'total_mib':int(total),'utilization':int(util)})
    return records


def check(require_cuda=True):
    cfg = config()
    hashes = verify_data()
    env = environment()
    if not env['torch'].startswith('2.6.0') or env['torch_geometric']!='2.6.1' or env['sklearn']!='1.6.1':
        raise RuntimeError('Use the tested gcmg-drp runtime: torch 2.6.0, PyG 2.6.1, sklearn 1.6.1; do not silently mix runtimes')
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    disk = shutil.disk_usage(ROOT)
    if disk.free < 8*1024**3:
        raise RuntimeError('At least 8 GiB free storage required; 15 GiB recommended')
    counts = {}
    for variant in VARIANTS:
        model = MedicalGAT(128,variant)
        counts[variant] = sum(p.numel() for p in model.parameters())
    capacity = MedicalGAT(128,'capacity')
    report = {'environment':env,'data_sha256':hashes,'formal_tasks':len(build_tasks()),
              'parameter_counts_compact':counts,'allowed_physical_gpus':cfg['allowed_physical_gpus'],
              'reserved_physical_gpus':cfg['reserved_physical_gpus'],
              'capacity_extra_parameters':capacity.capacity_extra_parameters,
              'full_extra_parameters':capacity.capacity_target_parameters,
              'free_disk_gib':disk.free/(1024**3)}
    if require_cuda:
        visible = gpu_status()
        if not set(cfg['allowed_physical_gpus']) <= {g['index'] for g in visible}:
            raise RuntimeError('Not all allowed physical GPU IDs 0..5 are present')
        report['gpu_snapshot'] = visible
    write_json(ROOT/'logs'/'preflight.json',report)
    print('PREFLIGHT PASS: 80 tasks; physical GPUs 6,7 reserved.',flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-check',action='store_true')
    args = parser.parse_args()
    check(not args.cpu_check)
