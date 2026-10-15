from pathlib import Path
from src.common import ROOT, read_json, sha256, fingerprint

MAIN = ['baseline','attn_pool','layer_gate','full','equal_fusion','capacity','gcn','gatv2']
CORE = ['baseline','attn_pool','layer_gate','full']


def config():
    cfg = read_json(ROOT / 'configs' / 'experiment.json')
    if cfg['epochs'] != 80 or cfg['repeats'] != 5 or cfg['compact_dimension'] != 128:
        raise ValueError('Frozen 80-epoch/5-repeat/128-dimensional protocol changed')
    if cfg['allowed_physical_gpus'] != list(range(6)) or cfg['reserved_physical_gpus'] != [6,7]:
        raise ValueError('GPU reservation policy changed')
    return cfg


def verify_data():
    expected = read_json(ROOT / 'data' / 'source_manifest.json')['sha256']
    actual = {name: sha256(ROOT / 'data' / name) for name in expected}
    if actual != expected:
        raise ValueError('Bundled scientific input identity mismatch')
    return actual


def protocol_signature(hashes=None):
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'src').glob('*.py'))}
    return fingerprint({'config':config(), 'data':hashes or verify_data(), 'code':code,
                        'task_manifest':build_tasks()})


def build_tasks():
    tasks = []
    def add(group, regime, feature, variants, repeat):
        for variant in variants:
            split = repeat if regime == 'warm' else 0
            tid = f'{regime}__{feature}__{variant}__r{repeat}'
            tasks.append({'id':tid, 'group':group, 'regime':regime, 'feature':feature,
                          'variant':variant, 'repeat':repeat, 'split':split, 'seed':42+repeat})
    # Finish paired core evidence early; no allocation depends on test outcomes.
    for repeat in range(5):
        add('01_core_ablation','warm','compact',CORE,repeat)
        add('02_cold_cell','cold_cell','compact',CORE,repeat)
    for repeat in range(5):
        add('03_algorithm_controls','warm','compact',MAIN[4:],repeat)
        add('04_input_configuration','warm','raw',['baseline','full'],repeat)
        add('05_modality_ablation','warm','compact',['drug_only','cell_only'],repeat)
    assert len(tasks) == 80 and len({t['id'] for t in tasks}) == 80
    return tasks


def task_by_id(tid):
    matches = [t for t in build_tasks() if t['id'] == tid]
    if len(matches) != 1:
        raise ValueError('Unknown frozen task ID')
    return matches[0]


def task_root(task, mode='formal'):
    return ROOT/'results'/mode/task['group']/task['id']


def split_root(task):
    return ROOT/'cache'/'splits'/task['regime']/f'split_{task["split"]}'
