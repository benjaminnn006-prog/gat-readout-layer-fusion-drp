from __future__ import annotations
import argparse
import time
import joblib
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from src.common import ROOT, read_json, write_json, sha256, fingerprint, file_lock, environment
from src.protocol import config, protocol_signature, verify_data, build_tasks
from src.data import audit_splits, split_frames, load_dictionaries, validate_features, pairs


def cold_frames(full, cfg):
    cells = np.array(sorted(set(full.Cell)))
    shuffled = np.random.default_rng(cfg['cold_cell_split_seed']).permutation(cells)
    ntrain = int(len(cells) * cfg['cold_cell_fractions'][0])
    nval = int(len(cells) * cfg['cold_cell_fractions'][1])
    subsets = [sorted(shuffled[:ntrain].tolist()),
               sorted(shuffled[ntrain:ntrain+nval].tolist()),
               sorted(shuffled[ntrain+nval:].tolist())]
    frames = [full[full.Cell.isin(ids)].reset_index(drop=True) for ids in subsets]
    if any(set(subsets[i]) & set(subsets[j]) for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Cold cell sets overlap')
    if sum(len(f) for f in frames) != len(full):
        raise ValueError('Cold split coverage mismatch')
    return frames, subsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    cfg = config()
    hashes = verify_data()
    sig = protocol_signature(hashes)
    cache = ROOT / 'cache'
    with file_lock(cache / '.prepare.lock'):
        old = cache / 'audit.json'
        if old.exists() and not args.rebuild:
            if read_json(old)['signature'] != sig:
                raise ValueError('Cache stale; do not alter frozen code/config mid-run')
            verify_cache()
            print('Existing preparation verified; no training started.', flush=True)
            return
        if args.rebuild and any((ROOT / 'results' / 'formal').rglob('last.pt')):
            raise RuntimeError('Formal checkpoints exist. Never rebuild a live experiment.')
        started = time.perf_counter()
        full, warm_audit = audit_splits(ROOT / 'data')
        rma, graphs, atoms = load_dictionaries(ROOT / 'data')
        feature_audit = validate_features(full, rma, graphs, atoms)
        cells = sorted(set(full.Cell))
        write_json(cache / 'cell_ids.json', cells)
        all_x = np.stack([rma[c] for c in cells]).astype(np.float32)
        np.save(cache / 'raw_features.npy', all_x)
        cold, cold_ids = cold_frames(full, cfg)
        regimes = [('warm', f, split_frames(ROOT / 'data', f)) for f in range(5)]
        regimes.append(('cold_cell', 0, cold))
        split_audit = []
        for regime, split, frames in regimes:
            root = cache / 'splits' / regime / f'split_{split}'
            root.mkdir(parents=True, exist_ok=True)
            frame_hashes = {}
            for role, frame in zip(('train', 'validation', 'test'), frames):
                path = root / f'{role}.csv'
                frame.to_csv(path, index=False)
                frame_hashes[role] = sha256(path)
            if any(pairs(frames[i]) & pairs(frames[j]) for i in range(3) for j in range(i+1, 3)):
                raise ValueError('Pair leakage')
            train_cells = sorted(set(frames[0].Cell))
            fit_indices = [cells.index(c) for c in train_cells]
            if cfg['compact_dimension'] >= len(train_cells):
                raise ValueError('Compact width exceeds training-cell rank')
            begin = time.perf_counter()
            with threadpool_limits(limits=4):
                transform = PCA(n_components=cfg['compact_dimension'], svd_solver='randomized',
                                iterated_power=7, random_state=42+split)
                transform.fit(all_x[fit_indices])
                compact = transform.transform(all_x).astype(np.float32)
            if not np.isfinite(compact).all():
                raise ValueError('Nonfinite projection')
            np.save(root / 'compact_features.npy', compact)
            joblib.dump(transform, root / 'projection.joblib')
            meta = {'signature': sig, 'regime': regime, 'split': split,
                    'fit_cell_ids': train_cells, 'uses_response_labels_for_projection': False,
                    'frames_sha256': frame_hashes,
                    'raw_sha256': sha256(cache / 'raw_features.npy'),
                    'compact_sha256': sha256(root / 'compact_features.npy'),
                    'projection_sha256': sha256(root / 'projection.joblib'),
                    'cell_ids_sha256': sha256(cache / 'cell_ids.json'),
                    'explained_variance': float(transform.explained_variance_ratio_.sum()),
                    'projection_fit_transform_seconds': time.perf_counter()-begin,
                    'sample_counts': [len(f) for f in frames],
                    'cell_counts': [f.Cell.nunique() for f in frames],
                    'drug_counts': [f.Drug.nunique() for f in frames]}
            write_json(root / 'metadata.json', meta)
            split_audit.append(meta)
            print(f'Prepared {regime} {split}: pairs={meta["sample_counts"]}, cells={meta["cell_counts"]}', flush=True)
        write_json(cache / 'cold_cell_ids.json', dict(zip(('train','validation','test'), cold_ids)))
        write_json(cache / 'audit.json', {'signature': sig, 'source_sha256': hashes,
                    'environment': environment(), 'feature_audit': feature_audit,
                    'warm_split_audit': warm_audit, 'splits': split_audit,
                    'preparation_seconds': time.perf_counter()-started,
                    'upstream_preprocessing_train_only_provenance': 'not established',
                    'response_range': [float(full.IC50.min()), float(full.IC50.max())]})
        write_json(ROOT / 'configs' / 'task_manifest.json', build_tasks())
        verify_cache()
        print('Prepared 80 frozen formal tasks. No training has run.', flush=True)


def verify_cache():
    cfg = config()
    sig = protocol_signature(verify_data())
    audit = read_json(ROOT / 'cache' / 'audit.json')
    if audit['signature'] != sig:
        raise ValueError('Preparation protocol signature mismatch')
    for meta in audit['splits']:
        root = ROOT / 'cache' / 'splits' / meta['regime'] / f'split_{meta["split"]}'
        current = read_json(root / 'metadata.json')
        if current != meta:
            raise ValueError('Metadata changed')
        for role, expected in meta['frames_sha256'].items():
            if sha256(root / f'{role}.csv') != expected:
                raise ValueError('Split file changed')
        for name, key in [('compact_features.npy','compact_sha256'),('projection.joblib','projection_sha256')]:
            if sha256(root / name) != meta[key]:
                raise ValueError('Projection cache changed')
        if sha256(ROOT / 'cache' / 'raw_features.npy') != meta['raw_sha256']:
            raise ValueError('Raw features changed')
        if sha256(ROOT / 'cache' / 'cell_ids.json') != meta['cell_ids_sha256']:
            raise ValueError('Cell ID order changed')
    if cfg['repeats'] != 5:
        raise ValueError('Frozen repeats changed')


if __name__ == '__main__':
    main()
