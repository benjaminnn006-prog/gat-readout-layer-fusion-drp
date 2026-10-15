from __future__ import annotations
import argparse
import os
import random
import signal
import socket
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.common import ROOT, read_json, write_json, sha256, fingerprint, file_lock, environment
from src.protocol import config, task_by_id, task_root, split_root, protocol_signature
from src.data import PairDataset, collate_pairs, load_dictionaries, read_frame
from src.metrics import regression_metrics
from src.model import MedicalGAT

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print('Stop requested: finish current epoch and save resumable state.', flush=True)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def atomic_save(value, path):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def rng_state(generator, device):
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(), 'loader':generator.get_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, generator, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['loader'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def predict(model, loader, device):
    model.eval()
    out = np.empty(len(loader.dataset), dtype=np.float64)
    seen = np.zeros(len(out), dtype=bool)
    with torch.inference_mode():
        for batch in loader:
            indices = batch.row_index.numpy()
            if seen[indices].any() or len(set(indices.tolist())) != len(indices):
                raise ValueError('Repeated prediction indices')
            out[indices] = model(batch.to(device)).detach().cpu().numpy()
            seen[indices] = True
    if not seen.all():
        raise ValueError('Incomplete prediction coverage')
    return out, regression_metrics(loader.dataset.targets, out)


def make_loader(frame, cells, features, graphs, atoms, cfg, device, shuffle=False, generator=None):
    dataset = PairDataset(frame, cells, features, graphs, atoms)
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=shuffle,
                      generator=generator, collate_fn=collate_pairs, num_workers=0,
                      pin_memory=device.type == 'cuda')


def export_attention(model, frame, cells, features, graphs, atoms, cfg, device, root):
    if model.variant not in ('full','attn_pool','layer_gate','equal_fusion'):
        return
    # One example per drug, in ID order, irrespective of prediction error or label.
    selected = frame.sort_values(['Drug','Cell']).drop_duplicates('Drug').reset_index(drop=True)
    loader = make_loader(selected, cells, features, graphs, atoms, cfg, device)
    arrays, layer_rows = {}, []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            indices = batch.row_index.numpy()
            ptr = batch.ptr.numpy()
            _, details = model.drug_vector(batch.to(device), explain=True)
            details = {k:v.cpu().numpy() for k,v in details.items()}
            for i, row in enumerate(indices):
                drug = str(selected.iloc[row].Drug)
                arrays[f'{row}_drug_id'] = np.asarray(drug)
                for key in ('atom_weights_layer1','atom_weights_layer2'):
                    if key in details:
                        arrays[f'{row}_{key}'] = details[key][ptr[i]:ptr[i+1]]
                if 'layer_weights' in details:
                    weights = details['layer_weights'][i]
                    layer_rows.append({'Drug':drug, 'layer1_weight':float(weights[0]),
                                       'layer2_weight':float(weights[1])})
    location = root/'attention'
    location.mkdir(exist_ok=True)
    np.savez_compressed(location/'atom_readout_weights.npz', **arrays)
    if layer_rows:
        pd.DataFrame(layer_rows).to_csv(location/'layer_fusion_weights.csv', index=False)
    write_json(location/'interpretation_boundary.json', {
        'selection':'all test drugs, one representative graph per drug; no error-based selection',
        'node_order':'exact original MolGNet/GRAPH node order; no atom-name mapping inferred',
        'weights':'graph-readout and layer-fusion weights; not GAT edge attention coefficients',
        'cell_conditioned':False, 'chemical_or_clinical_mechanism_evidence':False})


def run(args):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    cfg = config()
    task = task_by_id(args.task)
    device = torch.device(args.device)
    if args.mode == 'formal':
        if device.type != 'cuda' or args.epochs is not None or args.max_samples is not None:
            raise ValueError('Formal mode requires CUDA and frozen full-data 80-epoch protocol')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES','')
        if visible not in [str(g) for g in cfg['allowed_physical_gpus']]:
            raise ValueError('Formal process must see exactly one allowed physical GPU (0..5)')
        if str(device) not in ('cuda:0','cuda'):
            raise ValueError('Visible-device mapping requires logical cuda:0')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; no silent CPU fallback')
    epochs = args.epochs or cfg['epochs']
    if epochs < 1:
        raise ValueError('Positive epochs required')
    torch.set_num_threads(cfg['cpu_threads_per_job'])
    random.seed(task['seed'])
    np.random.seed(task['seed'])
    torch.manual_seed(task['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(task['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sig = protocol_signature()
    audit = read_json(ROOT/'cache'/'audit.json')
    if audit['signature'] != sig:
        raise ValueError('Frozen preparation signature differs from code/data/config')
    location = split_root(task)
    meta = read_json(location/'metadata.json')
    if meta['signature'] != sig:
        raise ValueError('Split preparation mismatch')
    frames = []
    for role in ('train','validation','test'):
        if sha256(location/f'{role}.csv') != meta['frames_sha256'][role]:
            raise ValueError('Split CSV changed')
        frames.append(read_frame(location/f'{role}.csv'))
    if sorted(set(frames[0].Cell)) != meta['fit_cell_ids']:
        raise ValueError('Projection was not fit on precisely the training cells')
    feature_path = (ROOT/'cache'/'raw_features.npy') if task['feature']=='raw' else location/'compact_features.npy'
    feature_hash = sha256(feature_path)
    if feature_hash != meta['raw_sha256' if task['feature']=='raw' else 'compact_sha256']:
        raise ValueError('Feature cache changed')
    if sha256(ROOT/'cache'/'cell_ids.json') != meta['cell_ids_sha256']:
        raise ValueError('Cell ordering changed')
    features = np.load(feature_path)
    cells = read_json(ROOT/'cache'/'cell_ids.json')
    if args.mode == 'smoke':
        cap = args.max_samples or 256
        frames = [f.iloc[:min(len(f), cap)].reset_index(drop=True) for f in frames]
    identity = {'protocol_signature':sig, 'task':task, 'mode':args.mode, 'epochs':epochs,
                'sample_counts':[len(f) for f in frames], 'feature_sha256':feature_hash}
    run_sig = fingerprint(identity)
    root = task_root(task, args.mode)
    with file_lock(root/'.train.lock'):
        if (root/'metrics.json').exists():
            existing = read_json(root/'metrics.json')
            if existing['run_signature'] != run_sig:
                raise ValueError('Completed task belongs to a different protocol')
            for name, expected in existing['artifacts_sha256'].items():
                if sha256(root/name) != expected:
                    raise ValueError('Completed task artifact changed')
            print(f'Already complete {task["id"]}', flush=True)
            return
        started = time.perf_counter()
        _, graphs, atoms = load_dictionaries(ROOT/'data')
        generator = torch.Generator().manual_seed(task['seed'])
        train, validation, test = frames
        train_loader = make_loader(train,cells,features,graphs,atoms,cfg,device,True,generator)
        val_loader = make_loader(validation,cells,features,graphs,atoms,cfg,device)
        model = MedicalGAT(features.shape[1],task['variant']).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=0)
        env = environment()
        env.update({'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                    'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'cpu_threads':cfg['cpu_threads_per_job'], 'AMP':False, 'TF32':False,
                    'sparse_cuda_bitwise_determinism_guaranteed':False})
        history, first_epoch = [], 0
        best_pcc, best_epoch, best_state, best_val = -float('inf'), -1, None, None
        train_seconds, peak, resumes = 0.0, 0, 0
        checkpoint = root/'last.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint,map_location=device,weights_only=False)
            if saved['run_signature'] != run_sig:
                raise ValueError('Checkpoint protocol mismatch')
            for key in ('hostname','torch','torch_geometric','gpu_name'):
                if saved['environment'][key] != env[key]:
                    raise ValueError('Resume host/runtime changed')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            first_epoch = saved['epoch']+1
            history = saved['history']
            best_pcc, best_epoch = saved['best_pcc'], saved['best_epoch']
            best_state = {k:v.cpu() for k,v in saved['best_state'].items()}
            best_val = saved['best_val']
            train_seconds, peak = saved['train_seconds'], saved['peak']
            resumes = saved['resumes']+1
            restore_rng(saved['rng'],generator,device)
            print(f'Resume {task["id"]} from epoch {first_epoch+1}',flush=True)
        write_json(root/'run_config.json',{**identity,'run_signature':run_sig,'config':cfg,
                    'environment':env,'preparation':meta})
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        for epoch in range(first_epoch,epochs):
            sync(device)
            begin = time.perf_counter()
            model.train()
            loss_sum, count = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                predicted = model(batch)
                loss = F_mse(predicted,batch.ic50.view(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())*batch.num_graphs
                count += batch.num_graphs
            _, val_metrics = predict(model,val_loader,device)
            if val_metrics['PCC'] is None:
                raise FloatingPointError('Undefined validation PCC')
            sync(device)
            elapsed = time.perf_counter()-begin
            train_seconds += elapsed
            if val_metrics['PCC'] > best_pcc:
                best_pcc, best_epoch, best_val = val_metrics['PCC'],epoch,val_metrics
                best_state = cpu_state(model)
            if device.type=='cuda':
                peak = max(peak,torch.cuda.max_memory_allocated(device))
            history.append({'epoch':epoch+1,'train_MSE':loss_sum/count,
                            'validation_PCC':val_metrics['PCC'],'validation_RMSE':val_metrics['RMSE'],
                            'train_validation_seconds':elapsed,'train_samples':count,
                            'validation_samples':len(validation)})
            pd.DataFrame(history).to_csv(root/'history.csv',index=False)
            print(f'{task["id"]} epoch={epoch+1}/{epochs} val_PCC={best_pcc:.6f} seconds={elapsed:.2f}',flush=True)
            if (epoch+1)%cfg['checkpoint_interval']==0 or epoch+1==epochs or STOP_REQUESTED:
                atomic_save({'run_signature':run_sig,'epoch':epoch,'model':model.state_dict(),
                             'optimizer':optimizer.state_dict(),'best_state':best_state,
                             'best_pcc':best_pcc,'best_epoch':best_epoch,'best_val':best_val,
                             'history':history,'train_seconds':train_seconds,'peak':peak,
                             'rng':rng_state(generator,device),'environment':env,'resumes':resumes},checkpoint)
            if STOP_REQUESTED and epoch+1<epochs:
                raise RuntimeError('Stopped with resumable checkpoint; no formal test result generated')
        if best_state is None:
            raise RuntimeError('No valid validation-selected model')
        model.load_state_dict(best_state)
        atomic_save(best_state,root/'best_model.pt')
        val_pred,val_metrics = predict(model,val_loader,device)
        out = validation.rename(columns={'IC50':'y_true'}).copy()
        out['y_pred'] = val_pred
        out.to_csv(root/'validation_predictions.csv',index=False)
        test_loader = make_loader(test,cells,features,graphs,atoms,cfg,device)
        sync(device)
        begin = time.perf_counter()
        test_pred,test_metrics = predict(model,test_loader,device)
        sync(device)
        test_seconds = time.perf_counter()-begin
        out = test.rename(columns={'IC50':'y_true'}).copy()
        out['y_pred'] = test_pred
        out.to_csv(root/'test_predictions.csv',index=False)
        export_attention(model,test,cells,features,graphs,atoms,cfg,device,root)
        names = ['best_model.pt','history.csv','test_predictions.csv','validation_predictions.csv','run_config.json']
        artifacts = {n:sha256(root/n) for n in names}
        for path in (root/'attention').glob('*'):
            artifacts[path.relative_to(root).as_posix()] = sha256(path)
        result = {**identity,'run_signature':run_sig,'status':'complete',
                  'publication_eligible':args.mode=='formal' and len(history)==80,
                  'completed_epochs':len(history),'best_epoch':best_epoch+1,
                  'metrics':test_metrics,'best_validation_metrics':best_val,
                  'parameter_count':sum(p.numel() for p in model.parameters()),
                  'capacity_extra_parameters':model.capacity_extra_parameters,
                  'capacity_target_parameters':model.capacity_target_parameters,
                  'train_validation_seconds':train_seconds,'test_seconds':test_seconds,
                  'peak_allocated_gpu_mib':peak/(1024**2),'resumes':resumes,
                  'this_invocation_wall_seconds':time.perf_counter()-started,
                  'timing_boundary':'epoch training+validation only; excludes checkpoint I/O, best-state copy, final prediction warm-up and explanation exports; concurrent workload',
                  'environment':env,'artifacts_sha256':artifacts}
        write_json(root/'metrics.json',result)
        print(f'COMPLETE {task["id"]}: {test_metrics}',flush=True)


def F_mse(predicted,target):
    return torch.nn.functional.mse_loss(predicted,target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--mode',choices=['formal','smoke'],default='formal')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--max-samples',type=int)
    run(parser.parse_args())


if __name__=='__main__':
    main()
