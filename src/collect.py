from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd
from src.common import ROOT, read_json, write_json, sha256, fingerprint
from src.protocol import build_tasks, protocol_signature, task_root, split_root
from src.prepare import verify_cache
from src.data import read_frame
from src.metrics import regression_metrics

METRICS = ['PCC','RMSE','MAE','R2']


def audit_result(task,sig):
    root = task_root(task)
    result = read_json(root/'metrics.json')
    if result['protocol_signature']!=sig or result['task']!=task or result['mode']!='formal':
        raise ValueError('Formal identity mismatch')
    if not result['publication_eligible'] or result['epochs']!=80 or result['completed_epochs']!=80:
        raise ValueError('Partial or smoke output in formal results')
    metadata = read_json(split_root(task)/'metadata.json')
    if result['sample_counts']!=metadata['sample_counts']:
        raise ValueError('Formal task did not use the complete prepared data')
    expected_feature = metadata['raw_sha256' if task['feature']=='raw' else 'compact_sha256']
    if result['feature_sha256']!=expected_feature:
        raise ValueError('Formal feature identity mismatch')
    identity = {k:result[k] for k in ('protocol_signature','task','mode','epochs','sample_counts','feature_sha256')}
    if fingerprint(identity)!=result['run_signature']:
        raise ValueError('Run identity hash mismatch')
    for name,expected in result['artifacts_sha256'].items():
        if sha256(root/name)!=expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    run_config = read_json(root/'run_config.json')
    if run_config['run_signature']!=result['run_signature']:
        raise ValueError('Run config mismatch')
    history = pd.read_csv(root/'history.csv')
    if history.epoch.tolist()!=list(range(1,81)):
        raise ValueError('Incomplete epoch sequence')
    if not np.isfinite(history.select_dtypes('number').to_numpy()).all():
        raise ValueError('Nonfinite training history')
    if not (history.train_samples==metadata['sample_counts'][0]).all() or not (
            history.validation_samples==metadata['sample_counts'][1]).all():
        raise ValueError('An epoch did not cover all training/validation samples')
    # idxmax chooses earliest epoch on ties, matching the train loop.
    if int(history.loc[history.validation_PCC.idxmax(),'epoch'])!=result['best_epoch']:
        raise ValueError('Checkpoint was not selected by validation PCC')
    expected_frame = read_frame(split_root(task)/'test.csv')
    prediction = pd.read_csv(root/'test_predictions.csv',dtype={'Cell':str,'Drug':str})
    if prediction[['Cell','Drug']].to_dict('list')!=expected_frame[['Cell','Drug']].to_dict('list'):
        raise ValueError('Prediction row coverage/order mismatch')
    if not np.allclose(prediction.y_true,expected_frame.IC50,rtol=0,atol=1e-12):
        raise ValueError('Prediction labels changed')
    checked = regression_metrics(prediction.y_true.to_numpy(dtype=np.float32),prediction.y_pred)
    for key,value in checked.items():
        if value is None or not math.isclose(value,result['metrics'][key],rel_tol=1e-7,abs_tol=1e-9):
            raise ValueError(f'Recomputed test metric mismatch: {key}')
    physical = result['environment']['visible_devices']
    if physical not in [str(g) for g in range(6)]:
        raise ValueError('Reserved GPU policy violated')
    return result,prediction


def collect(require_complete=False):
    verify_cache()
    sig = protocol_signature()
    tasks = build_tasks()
    output = ROOT/'summary'
    output.mkdir(exist_ok=True)
    rows,missing,invalid,drug_rows = [],[],[],[]
    for task in tasks:
        if not (task_root(task)/'metrics.json').exists():
            missing.append(task['id'])
            continue
        try:
            result,prediction = audit_result(task,sig)
        except Exception as error:
            invalid.append({'task':task['id'],'error':str(error)})
            continue
        row = {**task,**result['metrics'],'best_epoch':result['best_epoch'],
               'parameters':result['parameter_count'],
               'train_validation_minutes':result['train_validation_seconds']/60,
               'test_seconds':result['test_seconds'],
               'peak_allocated_gpu_mib':result['peak_allocated_gpu_mib'],
               'resumes':result['resumes'],'gpu':result['environment']['visible_devices']}
        rows.append(row)
        for drug,part in prediction.groupby('Drug',sort=True):
            metrics = regression_metrics(part.y_true.to_numpy(dtype=np.float32),part.y_pred) if len(part)>=5 else {}
            drug_rows.append({'task_id':task['id'],'regime':task['regime'],'feature':task['feature'],
                              'variant':task['variant'],'repeat':task['repeat'],'Drug':drug,
                              'n_pairs':len(part),'eligible_n_ge_5':len(part)>=5,**metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'all_runs.csv',index=False)
    pd.DataFrame(missing,columns=['task_id']).to_csv(output/'missing_tasks.csv',index=False)
    pd.DataFrame(invalid,columns=['task','error']).to_csv(output/'invalid_tasks.csv',index=False)
    drugs = pd.DataFrame(drug_rows)
    drugs.to_csv(output/'per_drug_per_run.csv',index=False)
    aggregates = []
    if not frame.empty:
        measures = METRICS+['parameters','train_validation_minutes','test_seconds','peak_allocated_gpu_mib']
        for keys,part in frame.groupby(['regime','feature','variant'],sort=True):
            aggregate = dict(zip(['regime','feature','variant'],keys))
            aggregate.update({'n_runs':len(part),'complete_5_runs':len(part)==5})
            for measure in measures:
                aggregate[f'{measure}_mean'] = float(part[measure].mean())
                aggregate[f'{measure}_sample_sd'] = float(part[measure].std(ddof=1)) if len(part)>1 else None
            aggregates.append(aggregate)
    aggregated = pd.DataFrame(aggregates)
    aggregated.to_csv(output/'aggregate_metrics.csv',index=False)
    if not aggregated.empty:
        for name,mask in [
            ('table_main_algorithms', (aggregated.regime=='warm')&(aggregated.feature=='compact')),
            ('table_cold_cell', aggregated.regime=='cold_cell'),
            ('table_input_configuration',aggregated.variant.isin(['baseline','full'])&(aggregated.regime=='warm'))]:
            aggregated.loc[mask].to_csv(output/f'{name}.csv',index=False)
    paired,interactions = [],[]
    if not frame.empty:
        for (regime,feature,repeat),part in frame.groupby(['regime','feature','repeat']):
            by_variant = {r['variant']:r for r in part.to_dict('records')}
            if 'baseline' in by_variant:
                base = by_variant['baseline']
                for variant,row in by_variant.items():
                    if variant!='baseline':
                        paired.append({'regime':regime,'feature':feature,'repeat':int(repeat),
                                       'variant':variant,'delta_PCC':row['PCC']-base['PCC'],
                                       'delta_RMSE':row['RMSE']-base['RMSE'],
                                       'same_split_and_seed':True})
            if set(['baseline','attn_pool','layer_gate','full'])<=set(by_variant):
                for measure in ['PCC','RMSE']:
                    interactions.append({'regime':regime,'feature':feature,'repeat':int(repeat),
                        'metric':measure,'interaction_contrast':by_variant['full'][measure]-
                        by_variant['attn_pool'][measure]-by_variant['layer_gate'][measure]+by_variant['baseline'][measure]})
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False)
    pd.DataFrame(interactions).to_csv(output/'module_interaction.csv',index=False)
    macro_rows = []
    if not drugs.empty:
        for task_id,part in drugs[drugs.eligible_n_ge_5].groupby('task_id'):
            macro_rows.append({'task_id':task_id,'n_drugs_n_ge_5':len(part),
                               **{f'macro_{m}':float(part[m].mean()) for m in METRICS},
                               'n_drugs_defined_PCC':int(part.PCC.notna().sum())})
    pd.DataFrame(macro_rows).to_csv(output/'macro_drug_metrics.csv',index=False)
    eligible = len(rows)==80 and not invalid and not missing
    audit = {'protocol_signature':sig,'expected_tasks':80,'audited_complete_tasks':len(rows),
             'missing_tasks':missing,'invalid_tasks':invalid,'publication_eligible':eligible,
             'reserved_gpu_policy_pass':not any('GPU' in e['error'] for e in invalid),
             'test_used_for_selection':False,'warm_test_sets_independent':False,
             'cold_test_sets_independent_across_repeats':False,
             'upstream_train_only_preprocessing_provenance':'not established',
             'sd_interpretation':'run variability; not confidence intervals or independent test folds',
             'no_clinical_effectiveness_claim':True}
    write_json(output/'audit.json',audit)
    lines = ['# 实验汇总 / Experiment summary','',
             f'正式任务：{len(rows)}/80 通过逐次审计。',
             f'完整实验可用于论文：{eligible}。','',
             '## 结果表','',
             '| Regime | Input | Model | n | PCC mean ± SD | RMSE mean ± SD |',
             '|---|---|---|---:|---:|---:|']
    for row in aggregates:
        def fmt(key):
            sd = row[f'{key}_sample_sd']
            return f'{row[f"{key}_mean"]:.5f} ± {sd:.5f}' if sd is not None else f'{row[f"{key}_mean"]:.5f} (one run)'
        lines.append(f'| {row["regime"]} | {row["feature"]} | {row["variant"]} | {row["n_runs"]} | {fmt("PCC")} | {fmt("RMSE")} |')
    lines += ['', '## 解读边界','',
              '- 未完成或审计失败的结果不进入完整实验结论；查看 missing_tasks.csv / invalid_tasks.csv。',
              '- warm：五个原验证分区，共享原固定测试集；cold_cell：一个固定实体隔离划分、五个训练种子。',
              '- 样本标准差代表运行波动，不是五个独立测试集的误差或置信区间。',
              '- 模块交互对比是描述性指标，不自动代表统计显著性或因果机制。',
              '- 分药物指标只计算至少五个测试对的药物；常数标签的 PCC 缺失，宏平均不包含缺失 PCC。',
              '- 预训练原子输入和上游基因选择的训练隔离来源尚未建立，不能宣称端到端全流程无泄漏。',
              '- 原子汇总权重和层权重用于模型行为分析，不是化学机制或临床有效性的证据。',
              '- 训练时间来自并发负载；不与旧论文成本数字直接混用。',
              '- 没有自动挑选最优测试配置或自动宣称方法改进成立。']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'AUDIT: {len(rows)}/80 valid; missing={len(missing)} invalid={len(invalid)} publication_eligible={eligible}',flush=True)
    if invalid or (require_complete and not eligible):
        raise RuntimeError('Formal audit failed or required complete experiment is incomplete')
    return audit


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete',action='store_true')
    args = parser.parse_args()
    collect(args.require_complete)
