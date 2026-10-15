# Graph Readout and Layer Fusion in Graph Attention Networks for Drug Response Prediction

Research code and reproducibility metadata for the ISAIMS 2026 manuscript. The paper is a controlled evaluation of graph readout and layer fusion within a fixed GAT predictor, with expression-input and modality controls. Improvements are small and vary by evaluation setting.

## Evaluation identities

`splits/warm/split_0` through `split_4` provide exact Cell/Drug identifiers for training, validation and testing. Their 15,910 test pairs are shared; both development partitions and training seeds vary. `splits/cold_cell/split_0` is one fixed cell-disjoint split generated with seed 20261011, containing 669/143/145 training/validation/test cells. Its five repeats vary training seeds only (42–46). Neither design samples alternative held-out cell populations.

## Files

- `src/`: original experiment implementation.
- `tests/`: original implementation checks.
- `configs/`: complete experiment configuration and 80-task manifest.
- `splits/`: exact pair identities and projection-fit cell IDs; response values are not duplicated here.
- `metadata/gene_order.csv`: all 5,757 gene identifiers in input-column order.
- `metadata/response_definition.json`: response target definition, source artifact hash and unresolved transformation. The target is the prediction label; no therapeutic-target annotation enters the model.
- `metadata/drug_structures.csv`: stored structures and feature dimensions, not a new structure-resolution exercise.
- `metadata/*provenance*`, `DIPK_source_lineage.json`, `DIPK_code_comparison.json`: recovered code lineage and feature history checks.
- `results/`: run-level metrics, 60 individual paired PCC/RMSE contrasts, per-drug correlations and counts, and the exact Figure 2 source data.
- `analysis/`: checks and figure reconstruction using these saved summaries.
- `data/expected_source_hashes.json`: identities of the processed input artifacts required by the original training code.

## Data provenance and limits

GDSC is the original pharmacogenomic resource: https://www.cancerrxgene.org/ . The local preprocessing lineage matches the official DIPK implementation: https://github.com/user15632/DIPK (Li et al., 2024, https://doi.org/10.1093/bib/bbae153). Its source data are linked by that repository. Nine of ten relevant local preprocessing scripts match the official clone at commit `0edfc151fbbdb7864d577865f9541a689d706c5b` byte for byte; the remaining MolGNet export script changes only a relative checkpoint path. This identifies code lineage, not a byte-verified historical data download.

The actual study response file is `lenient_split_fold_0_dataset.csv`: 159,114 pairs, 957 cells, 206 drugs. Its legacy IC50 column stores responses in [0, 1], imported unchanged from the float32 `ic50` feature of existing TFRecord files. The preceding assay transformation is unknown. Errors are in processed units; no inverse transform or raw/log concentration interpretation is claimed.

The stored fixed expression and molecular feature artifacts are identified by hashes. Expression values agree with the locally available source matrix. CPU reconstruction from the local candidate MolGNet checkpoint differs from stored vectors by at most 0.00379086; the historical export remains incompletely verified. Use the exact stored features for the reported processed benchmark. The original key-gene selection rationale is unavailable.

This repository provides the requested code, split identifiers, ordered gene metadata, response metadata, configurations and derived results. It does not republish third-party raw response files, expression matrices, pretrained checkpoints or binary feature dictionaries, and does not assert a new license over those inputs. Retrieve source inputs through the original resources subject to their terms; compare hashes before using them. This is not an assertion that all raw-assay preprocessing has been reconstructed.

## Check the reported summaries

With Python, NumPy and pandas:

```bash
python analysis/check_summaries.py
```

For the quantitative figure, also install matplotlib:

```bash
python analysis/figure2.py
```

All 206 drugs are included for each baseline setting. The figure averages five separately computed within-drug correlations; it does not average or pool predictions first. Error bars are sample SD across five repeats, not confidence intervals over cell populations. Original training is not started by either command.

## Training and resources

The original training implementation requires the exact processed input artifacts; this metadata repository alone is not a complete raw-data training package. Configurations use 80 epochs, batch size 128, Adam at 1e-4 and validation-PCC checkpoint selection. Each task used one NVIDIA A100-PCIE-40GB GPU under a recorded six-job concurrent workload. Time is the sum of synchronized training/validation epochs; projection fitting, feature loading, checkpoint/history writes and final testing are excluded. Memory is the peak PyTorch allocation after resetting peak counters, not total device or reserved-cache memory.

No patient, external-resource or unseen-drug validation is claimed. No original fitted projection objects or trained DRP checkpoints were present in the returned result archive.
