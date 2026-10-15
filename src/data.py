from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


def read_frame(path):
    frame = pd.read_csv(path, dtype={"Cell": str, "Drug": str})
    if list(frame.columns) != ["Cell", "Drug", "IC50"]:
        raise ValueError(f"Unexpected CSV schema: {path}: {list(frame.columns)}")
    if frame.isna().any().any() or not np.isfinite(frame.IC50.to_numpy()).all():
        raise ValueError(f"Missing or non-finite entries: {path}")
    if frame.duplicated(["Cell", "Drug"]).any():
        raise ValueError(f"Duplicated cell-drug pairs: {path}")
    return frame


def pairs(frame):
    return set(zip(frame.Cell, frame.Drug))


def split_frames(data_dir, fold):
    data_dir = Path(data_dir)
    full = read_frame(data_dir / "lenient_split_fold_0_dataset.csv")
    val = read_frame(data_dir / f"lenient_split_fold_{fold}_validation.csv")
    test = read_frame(data_dir / "lenient_split_test.csv")
    full_pairs, val_pairs, test_pairs = pairs(full), pairs(val), pairs(test)
    if val_pairs & test_pairs or not (val_pairs | test_pairs) <= full_pairs:
        raise ValueError("Validation/test overlap or pairs outside full dataset.")
    reference = full.set_index(["Cell", "Drug"]).IC50
    for subset in (val, test):
        aligned = reference.reindex(pd.MultiIndex.from_frame(subset[["Cell", "Drug"]]))
        if not np.array_equal(aligned.to_numpy(), subset.IC50.to_numpy()):
            raise ValueError("A pair has inconsistent response labels across split files.")
    excluded = val_pairs | test_pairs
    keep = [(c, d) not in excluded for c, d in zip(full.Cell, full.Drug)]
    train = full.loc[keep].reset_index(drop=True)
    if pairs(train) & excluded or pairs(train) | excluded != full_pairs:
        raise ValueError("Invalid training split.")
    return train, val.reset_index(drop=True), test.reset_index(drop=True)


def audit_splits(data_dir):
    full = read_frame(Path(data_dir) / "lenient_split_fold_0_dataset.csv")
    test = read_frame(Path(data_dir) / "lenient_split_test.csv")
    validation_sets = []
    rows = []
    for fold in range(5):
        train, val, _ = split_frames(data_dir, fold)
        validation_sets.append(pairs(val))
        rows.append({"fold": fold, "train_pairs": len(train),
                     "validation_pairs": len(val), "test_pairs": len(test),
                     "train_cells": int(train.Cell.nunique())})
    for left in range(5):
        for right in range(left + 1, 5):
            if validation_sets[left] & validation_sets[right]:
                raise ValueError("Validation folds overlap.")
    if set.union(*validation_sets) != pairs(full) - pairs(test):
        raise ValueError("Validation folds do not cover the development pool.")
    return full, rows


def string_keys(values):
    normalized = {str(k): v for k, v in values.items()}
    if len(normalized) != len(values):
        raise ValueError("Duplicate dictionary IDs after string conversion.")
    return normalized


def load_dictionaries(data_dir):
    """Load only trusted local project pickles, never arbitrary downloaded pickles."""
    root = Path(data_dir)
    rma = string_keys(joblib.load(root / "RMA_dict.pkl"))
    graphs = string_keys(joblib.load(root / "GRAPH_dict.pkl"))
    atoms = string_keys(joblib.load(root / "MolGNet_dict.pkl"))
    return rma, graphs, atoms


def validate_features(full, rma, graphs, atoms):
    cells, drugs = set(full.Cell), set(full.Drug)
    for name, ids, dictionary in [("expression", cells, rma),
                                  ("graph", drugs, graphs), ("atoms", drugs, atoms)]:
        missing = ids - set(dictionary)
        if missing:
            raise ValueError(f"Missing {name} IDs: {sorted(missing)[:8]}")
    lengths = set()
    for cell in cells:
        vector = np.asarray(rma[cell], dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"Invalid expression vector: {cell}")
        lengths.add(len(vector))
    if lengths != {5757}:
        raise ValueError(f"Expected 5757 expression features; found {lengths}.")
    atom_lengths = []
    for drug in drugs:
        x = torch.as_tensor(atoms[drug]).detach().cpu()
        edge = graphs[drug].edge_index.detach().cpu()
        if x.ndim != 2 or x.shape[1] != 768 or not torch.isfinite(x).all():
            raise ValueError(f"Invalid pretrained features: {drug}")
        if graphs[drug].num_nodes != x.shape[0]:
            raise ValueError(f"Atom/graph node count mismatch: {drug}")
        if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype != torch.int64:
            raise ValueError(f"Invalid edge array: {drug}")
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= x.shape[0]):
            raise ValueError(f"Out-of-range graph edge: {drug}")
        atom_lengths.append(int(x.shape[0]))
    return {"cells": len(cells), "drugs": len(drugs), "expression_dim": 5757,
            "atom_dim": 768, "min_atoms": min(atom_lengths), "max_atoms": max(atom_lengths)}


class PairDataset(Dataset):
    """Share 206 graphs and per-cell vectors rather than materializing 114k graphs."""
    def __init__(self, frame, cells, features, graphs, atoms):
        self.frame = frame.reset_index(drop=True)
        self.cells = {cell: i for i, cell in enumerate(cells)}
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.graphs = graphs
        self.atoms = {drug: torch.as_tensor(atoms[drug], dtype=torch.float32).detach().cpu()
                      for drug in set(frame.Drug)}
        self.cell_ids = self.frame.Cell.to_list()
        self.drug_ids = self.frame.Drug.to_list()
        self.targets = self.frame.IC50.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        drug = self.drug_ids[index]
        return Data(x=self.atoms[drug], edge_index=self.graphs[drug].edge_index,
                    GEF=self.features[self.cells[self.cell_ids[index]]].unsqueeze(0),
                    ic50=torch.tensor([self.targets[index]], dtype=torch.float32),
                    row_index=torch.tensor([index], dtype=torch.long))


def collate_pairs(items):
    # PyG increments attributes containing 'index'; exclude row_index and restore it.
    batch = Batch.from_data_list(items, exclude_keys=["row_index"])
    batch.row_index = torch.cat([item.row_index for item in items])
    return batch
