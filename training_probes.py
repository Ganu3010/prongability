#!/usr/bin/env python3
"""
Trains a linear probe and a non-linear (MLP) probe per target variable on
top of DINOv3 frame-window encodings (build_encoded_dataset.py's output).

Feature construction from each example's N-frame window (--feature-mode):
    concat (default): flatten all N frame embeddings into one vector,
                       preserving temporal order (oldest -> newest).
                       Feature dim = N * embedding_dim.
    mean:              average-pool across the N frames -> fixed dim
                       regardless of window size, loses frame ordering.
    last:              use ONLY the current (most recent) frame's
                       embedding -- an ablation for "does temporal
                       context help at all," ignoring the rest of the
                       window entirely.

Per target column, rows where that column is null (e.g. nearest_in_O__*
on frames with nothing in O) are dropped for THAT target's probe only --
other targets' training data isn't affected by one target's nulls.

Classification targets (any_colliding, *_occlusion, *_obj_class) get
LogisticRegression (linear, L2-regularized by default) + MLPClassifier
(non-linear): accuracy + macro-F1. Everything else is treated as
regression and gets RidgeCV (linear, L2-regularized with the penalty
strength chosen by internal cross-validation) + MLPRegressor: R^2 + MAE.

RidgeCV rather than plain LinearRegression for the regression linear
probe deliberately: --feature-mode concat makes feature dimension =
N * embedding_dim, which for real DINOv3 embeddings (768+ dims per
frame) with even a modest window size easily exceeds the number of
training examples in a probe-scale dataset. Unregularized OLS in that
regime overfits catastrophically (observed directly during testing: R^2
of -2500 on a held-out split) -- Ridge is also the standard choice in
probing literature for exactly this reason, not just a fix for a test
artifact.

Note: MLP early stopping carves its validation split out of the training
set internally (sklearn default) -- this script's own --dataset-dir val
split is used only for reporting metrics, not for that internal
early-stopping split or any hyperparameter search. This is a first-pass
baseline, not a tuned model.

Usage:
    python training_probes.py --dataset-dir ./encoded_dataset --feature-mode concat
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

CLASSIFICATION_COLUMNS_SUFFIXES = ("_occlusion", "_obj_class")
CLASSIFICATION_COLUMNS_EXACT = {"any_colliding"}
MIN_TRAIN_EXAMPLES = 10


def infer_target_type(column):
    if column in CLASSIFICATION_COLUMNS_EXACT:
        return "classification"
    if any(column.endswith(suf) for suf in CLASSIFICATION_COLUMNS_SUFFIXES):
        return "classification"
    return "regression"


def load_split(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


def build_features(records, mode, expected_dim=None):
    if not records:
        return np.zeros((0, expected_dim or 0), dtype=np.float32)
    feats = []
    for r in records:
        window = np.array(r["frame_encodings"], dtype=np.float32)  # (N, D)
        if mode == "concat":
            feats.append(window.reshape(-1))
        elif mode == "mean":
            feats.append(window.mean(axis=0))
        elif mode == "last":
            feats.append(window[-1])
        else:
            raise ValueError(f"Unknown feature mode: {mode}")
    return np.stack(feats)


def get_target_columns(records):
    if not records:
        return []
    return [k for k in records[0].keys() if k != "frame_encodings"]


def prepare_target(records, column):
    """Returns (row_indices, values) for rows where `column` is not null."""
    indices, values = [], []
    for i, r in enumerate(records):
        v = r.get(column)
        if v is not None and v != "":
            indices.append(i)
            values.append(v)
    return np.array(indices, dtype=int), values


def run_regression_probe(X_train, y_train, X_val, y_val, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) else X_val
    X_test_s = scaler.transform(X_test) if len(X_test) else X_test

    results = {}
    for name, model in [
        ("linear", RidgeCV(alphas=np.logspace(-3, 4, 15))),
        ("mlp", MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=1000,
                              early_stopping=True, random_state=0)),
    ]:
        model.fit(X_train_s, y_train)
        metrics = {}
        for split_name, X_s, y in [("val", X_val_s, y_val), ("test", X_test_s, y_test)]:
            if len(y) == 0:
                metrics[split_name] = None
                continue
            pred = model.predict(X_s)
            metrics[split_name] = {"r2": float(r2_score(y, pred)),
                                    "mae": float(mean_absolute_error(y, pred)),
                                    "n": len(y)}
        results[name] = metrics
    return results


def run_classification_probe(X_train, y_train, X_val, y_val, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) else X_val
    X_test_s = scaler.transform(X_test) if len(X_test) else X_test

    le = LabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)

    def encode_seen_only(y, X_s):
        y_arr = np.array(y)
        mask = np.isin(y_arr, le.classes_)  # val/test may contain a class never seen in train
        if not mask.any():
            return np.zeros((0, X_s.shape[1]) if len(X_s) else (0, 0)), np.array([]), 0
        return X_s[mask], le.transform(y_arr[mask]), int(mask.sum())

    results = {}
    for name, model in [
        ("linear", LogisticRegression(max_iter=2000)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=1000,
                               early_stopping=True, random_state=0)),
    ]:
        model.fit(X_train_s, y_train_enc)
        metrics = {}
        for split_name, X_s, y in [("val", X_val_s, y_val), ("test", X_test_s, y_test)]:
            if len(y) == 0:
                metrics[split_name] = None
                continue
            X_seen, y_seen, n_seen = encode_seen_only(y, X_s)
            if n_seen == 0:
                metrics[split_name] = None
                continue
            pred = model.predict(X_seen)
            metrics[split_name] = {"accuracy": float(accuracy_score(y_seen, pred)),
                                    "macro_f1": float(f1_score(y_seen, pred, average="macro")),
                                    "n": n_seen}
        results[name] = metrics
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True,
                   help="Directory with train.json/val.json/test.json from build_encoded_dataset.py")
    p.add_argument("--feature-mode", choices=["concat", "mean", "last"], default="concat")
    args = p.parse_args()

    train = load_split(os.path.join(args.dataset_dir, "train.json"))
    val = load_split(os.path.join(args.dataset_dir, "val.json"))
    test = load_split(os.path.join(args.dataset_dir, "test.json"))

    if not train:
        raise ValueError(f"{args.dataset_dir}/train.json is empty or missing -- nothing to fit probes on")

    print(f"train={len(train)}  val={len(val)}  test={len(test)}  feature_mode={args.feature_mode}")

    X_train_full = build_features(train, args.feature_mode)
    feat_dim = X_train_full.shape[1]
    X_val_full = build_features(val, args.feature_mode, expected_dim=feat_dim)
    X_test_full = build_features(test, args.feature_mode, expected_dim=feat_dim)
    print(f"Feature dim: {feat_dim}")

    target_columns = get_target_columns(train)
    print(f"Target columns: {target_columns}\n")

    all_results = {}
    for col in target_columns:
        ttype = infer_target_type(col)
        train_idx, y_train = prepare_target(train, col)
        val_idx, y_val = prepare_target(val, col)
        test_idx, y_test = prepare_target(test, col)

        if len(train_idx) < MIN_TRAIN_EXAMPLES:
            print(f"[{col}] SKIPPED -- only {len(train_idx)} non-null training examples "
                  f"(need >= {MIN_TRAIN_EXAMPLES})\n")
            continue
        if ttype == "classification" and len(set(y_train)) < 2:
            print(f"[{col}] SKIPPED -- only one class present in training data\n")
            continue

        X_train = X_train_full[train_idx]
        X_val = X_val_full[val_idx] if len(val_idx) else np.zeros((0, feat_dim))
        X_test = X_test_full[test_idx] if len(test_idx) else np.zeros((0, feat_dim))

        print(f"[{col}] type={ttype}  n_train={len(y_train)}  n_val={len(y_val)}  n_test={len(y_test)}")

        if ttype == "regression":
            results = run_regression_probe(X_train, y_train, X_val, y_val, X_test, y_test)
            for probe_name, metrics in results.items():
                for split_name, m in metrics.items():
                    if m is None:
                        print(f"    {probe_name:6s} {split_name:5s}: (no data)")
                    else:
                        print(f"    {probe_name:6s} {split_name:5s}: "
                              f"R2={m['r2']:.3f}  MAE={m['mae']:.3f}  n={m['n']}")
        else:
            results = run_classification_probe(X_train, y_train, X_val, y_val, X_test, y_test)
            for probe_name, metrics in results.items():
                for split_name, m in metrics.items():
                    if m is None:
                        print(f"    {probe_name:6s} {split_name:5s}: (no data)")
                    else:
                        print(f"    {probe_name:6s} {split_name:5s}: "
                              f"acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  n={m['n']}")

        all_results[col] = {"type": ttype, "results": results}
        print()

    summary_path = os.path.join(args.dataset_dir, "probe_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results written to {summary_path}")


if __name__ == "__main__":
    main()