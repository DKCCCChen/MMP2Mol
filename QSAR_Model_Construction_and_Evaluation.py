#!/usr/bin/env python
"""Construct and evaluate target-specific deployment QSAR classifiers.

This script implements the model-selection workflow described for MMP2Mol:

* four molecular representations: MACCS, ECFP6, FCFP6, and their concatenation;
* six prespecified classifiers: RF, XGBoost, SVM, Bernoulli NB, logistic
  regression, and a multilayer perceptron;
* five-fold scaffold-grouped cross-validation with a fixed 0.5 decision
  threshold;
* selection by mean ROC-AUC, followed by PR-AUC, MCC, and Brier score;
* refitting of the selected configuration on the complete curated dataset.

The input CSV must contain a SMILES column and a binary activity-label column.
An optional pChEMBL/Pa column is used to resolve duplicate structures by
retaining the record with the highest reported activity, as described in the
manuscript.

Example
-------
python QSAR_Model_Construction_and_Evaluation.py \
    --input data/input.csv \
    --output-dir outputs/input_qsar \
    --target input \
    --seed 42
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# Newer RDKit releases emit one deprecation warning per Morgan fingerprint.
# The manuscript-compatible RDKit 2022.03 API remains intentionally supported.
RDLogger.DisableLog("rdApp.warning")


DESCRIPTOR_ORDER = ("MACCS", "ECFP6", "FCFP6", "Combined")
CLASSIFIER_ORDER = ("RF", "XGBoost", "SVM", "NB", "LogisticRegression", "MLP")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Five-fold scaffold-grouped selection and full-data fitting of QSAR models."
    )
    parser.add_argument("--input", default="data/input.csv", help="Input CSV file.")
    parser.add_argument(
        "--output-dir", default="outputs/qsar", help="Directory for tables, plots, and model files."
    )
    parser.add_argument("--target", default=None, help="Target name written to output tables.")
    parser.add_argument("--smiles-column", default=None, help="Override automatic SMILES-column detection.")
    parser.add_argument("--label-column", default=None, help="Override automatic label-column detection.")
    parser.add_argument(
        "--activity-column",
        default=None,
        help="Optional pChEMBL/Pa column used to resolve duplicate structures.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of scaffold-grouped CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for supported models.")
    parser.add_argument(
        "--descriptors",
        default=",".join(DESCRIPTOR_ORDER),
        help="Comma-separated subset used mainly for diagnostic runs.",
    )
    parser.add_argument(
        "--classifiers",
        default=",".join(CLASSIFIER_ORDER),
        help="Comma-separated subset used mainly for diagnostic runs.",
    )
    return parser.parse_args()


def find_column(columns, explicit, candidates, kind):
    if explicit is not None:
        if explicit not in columns:
            raise ValueError("Requested {0} column '{1}' was not found.".format(kind, explicit))
        return explicit
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(
        "Could not detect a {0} column. Available columns: {1}".format(kind, list(columns))
    )


def find_optional_column(columns, explicit, candidates):
    if explicit is not None:
        if explicit not in columns:
            raise ValueError("Requested activity column '{0}' was not found.".format(explicit))
        return explicit
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def parse_binary_labels(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
    else:
        mapping = {
            "0": 0,
            "1": 1,
            "inactive": 0,
            "active": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
        }
        values = series.astype(str).str.strip().str.lower().map(mapping)
    if values.isna().any():
        bad = series[values.isna()].astype(str).head(5).tolist()
        raise ValueError("Unrecognized or missing activity labels, for example: {0}".format(bad))
    values = values.astype(int)
    invalid = sorted(set(values.tolist()) - {0, 1})
    if invalid:
        raise ValueError("Labels must be binary 0/1; found: {0}".format(invalid))
    return values


def standardize_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def curate_input(path, smiles_column=None, label_column=None, activity_column=None):
    raw = pd.read_csv(path)
    smiles_col = find_column(
        raw.columns, smiles_column, ("smiles", "canonical_smiles", "canonical smiles"), "SMILES"
    )
    label_col = find_column(
        raw.columns, label_column, ("label", "activity_label", "class"), "label"
    )
    activity_col = find_optional_column(
        raw.columns,
        activity_column,
        ("pchembl", "pchembl_value", "pchembl value", "pa", "pactivity"),
    )

    work = pd.DataFrame()
    work["source_smiles"] = raw[smiles_col]
    work["label"] = parse_binary_labels(raw[label_col])
    if activity_col is not None:
        work["activity_value"] = pd.to_numeric(raw[activity_col], errors="coerce")
    work["smiles"] = work["source_smiles"].map(standardize_smiles)
    invalid_count = int(work["smiles"].isna().sum())
    work = work.dropna(subset=["smiles"]).copy()

    if activity_col is not None:
        # Stable sorting ensures that the highest pChEMBL/Pa record is retained.
        work = work.sort_values(
            ["smiles", "activity_value"], ascending=[True, False], na_position="last", kind="mergesort"
        )
        curated = work.drop_duplicates("smiles", keep="first").copy()
    else:
        label_counts = work.groupby("smiles")["label"].nunique()
        conflicts = label_counts[label_counts > 1]
        if len(conflicts):
            raise ValueError(
                "Found {0} canonical structures with conflicting labels, but no pChEMBL/Pa "
                "column was available to resolve them.".format(len(conflicts))
            )
        curated = work.drop_duplicates("smiles", keep="first").copy()

    curated = curated.sort_values("smiles").reset_index(drop=True)
    if curated["label"].nunique() != 2:
        raise ValueError("The curated dataset must contain both active and inactive compounds.")
    report = {
        "raw_records": int(len(raw)),
        "invalid_or_unusable_smiles": invalid_count,
        "valid_records_before_deduplication": int(len(work)),
        "canonical_unique_compounds": int(len(curated)),
        "active_compounds": int(curated["label"].sum()),
        "inactive_compounds": int((1 - curated["label"]).sum()),
        "active_fraction": float(curated["label"].mean()),
        "smiles_column": str(smiles_col),
        "label_column": str(label_col),
        "activity_column": None if activity_col is None else str(activity_col),
    }
    return curated, report


def scaffold_for_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    # Empty Murcko scaffolds are common for acyclic molecules. Keeping each
    # acyclic structure as its own group prevents one artificial mega-group.
    return scaffold if scaffold else "ACYCLIC::{0}".format(smiles)


def _fallback_scaffold_assignment(y, groups, n_splits, seed):
    group_to_indices = defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_indices[group].append(idx)
    if len(group_to_indices) < n_splits:
        raise ValueError("Fewer unique scaffold groups than requested CV folds.")

    totals = np.array([(y == 0).sum(), (y == 1).sum()], dtype=float)
    target_class = totals / float(n_splits)
    target_size = len(y) / float(n_splits)
    best_assignment = None
    best_score = float("inf")

    for attempt in range(64):
        rng = random.Random(seed + attempt)
        items = []
        for group, indices in group_to_indices.items():
            counts = np.bincount(y[indices], minlength=2).astype(float)
            items.append((group, indices, counts, rng.random()))
        items.sort(key=lambda x: (-len(x[1]), -max(x[2]), x[3]))

        fold_groups = [[] for _ in range(n_splits)]
        fold_counts = np.zeros((n_splits, 2), dtype=float)
        fold_sizes = np.zeros(n_splits, dtype=float)

        for item_index, (group, indices, counts, _) in enumerate(items):
            if item_index < n_splits:
                chosen = item_index
            else:
                candidate_scores = []
                for fold in range(n_splits):
                    trial_counts = fold_counts.copy()
                    trial_sizes = fold_sizes.copy()
                    trial_counts[fold] += counts
                    trial_sizes[fold] += len(indices)
                    class_term = np.sum(
                        ((trial_counts - target_class) / np.maximum(target_class, 1.0)) ** 2
                    )
                    size_term = np.sum(
                        ((trial_sizes - target_size) / max(target_size, 1.0)) ** 2
                    )
                    candidate_scores.append((class_term + 0.25 * size_term, trial_sizes[fold], fold))
                chosen = min(candidate_scores)[2]
            fold_groups[chosen].append(group)
            fold_counts[chosen] += counts
            fold_sizes[chosen] += len(indices)

        missing_class_penalty = float(np.sum(fold_counts == 0)) * 1000000.0
        balance_score = np.sum(
            ((fold_counts - target_class) / np.maximum(target_class, 1.0)) ** 2
        ) + 0.25 * np.sum(((fold_sizes - target_size) / max(target_size, 1.0)) ** 2)
        score = missing_class_penalty + balance_score
        if score < best_score:
            best_score = score
            best_assignment = fold_groups

    if best_assignment is None:
        raise RuntimeError("Unable to construct scaffold-grouped folds.")

    splits = []
    all_indices = np.arange(len(y))
    group_array = np.asarray(groups, dtype=object)
    for fold_groups in best_assignment:
        test_mask = np.isin(group_array, np.asarray(fold_groups, dtype=object))
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]
        if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            raise ValueError(
                "A scaffold-grouped fold lacks one activity class. Use fewer folds or add data."
            )
        splits.append((train_idx, test_idx))
    return splits


def make_scaffold_grouped_splits(y, groups, n_splits, seed):
    # Prefer sklearn's implementation when available; retain a deterministic
    # fallback for the Python 3.7 / scikit-learn 1.0 environment.
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
    except (ImportError, TypeError):
        splits = _fallback_scaffold_assignment(y, groups, n_splits, seed)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups[i] for i in train_idx)
        test_groups = set(groups[i] for i in test_idx)
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError("Scaffold leakage detected in fold {0}.".format(fold))
    return splits


def bitvect_to_array(fp):
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calculate_descriptors(smiles_list):
    maccs = []
    ecfp6 = []
    fcfp6 = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Descriptor calculation failed for standardized SMILES: {0}".format(smiles))
        maccs.append(bitvect_to_array(MACCSkeys.GenMACCSKeys(mol)))
        ecfp6.append(
            bitvect_to_array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048))
        )
        fcfp6.append(
            bitvect_to_array(
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=3, nBits=2048, useFeatures=True
                )
            )
        )
    features = {
        "MACCS": np.asarray(maccs, dtype=np.uint8),
        "ECFP6": np.asarray(ecfp6, dtype=np.uint8),
        "FCFP6": np.asarray(fcfp6, dtype=np.uint8),
    }
    features["Combined"] = np.hstack(
        [features["MACCS"], features["ECFP6"], features["FCFP6"]]
    )
    return features


def make_models(seed, n_jobs, selected_classifiers=None):
    selected_classifiers = set(selected_classifiers or CLASSIFIER_ORDER)
    if "XGBoost" in selected_classifiers and XGBClassifier is None:
        raise ImportError("xgboost is required for the prespecified 24-model evaluation.")
    candidates = {
        "RF": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=n_jobs
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "NB": BernoulliNB(),
        "LogisticRegression": LogisticRegression(
            solver="liblinear", max_iter=2000, random_state=seed
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=seed
        ),
    }
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=n_jobs,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
    return {name: candidates[name] for name in selected_classifiers}


def positive_probability(model, x):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(decision, -35.0, 35.0)))
    raise TypeError("The classifier does not provide probabilities or decision scores.")


def metric_dict(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_configuration(model, x, y, splits, threshold):
    oof_probability = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        fitted = clone(model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fitted.fit(x[train_idx], y[train_idx])
        probability = positive_probability(fitted, x[test_idx])
        oof_probability[test_idx] = probability
        metrics = metric_dict(y[test_idx], probability, threshold)
        metrics.update(
            {
                "fold": fold,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "test_active_fraction": float(y[test_idx].mean()),
            }
        )
        fold_rows.append(metrics)
    if np.isnan(oof_probability).any():
        raise RuntimeError("Some compounds did not receive an out-of-fold prediction.")
    pooled = metric_dict(y, oof_probability, threshold)
    return fold_rows, pooled, oof_probability


def mean_sd(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def calculate_ad_threshold(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    if len(fingerprints) < 2:
        return float("nan")
    nearest = []
    for i, fp in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fingerprints)
        similarities[i] = -1.0
        nearest.append(max(similarities))
    return float(np.percentile(np.asarray(nearest, dtype=float), 5.0))


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Not JSON serializable: {0}".format(type(value)))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    descriptors = [x.strip() for x in args.descriptors.split(",") if x.strip()]
    classifiers = [x.strip() for x in args.classifiers.split(",") if x.strip()]
    unknown_descriptors = sorted(set(descriptors) - set(DESCRIPTOR_ORDER))
    unknown_classifiers = sorted(set(classifiers) - set(CLASSIFIER_ORDER))
    if unknown_descriptors or unknown_classifiers:
        raise ValueError(
            "Unknown descriptors {0} or classifiers {1}.".format(
                unknown_descriptors, unknown_classifiers
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = args.target or Path(args.input).stem

    curated, curation_report = curate_input(
        args.input, args.smiles_column, args.label_column, args.activity_column
    )
    curated.to_csv(output_dir / "curated_input.csv", index=False)
    smiles = curated["smiles"].tolist()
    y = curated["label"].to_numpy(dtype=int)
    groups = [scaffold_for_smiles(s) for s in smiles]
    splits = make_scaffold_grouped_splits(y, groups, args.folds, args.seed)

    assignment = np.full(len(y), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splits, start=1):
        assignment[test_idx] = fold
    pd.DataFrame(
        {
            "smiles": smiles,
            "label": y,
            "scaffold": groups,
            "cv_fold": assignment,
        }
    ).to_csv(output_dir / "scaffold_grouped_fold_assignments.csv", index=False)

    feature_sets = calculate_descriptors(smiles)
    models = make_models(args.seed, args.n_jobs, classifiers)
    summary_rows = []
    fold_metric_rows = []
    oof_rows = []
    curves = {}

    for descriptor in descriptors:
        x = feature_sets[descriptor]
        for classifier in classifiers:
            print("Evaluating {0} + {1}...".format(descriptor, classifier), flush=True)
            fold_rows, pooled, oof_probability = evaluate_configuration(
                models[classifier], x, y, splits, args.threshold
            )
            combination = "{0}-{1}".format(descriptor, classifier)
            for row in fold_rows:
                row.update(
                    {"target": target, "descriptor": descriptor, "classifier": classifier}
                )
                fold_metric_rows.append(row)

            summary = {
                "target": target,
                "descriptor": descriptor,
                "classifier": classifier,
                "combination": combination,
                "features": int(x.shape[1]),
                "full_data_n": int(len(y)),
                "active_fraction": float(y.mean()),
            }
            for metric in (
                "roc_auc",
                "pr_auc",
                "mcc",
                "balanced_accuracy",
                "f1",
                "brier",
            ):
                mean_value, sd_value = mean_sd(fold_rows, metric)
                summary[metric + "_mean"] = mean_value
                summary[metric + "_sd"] = sd_value
                summary[metric + "_pooled_oof"] = pooled[metric]
            summary_rows.append(summary)

            fpr, tpr, _ = roc_curve(y, oof_probability)
            curves[combination] = (fpr, tpr, pooled["roc_auc"])
            for index, probability in enumerate(oof_probability):
                oof_rows.append(
                    {
                        "target": target,
                        "descriptor": descriptor,
                        "classifier": classifier,
                        "smiles": smiles[index],
                        "label": int(y[index]),
                        "cv_fold": int(assignment[index]),
                        "probability": float(probability),
                        "prediction": int(probability >= args.threshold),
                    }
                )

    results = pd.DataFrame(summary_rows)
    # Larger values are preferred except for Brier score.
    results = results.sort_values(
        ["roc_auc_mean", "pr_auc_mean", "mcc_mean", "brier_mean"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    results.insert(0, "selection_rank", np.arange(1, len(results) + 1))
    results.to_csv(output_dir / "all_model_cv_results.csv", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(output_dir / "fold_level_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(output_dir / "out_of_fold_predictions.csv", index=False)

    best = results.iloc[0]
    best_descriptor = str(best["descriptor"])
    best_classifier = str(best["classifier"])
    best_model = clone(models[best_classifier])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_model.fit(feature_sets[best_descriptor], y)

    ad_threshold = calculate_ad_threshold(smiles)
    model_bundle = {
        "model": best_model,
        "descriptor": best_descriptor,
        "classifier": best_classifier,
        "threshold": float(args.threshold),
        "target": target,
        "training_smiles": smiles,
        "ad_definition": "maximum ECFP4 Tanimoto similarity to full-data training set",
        "ad_threshold": ad_threshold,
        "standardization": "RDKit Cleanup, largest-fragment parent, uncharging, canonical isomeric SMILES",
        "selection_rule": "mean scaffold-grouped CV ROC-AUC; PR-AUC, MCC, and Brier tie-breakers",
        "random_seed": int(args.seed),
        "cv_folds": int(args.folds),
    }
    joblib.dump(model_bundle, output_dir / "best_qsar_model.pkl", compress=3)

    best_summary = best.to_dict()
    best_summary.update(
        {
            "ad_threshold": ad_threshold,
            "curation": curation_report,
            "input_file": os.path.abspath(args.input),
        }
    )
    with open(output_dir / "best_model_summary.json", "w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2, ensure_ascii=False, default=json_ready)

    best_combination = str(best["combination"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for combination, (fpr, tpr, auc_value) in curves.items():
        if combination == best_combination:
            continue
        ax.plot(fpr, tpr, color="#C9CED6", linewidth=1.0, alpha=0.60)
    fpr, tpr, auc_value = curves[best_combination]
    ax.plot(
        fpr,
        tpr,
        color="#123B69",
        linewidth=3.0,
        label="Selected: {0} (OOF AUC={1:.3f})".format(best_combination, auc_value),
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("{0}: five-fold scaffold-grouped QSAR selection".format(target))
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "model_selection_roc.png", dpi=300)
    fig.savefig(output_dir / "model_selection_roc.pdf")
    plt.close(fig)

    print("\nSelected configuration: {0}".format(best_combination))
    print("Mean CV ROC-AUC: {0:.4f}".format(float(best["roc_auc_mean"])))
    print("Pooled OOF ROC-AUC: {0:.4f}".format(float(best["roc_auc_pooled_oof"])))
    print("Model bundle: {0}".format(output_dir / "best_qsar_model.pkl"))
    print("All outputs: {0}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
