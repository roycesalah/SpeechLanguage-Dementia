'''
Miscellaneous utils
- loading merged dataframe
- write results
- output predictions
- ensure directory
'''

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def load_metadata(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"ERR: {path} is not CSV")

    colvalidation = ["clip_id", "speaker", "label"]
    for c in colvalidation:
        if c not in df.columns:
            raise ValueError(f"ERR Columns DNE: {c}")

    df = df.copy()
    df["label"] = df["label"].astype(int)

    if "years_pre_dx" not in df.columns:
        df["years_pre_dx"] = np.nan

    return df

def get_core_audio_cols(df):
    # "core" audio signals you mentioned + MFCC1-4 summary stats if present
    base = ["voiced_sec", "wpm", "silence_pct", "mean_sil", "n_sil"]

    mfcc = []
    for i in [1, 2, 3, 4]:
        mfcc.append(f"mfcc{i}_sma3_amean")
        mfcc.append(f"mfcc{i}_sma3_stddevNorm")

    wanted = base + mfcc
    cols = [c for c in wanted if c in df.columns]
    if len(cols) == 0:
        raise ValueError(
            "Core audio columns not found. Expected at least some of: "
            + ", ".join(wanted[:10]) + " ... "
            + "Check your audio CSV column names."
        )
    return cols


def fit_audio_pca_transform(X_tr, X_te, n_components=50, seed=7):
    # Standardize before PCA (important)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # PCA cannot have more components than min(n_samples, n_features)
    max_comp = min(X_tr_s.shape[0], X_tr_s.shape[1])
    k = int(n_components)
    if k > max_comp:
        k = max_comp
    if k < 2:
        k = 2  # keep it meaningful

    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    Z_tr = pca.fit_transform(X_tr_s)
    Z_te = pca.transform(X_te_s)
    return Z_tr, Z_te


def summarize_results_csv(results_csv, print_table=True, save_grid_csv=True, out_path=None, sort_by="mean_auc"):
    """
    Summarize fold-level results.csv into mean/std across folds per (experiment_id, head).

    Expects results.csv columns like:
      experiment_id, fold, head, auc, f1, sens, spec, brier, n_pos, n_neg, n_test
    """
    results_csv = Path(results_csv)
    df = pd.read_csv(results_csv)

    required = {
        "experiment_id",
        "fold",
        "head",
        "auc",
        "f1",
        "sens",
        "spec",
        "brier",
        "n_pos",
        "n_neg",
        "n_test",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results.csv missing required columns: {sorted(missing)}")

    group_cols = ["experiment_id", "head"]
    if "eval_level" in df.columns:
        group_cols.insert(1, "eval_level")

    g = df.groupby(group_cols, dropna=False)

    summary = (
        g.agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_sens=("sens", "mean"),
            std_sens=("sens", "std"),
            mean_spec=("spec", "mean"),
            std_spec=("spec", "std"),
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            folds=("fold", "nunique"),
            total_test=("n_test", "sum"),
            total_pos=("n_pos", "sum"),
            total_neg=("n_neg", "sum"),
        )
        .reset_index()
    )

    summary["pos_rate_test"] = summary["total_pos"] / summary["total_test"]

    if sort_by in summary.columns:
        summary = summary.sort_values(sort_by, ascending=False).reset_index(drop=True)

    if print_table:
        cols_show = [
            "experiment_id",
            "head",
            "mean_auc",
            "std_auc",
            "mean_f1",
            "std_f1",
            "mean_sens",
            "mean_spec",
            "mean_brier",
            "folds",
            "total_test",
            "pos_rate_test",
        ]
        if "eval_level" in summary.columns:
            cols_show.insert(1, "eval_level")
        printable = summary[cols_show].copy()
        for c in [
            "mean_auc",
            "std_auc",
            "mean_f1",
            "std_f1",
            "mean_sens",
            "mean_spec",
            "mean_brier",
            "pos_rate_test",
        ]:
            printable[c] = printable[c].astype(float).round(4)

        print("\n=== Results summary (mean ± std across folds) ===")
        print(printable.to_string(index=False))

    if save_grid_csv:
        if out_path is None:
            out_path = results_csv.parent / "comparison_grid.csv"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)

    return summary

def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def write_results(out_dir: str, results_rows: list[dict]) -> str:
    ensure_out_dir(out_dir)
    path = os.path.join(out_dir, "results.csv")
    pd.DataFrame(results_rows).to_csv(path, index=False)
    return path


def write_predictions(out_dir: str, pred_rows: list[dict]) -> str:
    ensure_out_dir(out_dir)
    path = os.path.join(out_dir, "predictions.csv")
    pd.DataFrame(pred_rows).to_csv(path, index=False)
    return path
