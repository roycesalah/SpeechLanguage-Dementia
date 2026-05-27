#!/usr/bin/env python
# Generate figures

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    from sklearn.metrics import roc_curve, auc as auc_score
except Exception:
    roc_curve = None
    auc_score = None


BIN_ORDER = ["=0", "(0,5]", "(5,10]", "(10,15]"]

ROC_MODELS = [
    ("text_lr", "p_txt", "Text LR"),
    ("audio_lr", "p_aud", "Audio LR"),
    ("early_fusion_lr", "p_fus", "Early fusion LR"),
    ("late_fusion_lr", "p_fus", "Late fusion LR"),
]

AUC_MODELS = [
    ("text_lr", "Text LR"),
    ("audio_lr", "Audio LR"),
    ("late_fusion_lr", "Late fusion LR"),
    ("text_lr_novel_only", "Novel only"),
    ("text_lr_traditional_only", "Traditional only"),
]


def ensure_out_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_both(fig, out_dir: Path, stem: str) -> list[Path]:
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return [png, svg]



def find_prediction_files(out_root: str | Path) -> list[Path]:
    root = Path(out_root)
    if root.is_file() and root.name == "predictions.csv":
        return [root]
    return sorted(root.glob("**/predictions.csv"))


def load_speaker_predictions(out_root: str | Path, experiment_id: str, prob_col: str) -> pd.DataFrame:
    files = find_prediction_files(out_root)
    rows = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        required = {"experiment_id", "fold", "label", prob_col}
        if not required.issubset(df.columns):
            continue
        if "modality" in df.columns:
            df = df[df["modality"].astype(str).str.lower().eq("speaker")].copy()
        df = df[df["experiment_id"].astype(str).eq(experiment_id)].copy()
        df = df[np.isfinite(pd.to_numeric(df[prob_col], errors="coerce"))]
        if not df.empty:
            df["source_file"] = str(p)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "speaker" in out.columns:
        out = out.drop_duplicates(subset=["experiment_id", "fold", "speaker"], keep="first")
    return out


def make_ROC(out_root: str | Path, out_dir: str | Path) -> list[Path]:
    """
    Generate mean-of-folds ROC curves so that the AUC values match
    the per-fold means reported in Table 3 (rather than pooled-prediction AUC).
    """
    if roc_curve is None:
        raise RuntimeError("scikit-learn is required for ROC curve generation.")
    out_dir = ensure_out_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))

    LINE_STYLES = ["-", "--", "-.", ":"]
    mean_fpr_grid = np.linspace(0, 1, 200)

    found_any = False
    for idx, (exp_id, prob_col, label) in enumerate(ROC_MODELS):
        df = load_speaker_predictions(out_root, exp_id, prob_col)
        if df.empty:
            print(f"[skip] No speaker-level predictions found for {exp_id}:{prob_col}")
            continue

        fold_tprs = []
        fold_aucs = []
        for fold_id, fold_df in df.groupby("fold"):
            y = fold_df["label"].astype(int).to_numpy()
            p = pd.to_numeric(fold_df[prob_col], errors="coerce").astype(float).to_numpy()
            mask = np.isfinite(p)
            y, p = y[mask], p[mask]
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            fold_aucs.append(auc_score(fpr, tpr))
            # Interpolate this fold's TPR onto the common FPR grid
            interp_tpr = np.interp(mean_fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0
            fold_tprs.append(interp_tpr)

        if not fold_tprs:
            print(f"[skip] {exp_id}:{prob_col} — no valid folds")
            continue

        mean_tpr = np.mean(fold_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(fold_aucs)

        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        ax.plot(mean_fpr_grid, mean_tpr, linewidth=2.2, linestyle=ls,
                label=f"{label} (AUC={mean_auc:.3f})")
        found_any = True

    if not found_any:
        plt.close(fig)
        raise FileNotFoundError(
            "No matching speaker-level predictions were found. Make sure main_eval.py was run without --no_predictions "
            "and that predictions.csv files exist under --out_root."
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    paths = save_both(fig, out_dir, "ROC_speaker_roc")
    plt.close(fig)
    return paths


def normalize_bin(x) -> str:
    s = str(x).strip().replace(" ", "")
    s = s.replace("0.0", "0").replace("5.0", "5").replace("10.0", "10").replace("15.0", "15")
    s = s.replace("[0,5]", "(0,5]")
    return s


def normalize_model_id(x) -> str:
    s = str(x).strip().lower().replace("†", "")
    if "late_fusion" in s or "late fusion" in s:
        return "late_fusion_lr"
    if "novel" in s:
        return "text_lr_novel_only"
    if "traditional" in s:
        return "text_lr_traditional_only"
    if "audio_lr" in s or s == "audio":
        return "audio_lr"
    if "text_lr" in s or s == "text":
        return "text_lr"
    return s


def infer_auc_columns(df: pd.DataFrame) -> tuple[str, str, Optional[str]]:
    auc_col = "mean_auc" if "mean_auc" in df.columns else "auc"
    std_col = "std_auc" if "std_auc" in df.columns else "std"
    p_col = None
    for c in ["p", "p_value", "p_vs_chance", "pval"]:
        if c in df.columns:
            p_col = c
            break
    return auc_col, std_col, p_col


def load_bin_summary(
    out_root: str | Path | None,
    bin_summary: str | Path | None,
    table4_csv: str | Path | None,
) -> pd.DataFrame:
    candidates = []
    if bin_summary:
        candidates.append(Path(bin_summary))
    if out_root:
        candidates.append(Path(out_root) / "_analysis" / "bin_summary_all_runs.csv")
    if table4_csv:
        candidates.append(Path(table4_csv))

    for p in candidates:
        if p and p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        "Could not find bin summary data. Provide --bin_summary, --table4_csv, "
        "or run analyze.py so out/_analysis/bin_summary_all_runs.csv exists."
    )


def prepare_auc_bin_df(df: pd.DataFrame, runs: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()

    # Filter to specific runs if requested (prevents mixing 5-fold and 10-fold results).
    if runs and "run" in df.columns:
        df = df[df["run"].isin(runs)].copy()
        if df.empty:
            raise ValueError(f"No rows remain after filtering to runs={runs}")

    if "experiment_id" not in df.columns and "Model" in df.columns:
        df["experiment_id"] = df["Model"]
    if "bin" not in df.columns and "Bin" in df.columns:
        df["bin"] = df["Bin"]
    if "mean_auc" not in df.columns and "AUC" in df.columns:
        df["mean_auc"] = df["AUC"]
    if "std_auc" not in df.columns and "SD" in df.columns:
        df["std_auc"] = df["SD"]

    required = {"experiment_id", "bin"}
    if not required.issubset(df.columns):
        raise ValueError(f"Bin summary must contain {required}; got {list(df.columns)}")

    df["model_norm"] = df["experiment_id"].apply(normalize_model_id)
    df["bin_norm"] = df["bin"].apply(normalize_bin)
    auc_col, std_col, p_col = infer_auc_columns(df)
    if auc_col not in df.columns:
        raise ValueError("Could not find an AUC column. Expected mean_auc, auc, or AUC.")
    if std_col not in df.columns:
        df[std_col] = np.nan
    df["_auc"] = pd.to_numeric(df[auc_col], errors="coerce")
    df["_std"] = pd.to_numeric(df[std_col], errors="coerce")
    if p_col:
        df["_p"] = pd.to_numeric(df[p_col], errors="coerce")
    else:
        df["_p"] = np.nan

    # Compute p-values from mean_auc, std_auc, folds via one-sample t-test vs 0.5
    # when no p column was found or all p-values are missing.
    if df["_p"].isna().all() and "folds" in df.columns:
        from scipy import stats as _stats
        n = pd.to_numeric(df["folds"], errors="coerce")
        se = df["_std"] / np.sqrt(n)
        t_stat = (df["_auc"] - 0.5) / se
        df["_p"] = 2 * (1 - _stats.t.cdf(np.abs(t_stat), df=n - 1))
        df.loc[~np.isfinite(df["_p"]), "_p"] = np.nan

    # If the file contains multiple runs/heads per logical model/bin, keep the highest AUC row.
    df = df.sort_values("_auc", ascending=False).drop_duplicates(["model_norm", "bin_norm"], keep="first")
    return df


def make_AUC(
    out_dir: str | Path,
    out_root: str | Path | None = None,
    bin_summary: str | Path | None = None,
    table4_csv: str | Path | None = None,
    runs: list[str] | None = None,
) -> list[Path]:
    out_dir = ensure_out_dir(out_dir)
    raw = load_bin_summary(out_root, bin_summary, table4_csv)
    df = prepare_auc_bin_df(raw, runs=runs)

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = np.arange(len(BIN_ORDER))
    width = 0.15
    offsets = np.linspace(-2, 2, len(AUC_MODELS)) * width

    for offset, (model_id, label) in zip(offsets, AUC_MODELS):
        y = []
        err = []
        pvals = []
        present = []
        for b in BIN_ORDER:
            row = df[(df["model_norm"] == model_id) & (df["bin_norm"] == b)]
            if row.empty:
                y.append(np.nan)
                err.append(np.nan)
                pvals.append(np.nan)
                present.append(False)
            else:
                r = row.iloc[0]
                y.append(float(r["_auc"]))
                err.append(float(r["_std"]) if np.isfinite(r["_std"]) else np.nan)
                pvals.append(float(r["_p"]) if np.isfinite(r["_p"]) else np.nan)
                present.append(True)

        y_arr = np.asarray(y, dtype=float)
        err_arr = np.asarray(err, dtype=float)
        mask = np.isfinite(y_arr)
        if not mask.any():
            continue
        xpos = x[mask] + offset
        # Matplotlib skips nan yerr poorly in some versions; set missing error bars to 0.
        yerr = np.where(np.isfinite(err_arr[mask]), err_arr[mask], 0.0)
        bars = ax.bar(xpos, y_arr[mask], width=width, yerr=yerr, capsize=3, label=label, linewidth=0.7, edgecolor="black")
        for bar, pv in zip(bars, np.asarray(pvals, dtype=float)[mask]):
            if np.isfinite(pv) and pv < 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

    ax.axhline(0.5, linestyle="--", linewidth=1, label="Chance AUC = 0.50")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_ORDER)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Years before diagnosis bin")
    ax.set_ylim(0.25, 0.85)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    # NOTE: Caption text ("Error bars show fold-level SD. * p < 0.05 vs. chance.")
    # belongs in the manuscript \caption{}, not embedded in the figure.
    fig.subplots_adjust(right=0.80, bottom=0.12)
    paths = save_both(fig, out_dir, "AUC_auc_by_temporal_bin")
    plt.close(fig)
    return paths


def write_table4_csv(out_dir: str | Path) -> Path:
    rows = [
        # Model, Head, Bin, AUC, SD, p
        ("text_lr", "txt", "(0,5]", 0.678, 0.095, 0.014),
        ("text_lr", "txt", "(5,10]", 0.585, 0.118, 0.182),
        ("text_lr", "txt", "=0", 0.532, 0.056, 0.266),
        ("text_lr", "txt", "(10,15]", 0.487, 0.113, 0.812),
        ("audio_lr", "aud", "=0", 0.640, 0.088, 0.024),
        ("audio_lr", "aud", "(0,5]", 0.582, 0.045, 0.016),
        ("audio_lr", "aud", "(5,10]", 0.547, 0.090, 0.308),
        ("audio_lr", "aud", "(10,15]", 0.546, 0.200, 0.632),
        ("late_fusion_lr", "fus", "(0,5]", 0.652, 0.055, 0.003),
        ("late_fusion_lr", "fus", "=0", 0.584, 0.051, 0.022),
        ("late_fusion_lr", "fus", "(5,10]", 0.579, 0.110, 0.182),
        ("late_fusion_lr", "fus", "(10,15]", 0.462, 0.136, 0.565),
        ("text_lr_novel_only", "txt", "(0,5]", 0.610, 0.074, 0.029),
        ("text_lr_novel_only", "txt", "(10,15]", 0.570, 0.091, 0.161),
        ("text_lr_traditional_only", "txt", "(0,5]", 0.673, 0.108, 0.023),
        ("text_lr_traditional_only", "txt", "(10,15]", 0.479, 0.111, 0.690),
    ]
    df = pd.DataFrame(rows, columns=["experiment_id", "head", "bin", "mean_auc", "std_auc", "p"])
    out_dir = ensure_out_dir(out_dir)
    p = out_dir / "AUC_table4_values_from_draft.csv"
    df.to_csv(p, index=False)
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--figure", choices=["ROC", "AUC", "all"], default="all")
    p.add_argument("--out_root", default="out", help="Root containing run folders and/or _analysis/bin_summary_all_runs.csv")
    p.add_argument("--out_dir", default="figures", help="Where to save PNG/SVG figures")
    p.add_argument("--bin_summary", default="", help="Optional direct path to bin_summary_all_runs.csv")
    p.add_argument("--table4_csv", default="", help="Optional direct path to a Table 4-style CSV")
    p.add_argument("--runs", nargs="*", default=None,
                   help="Filter bin_summary to these run names (e.g. table3_core_targetSens80 text_ablation_lr)")
    p.add_argument("--write_table4_csv", action="store_true", help="Write a CSV using the current manuscript Table 4 values")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    made = []

    table4_csv = args.table4_csv or ""
    if args.write_table4_csv:
        table4_path = write_table4_csv(out_dir)
        table4_csv = str(table4_path)
        made.append(table4_path)

    if args.figure in {"ROC", "all"}:
        try:
            made.extend(make_ROC(args.out_root, out_dir))
        except FileNotFoundError as e:
            print(f"[Figure 2 skipped] {e}")
    if args.figure in {"AUC", "all"}:
        made.extend(make_AUC(
            out_dir,
            out_root=args.out_root,
            bin_summary=args.bin_summary or None,
            table4_csv=table4_csv or None,
            runs=args.runs,
        ))

    for p in made:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
