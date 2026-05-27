'''
Post-Processing for reading results.csv to produce tables and plots.
in - results.csv
'''

#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# python ModularFramework\analyze.py --out_root ModularFramework\out --print_top 30

METRIC_COLS = ["auc", "f1", "sens", "spec", "brier"]
ID_COLS = ["experiment_id", "head", "fold"]

def parse_bins(bin_spec):
    """
    bin_spec examples:
      "0,5,10,15,18" -> bins: [0,5], (5,10], (10,15], (15,18], (18,inf)
      "0,5,10,15"    -> bins: [0,5], (5,10], (10,15], (15,inf)
    """
    edges = [float(x.strip()) for x in bin_spec.split(",") if x.strip() != ""]
    if len(edges) < 2:
        raise ValueError("bin_spec must have at least two edges, e.g. '0,5,10,15,18'")
    return edges

def assign_bin(years_pre_dx, edges):
    """
    Dementia bins are based on years_pre_dx:
      first bin is inclusive [e0,e1]
      next bins are (e1,e2], ...
      last bin is (elast, inf)
    Returns string label.
    """
    y = float(years_pre_dx)
    if y <= edges[1]:
        return f"[{edges[0]},{edges[1]}]"
    for i in range(1, len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if y > lo and y <= hi:
            return f"({lo},{hi}]"
    return f"({edges[-1]},inf)"

def auc_safe(y_true, p):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, p))
    except Exception:
        return np.nan

def sens_at_fpr(y_true, p, fpr=0.10):
    # Choose threshold so that FPR on negatives is <= fpr (using negatives in the evaluation set then compute sensitivity on positives at that threshold.
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    neg = p[y_true == 0]
    pos = p[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan, np.nan

    # threshold = (1 - fpr) quantile of negative scores
    thr = float(np.quantile(neg, 1.0 - float(fpr)))
    sens = float(np.mean(pos >= thr))
    return sens, thr

def available_heads(pred_df):
    """
    Detect which probability columns exist.
    Returns list of tuples: (head_name, col_name)
    """
    heads = []
    if "p_txt" in pred_df.columns:
        heads.append(("txt", "p_txt"))
    if "p_aud" in pred_df.columns:
        heads.append(("aud", "p_aud"))
    if "p_fus" in pred_df.columns:
        heads.append(("fus", "p_fus"))
    return heads

def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        return None


def summarize_one_results(df):
    # Validate schema
    needed = set(["experiment_id", "head", "fold"] + METRIC_COLS + ["n_pos", "n_neg", "n_test"])
    missing = needed - set(df.columns)
    if missing:
        return None, f"Missing columns: {sorted(missing)}"

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
    return summary, None


def fold_diagnostics(df):
    # Per-fold counts and AUC range
    out = {}
    if "fold" in df.columns and "n_pos" in df.columns and "n_neg" in df.columns:
        diag_df = df
        if "eval_level" in df.columns:
            diag_df = df[df["eval_level"] == "clip"].copy()
        per_fold = (
            diag_df.groupby("fold", dropna=False)
              .agg(n_pos=("n_pos", "max"), n_neg=("n_neg", "max"), n_test=("n_test", "max"))
              .reset_index()
              .sort_values("fold")
        )
        out["per_fold_counts"] = per_fold

    if "auc" in df.columns and "fold" in df.columns:
        diag_df = df
        if "eval_level" in df.columns:
            diag_df = df[df["eval_level"] == "clip"].copy()
        auc_by_fold = (
            diag_df.groupby("fold", dropna=False)
              .agg(mean_auc=("auc", "mean"))
              .reset_index()
              .sort_values("fold")
        )
        out["auc_by_fold"] = auc_by_fold

    return out

def find_runs(out_root):
    out_root = Path(out_root)
    if not out_root.exists():
        raise FileNotFoundError(f"out directory not found: {out_root}")

    runs = []
    for d in out_root.iterdir():
        if not d.is_dir():
            continue
        results_p = d / "results.csv"
        if results_p.exists():
            runs.append(d)
    return runs

def parse_edges(edge_str):
    edges = [float(x.strip()) for x in edge_str.split(",") if x.strip() != ""]
    if len(edges) < 2:
        raise ValueError("Need at least two edges, e.g. '0,5,10,15'")
    return edges

def bin_label(y, edges, point_bins=None):
    y = float(y)
    point_bins = point_bins or set()

    # exact point bins
    for pb in point_bins:
        if abs(y - pb) < 1e-9:
            # keep it clean in output
            if float(int(pb)) == pb:
                return f"={int(pb)}"
            return f"={pb}"

    # regular bins
    for i in range(0, len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if y > lo and y <= hi:
            return f"({lo},{hi}]"

    return f"({edges[-1]},inf)"

def available_heads(pred):
    heads = []
    if "p_txt" in pred.columns: heads.append(("txt", "p_txt"))
    if "p_aud" in pred.columns: heads.append(("aud", "p_aud"))
    if "p_fus" in pred.columns: heads.append(("fus", "p_fus"))
    return heads

def auc_safe(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))

def sens_at_fpr_from_scores(p_pos, p_neg, fpr=0.10):
    p_pos = np.asarray(p_pos).astype(float)
    p_neg = np.asarray(p_neg).astype(float)
    if len(p_pos) == 0 or len(p_neg) == 0:
        return np.nan, np.nan
    thr = float(np.quantile(p_neg, 1.0 - float(fpr)))
    sens = float(np.mean(p_pos >= thr))
    return sens, thr

def compute_time_to_dx_table(pred_path, edges, point_bins=None, fpr=0.10):
    pred = pd.read_csv(pred_path)
    if "modality" in pred.columns:
        pred = pred[pred["modality"] == "clip"].copy()
    need = {"experiment_id", "fold", "label"}
    if not need.issubset(set(pred.columns)):
        return None

    if "years_pre_dx" not in pred.columns:
        return None

    heads = available_heads(pred)
    if not heads:
        return None

    point_bins = point_bins or set()

    out = []
    for (exp_id, fold), df in pred.groupby(["experiment_id", "fold"], dropna=False):
        ctrl = df[df["label"] == 0]
        dem = df[df["label"] == 1].copy()

        if len(ctrl) == 0 or len(dem) == 0:
            continue

        dem["years_pre_dx"] = pd.to_numeric(dem["years_pre_dx"], errors="coerce")
        dem = dem[dem["years_pre_dx"].notna()].copy()
        if len(dem) == 0:
            continue

        dem["dx_bin"] = dem["years_pre_dx"].astype(float).apply(
            lambda yy: bin_label(yy, edges, point_bins)
        )

        for head, col in heads:
            if col not in df.columns:
                continue

            p_neg = ctrl[col].to_numpy()
            if np.all(pd.isna(p_neg)):
                continue

            for b, dem_b in dem.groupby("dx_bin", dropna=False):
                p_pos = dem_b[col].to_numpy()
                if np.all(pd.isna(p_pos)):
                    continue

                y = np.concatenate([np.ones(len(p_pos), dtype=int), np.zeros(len(p_neg), dtype=int)])
                p = np.concatenate([p_pos.astype(float), p_neg.astype(float)])

                auc = auc_safe(y, p)
                sens, thr = sens_at_fpr_from_scores(p_pos, p_neg, fpr=fpr)

                out.append({
                    "experiment_id": exp_id,
                    "fold": int(fold),
                    "head": head,
                    "bin": str(b),
                    "auc": auc,
                    "sens_at_fpr": sens,
                    "thr_at_fpr": thr,
                    "n_pos": int(len(p_pos)),
                    "n_neg": int(len(p_neg)),
                    "fpr_target": float(fpr),
                })

    if not out:
        return None
    return pd.DataFrame(out)

def parse_point_bins(s):
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return set(float(x.strip()) for x in s.split(",") if x.strip() != "")

def aggregate_bins_across_folds(tdx_df):
    g = tdx_df.groupby(["experiment_id","head","bin"], dropna=False)
    agg = g.agg(
        mean_auc=("auc","mean"),
        std_auc=("auc","std"),
        mean_sens_at_fpr=("sens_at_fpr","mean"),
        std_sens_at_fpr=("sens_at_fpr","std"),
        folds=("fold","nunique"),
        n_pos_total=("n_pos","sum"),
        n_neg_total=("n_neg","sum"),
    ).reset_index()
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out", help="Path to out directory (default: out)")
    ap.add_argument("--min_folds", type=int, default=2, help="Minimum folds required to include a run")
    ap.add_argument("--print_top", type=int, default=20, help="How many leaderboard rows to print")
    ap.add_argument("--fpr", type=float, default=0.10, help="FPR target for sens_at_fpr")
    ap.add_argument("--bin_spec", default="0,5,10,15", help="Edges for bins, e.g. '0,5,10,15'")
    ap.add_argument("--point_bins", default="0", help="Comma-separated exact bins to isolate, e.g. '0'")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    runs = find_runs(out_root)

    analysis_dir = out_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    notes = []

    if not runs:
        print(f"No runs found under {out_root} (expected subfolders containing results.csv).")
        return

    for run_dir in sorted(runs):
        run_name = run_dir.name
        df = read_csv_safe(run_dir / "results.csv")
        if df is None:
            notes.append(f"[{run_name}] Failed to read results.csv")
            continue

        summary, err = summarize_one_results(df)
        if err:
            notes.append(f"[{run_name}] {err}")
            continue

        # Filter out incomplete runs
        if summary["folds"].max() < args.min_folds:
            notes.append(f"[{run_name}] Skipped (folds<{args.min_folds})")
            continue

        summary.insert(0, "run", run_name)
        all_rows.append(summary)

        # Diagnostics to note fold imbalance + high variance
        diags = fold_diagnostics(df)
        if "per_fold_counts" in diags:
            per_fold = diags["per_fold_counts"]
            if len(per_fold) > 0:
                min_pos = int(per_fold["n_pos"].min())
                max_pos = int(per_fold["n_pos"].max())
                if min_pos == 0:
                    notes.append(f"[{run_name}] WARNING: fold has n_pos=0 (AUC may be NaN).")
                if max_pos - min_pos >= 10:
                    notes.append(f"[{run_name}] NOTE: fold pos count varies significantly (min_pos={min_pos}, max_pos={max_pos}).")

        # Find best row in this run
        best = summary.sort_values(["mean_auc", "mean_f1"], ascending=False).head(1)
        if len(best) == 1:
            b = best.iloc[0]
            level_str = f"{b['eval_level']}:" if "eval_level" in best.columns else ""
            notes.append(
                f"[{run_name}] best={b['experiment_id']}:{level_str}{b['head']} "
                f"mean_auc={b['mean_auc']:.4f} std_auc={0.0 if pd.isna(b['std_auc']) else b['std_auc']:.4f} "
                f"mean_f1={b['mean_f1']:.4f}"
            )

    if not all_rows:
        print("No valid runs to summarize. See out/_analysis/run_notes.txt")
        (analysis_dir / "run_notes.txt").write_text("\n".join(notes), encoding="utf-8")
        return

    all_summary = pd.concat(all_rows, ignore_index=True)
    all_summary.to_csv(analysis_dir / "all_runs_summary.csv", index=False)

    # Leaderboard - pick best head per (run, experiment_id, head) already present; just sort all rows
    leaderboard = all_summary.copy()
    leaderboard = leaderboard.sort_values(["mean_auc", "mean_f1"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(analysis_dir / "leaderboard.csv", index=False)

    # bin leaderboards across all runs
    edges = parse_edges(args.bin_spec)  # e.g. "0,5,10,15"
    bin_rows = []

    for run_dir in sorted(runs):
        pred_path = Path(run_dir) / "predictions.csv"
        if not pred_path.exists():
            continue

        point_bins = parse_point_bins(args.point_bins)

        tdx = compute_time_to_dx_table(
            pred_path,
            edges,
            point_bins=point_bins,
            fpr=args.fpr)

        if tdx is None:
            continue

        agg = aggregate_bins_across_folds(tdx)
        agg.insert(0, "run", Path(run_dir).name)
        bin_rows.append(agg)

        # also write per-run
        tdx.to_csv(Path(run_dir) / "time_to_dx.csv", index=False)
        agg.to_csv(Path(run_dir) / "time_to_dx_summary.csv", index=False)

    if bin_rows:
        all_bins = pd.concat(bin_rows, ignore_index=True)

        analysis_dir = Path(args.out_root) / "_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        all_bins.to_csv(analysis_dir / "bin_summary_all_runs.csv", index=False)

        # leaderboard per bin - sort within each bin
        all_bins_sorted = all_bins.sort_values(["bin","mean_auc","mean_sens_at_fpr"], ascending=[True,False,False])
        all_bins_sorted.to_csv(analysis_dir / "bin_leaderboard.csv", index=False)

        top_n = args.print_top
        lines = []
        for b, dfb in all_bins_sorted.groupby("bin", dropna=False):
            lines.append(f"\n=== BIN {b} (top {top_n}) ===")
            view = dfb[["run","experiment_id","head","mean_auc","std_auc","mean_sens_at_fpr","folds","n_pos_total","n_neg_total"]].head(top_n).copy()
            for c in ["mean_auc","std_auc","mean_sens_at_fpr"]:
                view[c] = view[c].astype(float).round(4)
            lines.append(view.to_string(index=False))
        (analysis_dir / "bin_topN.txt").write_text("\n".join(lines), encoding="utf-8")

        print(f"Wrote: {analysis_dir / 'bin_summary_all_runs.csv'}")
        print(f"Wrote: {analysis_dir / 'bin_leaderboard.csv'}")
        print(f"Wrote: {analysis_dir / 'bin_topN.txt'}")

    # Write notes
    (analysis_dir / "run_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    # Print top
    show_cols = ["run", "experiment_id", "head", "mean_auc", "std_auc", "mean_f1", "mean_sens", "mean_spec", "mean_brier", "folds", "pos_rate_test"]
    if "eval_level" in leaderboard.columns:
        show_cols.insert(2, "eval_level")
    show = leaderboard[show_cols].head(args.print_top).copy()

    # Rounding
    for c in ["mean_auc", "std_auc", "mean_f1", "mean_sens", "mean_spec", "mean_brier", "pos_rate_test"]:
        show[c] = show[c].astype(float).round(4)

    print("\n=== Leaderboard (top rows across all out/* runs) ===")
    print(show.to_string(index=False))

    print(f"\nWrote: {analysis_dir / 'all_runs_summary.csv'}")
    print(f"Wrote: {analysis_dir / 'leaderboard.csv'}")
    print(f"Wrote: {analysis_dir / 'run_notes.txt'}")


if __name__ == "__main__":
    main()
