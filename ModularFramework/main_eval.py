'''
Primary infra for pipeline logic
TODO:
- load df
- define outer/inner CVs
- iter for folds
- call experiments
- comp metrics
- write out to results.csv and predictions.csv
'''

# main_eval.py
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from utils import load_metadata, write_predictions, write_results, summarize_results_csv
from metrics import compute_metrics, select_threshold_for_target_sensitivity
from experiments import EXPERIMENTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True, help="Metadata (.csv)")
    p.add_argument("--lexical_csv", default="", help="Path: text_features_lexical.csv")
    p.add_argument("--sbert_entropy_csv", default="", help="Path: text_features_sbert_entropy.csv")
    p.add_argument("--audio_csv", default="", help="Path to audio_features.csv")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--experiments", default="", help="Comma-separated experiment names. Leave empty for all.")
    p.add_argument("--outer_splits", type=int, default=5)
    p.add_argument("--inner_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no_predictions", action="store_true")
    p.add_argument("--threshold_policy", choices=["fixed_0.5", "target_sens"], default="fixed_0.5")
    p.add_argument("--target_sensitivity", type=float, default=0.80)
    p.add_argument("--embed_csv", default="", help="Embeddings features CSV keyed by clip_id")
    p.add_argument("--near_dx_years", type=float, default=5.0, help="Years-pre-dx cutoff for time-aware weighting")
    p.add_argument("--far_positive_weight", type=float, default=0.5, help="Relative weight for positives beyond near_dx_years")
    return p.parse_args()


def make_group_splitter(n_splits: int, seed: int):
    if StratifiedGroupKFold is None:
        return GroupKFold(n_splits=n_splits)

    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except TypeError:
        return StratifiedGroupKFold(n_splits=n_splits)


def aggregate_to_speaker_level(df: pd.DataFrame, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = df.loc[:, ["speaker", "label"]].copy()
    tmp["p"] = np.asarray(probs, dtype=float)
    speaker_df = (
        tmp.groupby("speaker", dropna=False)
        .agg(label=("label", "max"), p=("p", "mean"))
        .reset_index()
    )
    return (
        speaker_df["label"].to_numpy(dtype=int),
        speaker_df["p"].to_numpy(dtype=float),
    )


def aggregate_prediction_rows(
    df: pd.DataFrame,
    experiment_id: str,
    fold: int,
    p_txt: np.ndarray,
    p_aud: np.ndarray,
    p_fus: np.ndarray,
) -> list[dict]:
    tmp = df.loc[:, ["speaker", "label", "years_pre_dx"]].copy()
    tmp["p_txt"] = np.asarray(p_txt, dtype=float)
    tmp["p_aud"] = np.asarray(p_aud, dtype=float)
    tmp["p_fus"] = np.asarray(p_fus, dtype=float)

    speaker_df = (
        tmp.groupby("speaker", dropna=False)
        .agg(
            label=("label", "max"),
            years_pre_dx=("years_pre_dx", "min"),
            p_txt=("p_txt", "mean"),
            p_aud=("p_aud", "mean"),
            p_fus=("p_fus", "mean"),
        )
        .reset_index()
    )

    rows = []
    for _, row in speaker_df.iterrows():
        rows.append(
            {
                "experiment_id": experiment_id,
                "fold": fold,
                "clip_id": np.nan,
                "speaker": row["speaker"],
                "label": int(row["label"]),
                "years_pre_dx": row["years_pre_dx"],
                "p_txt": float(row["p_txt"]) if np.isfinite(row["p_txt"]) else np.nan,
                "p_aud": float(row["p_aud"]) if np.isfinite(row["p_aud"]) else np.nan,
                "p_fus": float(row["p_fus"]) if np.isfinite(row["p_fus"]) else np.nan,
                "modality": "speaker",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    df = load_metadata(args.metadata)
    df = df.sort_values(["speaker", "clip_id"]).reset_index(drop=True)


    selected = [e.strip() for e in args.experiments.split(",") if e.strip()] if args.experiments else list(EXPERIMENTS.keys())
    missing = [e for e in selected if e not in EXPERIMENTS]
    if missing:
        raise ValueError(f"Unknown experiments: {missing}. Available: {list(EXPERIMENTS.keys())}")

    lexical_experiments = {
    "text_lr",
    "text_lexical_embed_lr",
    "text_lr_speaker",
    "text_lr_timeaware",
    "late_fusion_lr",
    "early_fusion_lr_speaker",
    "early_fusion_lr_timeaware",
    "early_fusion_audio_pca_lr",
    }

    audio_experiments = {
        "audio_lr",
        "audio_lr_speaker",
        "audio_lr_timeaware",
        "late_fusion_lr",
        "early_fusion_lr_speaker",
        "early_fusion_lr_timeaware",
        "audio_core_lr",
        "early_fusion_audio_pca_lr",
    }
    requires_lexical = any(e in lexical_experiments for e in selected)
    requires_audio = any(e in audio_experiments for e in selected)
    if requires_lexical and not args.lexical_csv:
        raise ValueError("--lexical_csv is required for the selected text/fusion experiment(s)")
    if requires_audio and not args.audio_csv:
        raise ValueError("--audio_csv is required for the selected audio/fusion experiment(s)")

    outer_cv = make_group_splitter(n_splits=args.outer_splits, seed=args.seed)

    y_all = df["label"].to_numpy(dtype=int)
    groups_all = df["speaker"].astype(str).to_numpy()

    results_rows: list[dict] = []
    pred_rows: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(df, y_all, groups=groups_all), start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)

        # hard leakage check
        assert set(train_df["speaker"].astype(str)).isdisjoint(set(test_df["speaker"].astype(str)))

        inner_cv = make_group_splitter(n_splits=args.inner_splits, seed=args.seed + fold)
        y_tr = train_df["label"].to_numpy(dtype=int)
        g_tr = train_df["speaker"].astype(str).to_numpy()
        inner_splits_idx = list(inner_cv.split(train_df, y_tr, groups=g_tr))

        for exp_name in selected:
            print(f"running {exp_name}", flush=True)
            exp_fn = EXPERIMENTS[exp_name]

            test_probs, oof_probs = exp_fn(
                train_df=train_df,
                test_df=test_df,
                inner_splits_idx=inner_splits_idx,
                seed=args.seed,
                lexical_csv=args.lexical_csv,
                embed_csv=args.embed_csv,
                audio_csv=args.audio_csv,
                sbert_entropy_csv=args.sbert_entropy_csv,
                near_dx_years=args.near_dx_years,
                far_positive_weight=args.far_positive_weight,
            )

            # choose thresholds per head
            thresholds: dict[str, float] = {}
            if args.threshold_policy == "fixed_0.5":
                for head in test_probs.keys():
                    thresholds[head] = 0.5
            else:
                for head, p_oof in oof_probs.items():
                    thresholds[head] = select_threshold_for_target_sensitivity(
                        y_true=y_tr,
                        p=p_oof,
                        target_sens=args.target_sensitivity,
                    )
                    p_oof_arr = np.asarray(p_oof, dtype=float)
                    oof_pos = (y_tr == 1) & np.isfinite(p_oof_arr)
                    if np.any(oof_pos):
                        oof_sens = float((p_oof_arr[oof_pos] >= thresholds[head]).mean())
                    else:
                        oof_sens = np.nan
                    q0, q50, q100 = np.nanquantile(p_oof_arr, [0.0, 0.5, 1.0])
                    print(
                        f"[thr-check] fold={fold} exp={exp_name} head={head} "
                        f"thr={thresholds[head]:.4f} "
                        f"oof_sens={oof_sens:.3f} target={args.target_sensitivity:.3f} "
                        f"oof_q(min/med/max)=({q0:.4f}/{q50:.4f}/{q100:.4f})",
                        flush=True,
                    )

            # fold metrics
            y_te = test_df["label"].to_numpy(dtype=int)
            for head, p in test_probs.items():
                thr = thresholds.get(head, 0.5)
                for eval_level, y_eval, p_eval in (
                    ("clip", y_te, np.asarray(p, dtype=float)),
                    ("speaker",) + aggregate_to_speaker_level(test_df, p),
                ):
                    mr = compute_metrics(y_true=y_eval, p=p_eval, thr=thr)
                    results_rows.append(
                        {
                            "experiment_id": exp_name,
                            "fold": fold,
                            "head": head,
                            "eval_level": eval_level,
                            "thr": thr,
                            "auc": mr.auc,
                            "f1": mr.f1,
                            "sens": mr.sens,
                            "spec": mr.spec,
                            "brier": mr.brier,
                            "n_pos": mr.n_pos,
                            "n_neg": mr.n_neg,
                            "n_test": len(y_eval),
                            "threshold_policy": args.threshold_policy,
                            "target_sensitivity": args.target_sensitivity if args.threshold_policy == "target_sens" else np.nan,
                        }
                    )

            # row-level predictions (one row per clip, wide p_txt/p_aud/p_fus)
            if not args.no_predictions:
                p_txt = test_probs.get("txt", np.full(len(test_df), np.nan))
                p_aud = test_probs.get("aud", np.full(len(test_df), np.nan))
                p_fus = test_probs.get("fus", test_probs.get("main", np.full(len(test_df), np.nan)))

                for i in range(len(test_df)):
                    pred_rows.append(
                        {
                            "experiment_id": exp_name,
                            "fold": fold,
                            "clip_id": test_df.loc[i, "clip_id"],
                            "speaker": test_df.loc[i, "speaker"],
                            "label": int(test_df.loc[i, "label"]),
                            "years_pre_dx": test_df.loc[i, "years_pre_dx"],
                            "p_txt": float(p_txt[i]) if np.isfinite(p_txt[i]) else np.nan,
                            "p_aud": float(p_aud[i]) if np.isfinite(p_aud[i]) else np.nan,
                            "p_fus": float(p_fus[i]) if np.isfinite(p_fus[i]) else np.nan,
                            "modality": "clip",
                        }
                    )
                pred_rows.extend(
                    aggregate_prediction_rows(
                        test_df,
                        exp_name,
                        fold,
                        p_txt,
                        p_aud,
                        p_fus,
                    )
                )

    results_path = write_results(args.out_dir, results_rows)
    print(f"Wrote {results_path}")

    if not args.no_predictions:
        preds_path = write_predictions(args.out_dir, pred_rows)
        print(f"Wrote {preds_path}")

    results_path = args.out_dir + "/results.csv"
    summarize_results_csv(results_path, print_table=True, save_grid_csv=True)


if __name__ == "__main__":
    main()





