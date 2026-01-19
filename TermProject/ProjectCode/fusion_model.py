#!/usr/bin/env python
"""
fusion_model.py — speaker-level CV with CatBoost audio head
==========================================================
Implements:
* clip filters, speaker aggregation, time-to-Dx weights
* Text head  = CatBoostClassifier (replacing original LogisticRegression)
* Audio head = RandomForestClassifier (optionally CatBoost)
* Fusion     = MLP (2‑prob inputs)
* Adds: **feature-importance tables** for the 10 most predictive lexical and audio features
* Raw counts + ROC / calibration plots
"""

import re
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# ---------------- Definitions ---------------- # 
def make_text_model(pos_weight, seed):
    return CatBoostClassifier(
        iterations=700, depth=6, learning_rate=0.05,
        loss_function="Logloss", random_seed=seed,
        class_weights=[1.0, pos_weight], verbose=False
    )

def make_audio_model(seed):
    return RandomForestClassifier(
        n_estimators=500, n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=seed
    )

# ---------------- Hyper‑parameters ---------------- #
K_FOLDS      = 10
RANDOM_STATE = 50
TAU_YEARS    = 5.0     # decay for sample weights
MIN_SECONDS  = 5.0     # voiced speech seconds filter
MIN_WPM      = 40.0    # speaking‑rate filter
TOPN_FEATS   = 20      # number of top features to display per modality
# -------------------------------------------------- #

ROOT      = Path(__file__).resolve().parents[0]
TEXT_CSV  = ROOT / "text_features_lexical.csv"
AUDIO_CSV = ROOT / "audio_features.csv"

# ---------------- Data ingest --------------------- #
text_df  = pd.read_csv(TEXT_CSV)
audio_df = pd.read_csv(AUDIO_CSV)

sbert_cols = [c for c in text_df.columns if c.startswith("sbert_")]
text_cols  = [
    "sem_drift","lex_entropy","ttr","mattr",
    "mean_sentence_len","pronoun_ratio",
] + sbert_cols

# ---------------- Merge + filter ------------------ #
def parse_clip(path: str):
    parts = PurePath(path).parts
    label   = 1 if parts[1] == "dementia" else 0
    speaker = parts[2]
    years   = np.nan
    if label:
        m = re.match(r".+_(\d+)(?:_\d+)?$", PurePath(path).stem)
        if m:
            years = int(m.group(1))
    return label, speaker, years

meta_rows = [parse_clip(p) for p in text_df.clip_id]
meta = pd.DataFrame(meta_rows, columns=["label","speaker","years_pre_dx"])
meta["clip_id"] = text_df.clip_id

df = meta.merge(audio_df, on="clip_id").merge(text_df, on="clip_id")
print(f"Initial merged dataset: {df.shape[0]} clips")

qual_mask = (df.voiced_sec >= MIN_SECONDS) & (df.wpm >= MIN_WPM)
df = df[qual_mask].reset_index(drop=True)
print(f"After quality filter: {df.shape[0]} clips")

# ------------------ Clip level rows --------------- #
text_cols = [c for c in text_df.columns if c != "clip_id" and c in df.columns]
audio_cols = [c for c in audio_df.columns if c not in ["clip_id", "voiced_sec", "wpm"] and c in df.columns]

# Optional sanity check: ensure we have clip-level rows
print(f"Clip-level rows: {df.shape[0]}")
print(f"Unique speakers: {df['speaker'].nunique()}")

# change for clip level audio (change from speaker level aggregation)
X_txt = df[text_cols].to_numpy()
X_aud = df[audio_cols].to_numpy()
y = df["label"].to_numpy().astype(int)

# ensuring that groups must align with clip rows (speaker id per clip)
groups = df["speaker"].to_numpy()

# weights: exp(-years/TAU); controls should be ~1.0
# If years_pre_dx is NaN (controls), set weight = 1.0
years = df["years_pre_dx"].to_numpy()
years_safe = np.where(np.isfinite(years), years, 0.0)
weights = np.exp(-years_safe / TAU_YEARS).astype(float)

# ---------------- CV + Metrics -------------------- #

# Outer CV: group-aware by speaker (modification for clip level rows)
n_groups = len(np.unique(groups))
outer_splits = min(K_FOLDS, n_groups)
if outer_splits < 2:
    raise ValueError(f"Not enough unique speakers for CV (found {n_groups}).")

outer_cv = GroupKFold(n_splits=outer_splits)

auc_t, auc_a, auc_f = [], [], []
raw = {m: dict(tp=0, tn=0, fp=0, fn=0) for m in ["Text", "Audio", "Fusion"]}
all_y, all_p_txt, all_p_aud, all_p_fus = [], [], [], []

for fold, (tr, te) in enumerate(outer_cv.split(X_txt, y, groups=groups), 1):

    # outer-train data
    Xtxt_tr, Xaud_tr = X_txt[tr], X_aud[tr]
    y_tr = y[tr]
    w_tr = weights[tr]
    g_tr = groups[tr]

    # outer-test data
    Xtxt_te, Xaud_te = X_txt[te], X_aud[te]
    y_te = y[te]

    # --- build OOF preds on outer-train, using group-safe inner CV ---
    n_groups_tr = len(np.unique(g_tr))
    inner_splits = min(5, n_groups_tr)
    if inner_splits < 2:
        raise ValueError(
            f"Not enough unique speakers in outer-train for inner CV "
            f"(found {n_groups_tr} in fold {fold})."
        )


    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE
    )
    inner_splitter = inner_cv.split(Xtxt_tr, y_tr, groups=g_tr)

    oof_txt = np.zeros(len(tr), dtype=float)
    oof_aud = np.zeros(len(tr), dtype=float)

    for itrn, ival in inner_splitter:
        # ensuring that inner train has positives
        pos = int((y_tr[itrn] == 1).sum())
        neg = int((y_tr[itrn] == 0).sum())
        txt_pos_w_in = neg / max(1, pos)

        txt_model = make_text_model(pos_weight=txt_pos_w_in, seed=RANDOM_STATE)
        aud_model = make_audio_model(seed=RANDOM_STATE)

        txt_model.fit(Xtxt_tr[itrn], y_tr[itrn], sample_weight=w_tr[itrn])
        aud_model.fit(Xaud_tr[itrn], y_tr[itrn], sample_weight=w_tr[itrn])

        oof_txt[ival] = txt_model.predict_proba(Xtxt_tr[ival])[:, 1]
        oof_aud[ival] = aud_model.predict_proba(Xaud_tr[ival])[:, 1]

    Z_oof = np.column_stack([oof_txt, oof_aud])

    # --- fit blender on OOF preds (outer-train only) ---
    blender = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )
    blender.fit(Z_oof, y_tr, sample_weight=w_tr)

    # --- refit base models on full outer-train for final outer-test predictions ---
    pos_full = int((y_tr == 1).sum())
    neg_full = int((y_tr == 0).sum())
    txt_pos_w = neg_full / max(1, pos_full)

    text_clf = make_text_model(pos_weight=txt_pos_w, seed=RANDOM_STATE)
    aud_clf  = make_audio_model(seed=RANDOM_STATE)

    text_clf.fit(Xtxt_tr, y_tr, sample_weight=w_tr)
    aud_clf.fit(Xaud_tr, y_tr, sample_weight=w_tr)

    # --- predict outer-test ---
    p_txt = text_clf.predict_proba(Xtxt_te)[:, 1]
    p_aud = aud_clf.predict_proba(Xaud_te)[:, 1]
    Z_te  = np.column_stack([p_txt, p_aud])

    p_fus = blender.predict_proba(Z_te)[:, 1]

    # --- metrics (guard for single-class test fold) ---
    if len(np.unique(y_te)) < 2:
        # AUC undefined if only one class in y_te
        auc_t.append(np.nan)
        auc_a.append(np.nan)
        auc_f.append(np.nan)
        print(f"[fold {fold}] Warning: y_te has one class; skipping AUC for this fold.")
    else:
        auc_t.append(roc_auc_score(y_te, p_txt))
        auc_a.append(roc_auc_score(y_te, p_aud))
        auc_f.append(roc_auc_score(y_te, p_fus))

    # raw counts with your thresholds
    pred_map = {
        "Text":  (p_txt, 0.5),
        "Audio": (p_aud, 0.5),
        "Fusion": (p_fus, 0.15),
    }

    for mod, (p, thr) in pred_map.items():
        y_pred = (p >= thr)
        raw[mod]["tp"] += int(((y_pred == 1) & (y_te == 1)).sum())
        raw[mod]["tn"] += int(((y_pred == 0) & (y_te == 0)).sum())
        raw[mod]["fp"] += int(((y_pred == 1) & (y_te == 0)).sum())
        raw[mod]["fn"] += int(((y_pred == 0) & (y_te == 1)).sum())

    all_y.append(y_te)
    all_p_txt.append(p_txt)
    all_p_aud.append(p_aud)
    all_p_fus.append(p_fus)

# ---------------- Summary ------------------------- #
print("\n===== 10-fold CV (speaker-level) =====")
print(f"Text-only  : AUC {np.mean(auc_t):.3f} ± {np.std(auc_t):.3f}")
print(f"Audio-only : AUC {np.mean(auc_a):.3f} ± {np.std(auc_a):.3f}")
print(f"Fusion MLP : AUC {np.mean(auc_f):.3f} ± {np.std(auc_f):.3f}")

rows = [
    [m, d["tp"] + d["fn"], d["tn"] + d["fp"], d["tp"] + d["tn"], d["fp"] + d["fn"]]
    for m, d in raw.items()
]
print("\n===== Raw prediction counts (pooled folds) =====")
print(pd.DataFrame(rows, columns=["Modality", "# Dementia", "# Control", "Correct", "Incorrect"]).to_string(index=False))

# ----- F1 from pooled counts (threshold = 0.5) -----
f1_rows = []
for m, d in raw.items():
    tp, tn, fp, fn = d["tp"], d["tn"], d["fp"], d["fn"]
    denom_pos = 2*tp + fp + fn
    f1_pos = 0.0 if denom_pos == 0 else (2*tp) / denom_pos
    # (optional) F1 for the negative class:
    denom_neg = 2*tn + fn + fp
    f1_neg = 0.0 if denom_neg == 0 else (2*tn) / denom_neg
    f1_rows.append([m, f1_pos, f1_neg])

print("\n===== F1 (from pooled counts) =====")
print(pd.DataFrame(f1_rows, columns=["Modality", "F1(+)", "F1(-)"]).round(3).to_string(index=False))

# ---------------- Feature Importance -------------- #
print("\n===== Top predictive features =====")

full_txt_pos_w = (y == 0).sum() / (y == 1).sum()
final_text_clf = CatBoostClassifier(
    iterations=700,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=RANDOM_STATE,
    class_weights=[1.0, full_txt_pos_w],
    verbose=False,
)
final_text_clf.fit(X_txt, y, sample_weight=weights)

final_audio_clf = RandomForestClassifier(
    n_estimators=500,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
)
final_audio_clf.fit(X_aud, y, sample_weight=weights)

def show_top_features(model, feature_names, title, topn=20):
    if hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance(type="FeatureImportance")
    else:
        importances = model.feature_importances_
    top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(topn)
    print(f"\nTop {topn} features — {title}")
    print(top.round(4).to_string())

show_top_features(final_text_clf, text_cols, "Lexical (CatBoost)", TOPN_FEATS)
show_top_features(final_audio_clf, audio_cols, "Acoustic (RandomForest)", TOPN_FEATS)


# ---------------- ROC + Calibration -------------- #
all_y      = np.concatenate(all_y)
all_p_txt  = np.concatenate(all_p_txt)
all_p_aud  = np.concatenate(all_p_aud)
all_p_fus  = np.concatenate(all_p_fus)

plt.figure()
for p,label,col in [(all_p_txt,"Text","#003057"),(all_p_aud,"Audio","#B3A369"),(all_p_fus,"Fusion","#A0A0A0")]:
    fpr,tpr,_=roc_curve(all_y,p); plt.plot(fpr,tpr,label=label,color=col)
plt.plot([0,1],[0,1],'--',color='black')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Speaker‑level)"); plt.legend(); plt.tight_layout(); plt.savefig("ROCplot.png")

prob_true, prob_pred = calibration_curve(all_y, all_p_fus, n_bins=10, strategy='quantile')
plt.figure()
plt.plot(prob_pred, prob_true,'o',color="#A0A0A0"); plt.plot([0,1],[0,1],'--',color='black')
plt.xlabel("Mean Predicted Prob"); plt.ylabel("Fraction Positive"); plt.title("Calibration (Fusion)")
ax2 = plt.gca().twinx(); ax2.hist(all_p_fus, bins=15, alpha=0.3, color="#A0A0A0"); ax2.set_ylabel("Count")
plt.tight_layout(); plt.savefig("calibration.png")
