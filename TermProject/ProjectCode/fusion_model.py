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
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

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

# ---------------- Speaker aggregation ------------- #
num_cols = [c for c in df.columns if c not in ["clip_id","speaker","label","years_pre_dx"]]
agg_df = (
    df.groupby("speaker")
      .agg({**{c:"mean" for c in num_cols}, "label":"first", "years_pre_dx":"min"})
      .reset_index()
)
print(f"Speaker-level rows: {agg_df.shape[0]}")

text_cols  = [c for c in text_df.columns if c != "clip_id"]
audio_cols = [c for c in audio_df.columns if c not in ["clip_id","voiced_sec","wpm"]]

X_txt  = agg_df[text_cols].values
X_aud  = agg_df[audio_cols].values
y      = agg_df.label.values
groups = agg_df.speaker.values
weights = np.exp(-np.nan_to_num(agg_df.years_pre_dx, nan=0.0) / TAU_YEARS)

# ---------------- CV + Metrics -------------------- #
cv = GroupKFold(n_splits=K_FOLDS)
auc_t, auc_a, auc_f = [], [], []
raw = {m: dict(tp=0, tn=0, fp=0, fn=0) for m in ["Text", "Audio", "Fusion"]}
all_y, all_p_txt, all_p_aud, all_p_fus = [], [], [], []

for tr, te in cv.split(X_txt, y, groups):
    # --- text head (CatBoost) ---
    txt_pos_w = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    text_clf = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        class_weights=[1.0, txt_pos_w],
        verbose=False,
    )
    text_clf.fit(X_txt[tr], y[tr], sample_weight=weights[tr])
    p_txt = text_clf.predict_proba(X_txt[te])[:, 1]
    y_txt = p_txt >= 0.5

    # --- audio head (RandomForest) ---
    aud_clf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )
    aud_clf.fit(X_aud[tr], y[tr], sample_weight=weights[tr])
    p_aud = aud_clf.predict_proba(X_aud[te])[:, 1]
    y_aud = p_aud >= 0.5

    # --- fusion ---
    Z_tr = np.column_stack([
        text_clf.predict_proba(X_txt[tr])[:, 1],
        aud_clf.predict_proba(X_aud[tr])[:, 1],
    ])
    Z_te = np.column_stack([p_txt, p_aud])

    blender = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=RANDOM_STATE)
    blender.fit(Z_tr, y[tr], sample_weight=weights[tr])
    p_fus = blender.predict_proba(Z_te)[:, 1]
    y_fus = p_fus >= 0.15

    # --- metrics ---
    auc_t.append(roc_auc_score(y[te], p_txt))
    auc_a.append(roc_auc_score(y[te], p_aud))
    auc_f.append(roc_auc_score(y[te], p_fus))

    # raw counts
    for mod, y_pred in zip(["Text", "Audio", "Fusion"], [y_txt, y_aud, y_fus]):
        raw[mod]["tp"] += int(((y_pred == 1) & (y[te] == 1)).sum())
        raw[mod]["tn"] += int(((y_pred == 0) & (y[te] == 0)).sum())
        raw[mod]["fp"] += int(((y_pred == 1) & (y[te] == 0)).sum())
        raw[mod]["fn"] += int(((y_pred == 0) & (y[te] == 1)).sum())

    all_y.append(y[te])
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
