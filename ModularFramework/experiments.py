'''
Model variations and initializations. All experiments on varying models and params to be housed here.
'''

from __future__ import annotations

from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from utils import get_core_audio_cols, fit_audio_pca_transform



'''
Testing of different classifiers
* LinearSVC + calibration
* RandomForest
* CatBoost
* XGBoost
* MLP

Feature Families
* Lexical-only
* Entropy-only
* eGeMAPS-only
* Audio + pause + WPM


Fusion Variants
* Late stage
* Blender
'''

# utils
def make_lr(seed):
    # increase max_iter for convergence
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="l2",
            max_iter=5000,
            random_state=seed,
        )),
    ])

def make_svm_cal(seed):
    # LinearSVC calibrate it to get probs
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(random_state=seed)),
    ])
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)

def make_mlp(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=seed,
        )),
    ])


def build_X(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[cols].to_numpy(dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


# C grid tuning
LR_C_GRID = [0.01, 0.1, 1.0, 10.0]

def lr_factory(seed: int, C: float = 1.0):
    # bare lin reg w/o pipeline to hand-craft exp func
    return LogisticRegression(solver="liblinear", C=C, random_state=seed, max_iter=2000, class_weight="balanced",)


def select_C_by_inner_cv(
    X_tr: np.ndarray, y_tr: np.ndarray, inner_splits_idx: list, seed: int, c_grid: list = LR_C_GRID, sample_weights: np.ndarray = None):
    
    # Grid search over C using pre-comp inner CV splits, select C to max AUC on inner folds, and return best C

    best_C, best_auc = c_grid[0], -1.0
    for C in c_grid:
        fold_aucs = []
        for tr_idx, va_idx in inner_splits_idx:
            m = lr_factory(seed, C=C)
            w = sample_weights[tr_idx] if sample_weights is not None else None
            m.fit(X_tr[tr_idx], y_tr[tr_idx], sample_weight=w)
            p_va = m.predict_proba(X_tr[va_idx])[:, 1]
            if len(np.unique(y_tr[va_idx])) < 2:
                continue          # skip folds with only one class
            fold_aucs.append(roc_auc_score(y_tr[va_idx], p_va))
        if fold_aucs:
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_auc:
                best_auc, best_C = mean_auc, C
    return float(best_C)

def infer_numeric_cols(df, exclude=("clip_id",)):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    # enforce deterministic order across runs
    cols = sorted(cols)
    return cols

def build_X_from_loaded(df, cols):
    # always returns float matrix
    return df[cols].to_numpy(dtype=float)


def _embedding_cols_from_df(df: pd.DataFrame, exclude: Optional[set[str]] = None) -> List[str]:
    exclude = exclude or set()
    cols = []
    for c in infer_numeric_cols(df):
        if c in exclude:
            continue
        if c.startswith("emb_") or c.startswith("sbert_"):
            cols.append(c)
    return cols


def _fit_text_embedding_reducer(
    lex_train: pd.DataFrame,
    lex_test: pd.DataFrame,
    emb_train: pd.DataFrame,
    emb_test: pd.DataFrame,
    *,
    text_cols: List[str],
    embed_cols: List[str],
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_txt_tr = lex_train[text_cols].to_numpy(dtype=float)
    X_txt_te = lex_test[text_cols].to_numpy(dtype=float)

    embed_imputer = SimpleImputer(strategy="median")
    X_emb_tr = embed_imputer.fit_transform(emb_train[embed_cols].to_numpy(dtype=float))
    X_emb_te = embed_imputer.transform(emb_test[embed_cols].to_numpy(dtype=float))

    embed_scaler = StandardScaler()
    X_emb_tr = embed_scaler.fit_transform(X_emb_tr)
    X_emb_te = embed_scaler.transform(X_emb_te)

    max_comp = min(X_emb_tr.shape[0], X_emb_tr.shape[1])
    k = max(2, min(int(n_components), max_comp))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    Z_tr = pca.fit_transform(X_emb_tr)
    Z_te = pca.transform(X_emb_te)

    return np.concatenate([X_txt_tr, Z_tr], axis=1), np.concatenate([X_txt_te, Z_te], axis=1)


# TODO: reconfigure for df
DEFAULT_TEXT_COLS = [
    "mattr", "ttr", "mean_sentence_len", "pronoun_ratio", "lex_entropy", "sem_drift"
]
# removed due to too many audio cols
# DEFAULT_AUDIO_COLS = [
#     "egemaps_f0_mean", "egemaps_jitter", "egemaps_shimmer", "pause_pct", "wpm"
# ] 

# experiment functions
def infer_audio_cols(audio_csv: str) -> List[str]:
    df0 = pd.read_csv(audio_csv, nrows=1)
    # Keep existing clip_id; only fall back to first column when absent.
    if "clip_id" not in df0.columns:
        df0 = df0.rename(columns={df0.columns[0]: "clip_id"})
    cols = [c for c in df0.columns if c != "clip_id"]
    return cols

def load_features_for_split(features_csv: str, clip_ids: pd.Series) -> pd.DataFrame:
    feats = pd.read_csv(features_csv)

    # Use clip_id if it already exists (regardless of position)
    if "clip_id" not in feats.columns:
        feats = feats.rename(columns={feats.columns[0]: "clip_id"})

    # Normalize paths (slashes + whitespace)
    feats["clip_id"] = (
        feats["clip_id"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    )
    req = clip_ids.astype(str).str.strip().str.replace("\\", "/", regex=False)

    # Index and align
    feats = feats.set_index("clip_id")

    missing = set(req.values) - set(feats.index.values)
    if missing:
        examples = list(missing)[:10]
        raise ValueError(
            f"Missing {len(missing)} clip_id(s) in {features_csv}. Examples: {examples}"
        )

    return feats.loc[req.values].reset_index()


def _aggregate_features_to_speaker(
    meta_df: pd.DataFrame,
    feats_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    joined = meta_df.loc[:, ["clip_id", "speaker", "label", "years_pre_dx"]].merge(
        feats_df.loc[:, ["clip_id"] + feature_cols],
        on="clip_id",
        how="left",
        validate="one_to_one",
    )

    meta = (
        joined.groupby("speaker", dropna=False)
        .agg(
            label=("label", "max"),
            years_pre_dx=("years_pre_dx", "min"),
            clip_count=("clip_id", "size"),
        )
    )
    feat_mean = joined.groupby("speaker", dropna=False)[feature_cols].mean().add_suffix("__mean")
    feat_std = joined.groupby("speaker", dropna=False)[feature_cols].std().fillna(0.0).add_suffix("__std")

    return meta.join(feat_mean).join(feat_std).reset_index()


def _speaker_feature_cols(df: pd.DataFrame) -> List[str]:
    return infer_numeric_cols(df, exclude=("label", "years_pre_dx"))


def _map_speaker_probs_to_clips(df: pd.DataFrame, speaker_df: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    pred_map = pd.DataFrame({
        "speaker": speaker_df["speaker"].astype(str).to_numpy(),
        "p": np.asarray(probs, dtype=float),
    })
    mapped = (
        df.loc[:, ["speaker"]]
        .assign(speaker=lambda x: x["speaker"].astype(str))
        .merge(pred_map, on="speaker", how="left", validate="many_to_one")
    )
    return mapped["p"].to_numpy(dtype=float)


def _make_timeaware_sample_weights(
    df: pd.DataFrame,
    *,
    near_dx_years: float,
    far_positive_weight: float,
) -> np.ndarray:
    y = df["label"].to_numpy(dtype=int)
    years = pd.to_numeric(df["years_pre_dx"], errors="coerce").to_numpy(dtype=float)
    weights = np.ones(len(df), dtype=float)
    far_mask = (y == 1) & np.isfinite(years) & (years > float(near_dx_years))
    weights[far_mask] = float(far_positive_weight)
    return weights


def run_text_lr(train_df, test_df, inner_splits_idx, seed, lexical_csv=None, feature_cols=None, **kwargs):
    cols = feature_cols or DEFAULT_TEXT_COLS

    train_feats = load_features_for_split(lexical_csv, train_df["clip_id"])
    test_feats  = load_features_for_split(lexical_csv, test_df["clip_id"])

    X_tr = train_feats[cols].to_numpy(dtype=float)
    X_te = test_feats[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)

    # fit full model
    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    # OOF for threshold selection (inner CV)
    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"txt": p_test}, {"txt": p_oof}


def run_text_lexical_embed_lr(
    train_df,
    test_df,
    inner_splits_idx,
    seed,
    lexical_csv=None,
    sbert_entropy_csv=None,
    feature_cols=None,
    n_components=16,
    **kwargs,
):
    if not lexical_csv:
        raise ValueError("lexical_csv is required for text_lexical_embed_lr")

    emb_csv = sbert_entropy_csv or lexical_csv

    lex_tr = load_features_for_split(lexical_csv, train_df["clip_id"])
    lex_te = load_features_for_split(lexical_csv, test_df["clip_id"])
    emb_tr = load_features_for_split(emb_csv, train_df["clip_id"])
    emb_te = load_features_for_split(emb_csv, test_df["clip_id"])

    text_cols = feature_cols or DEFAULT_TEXT_COLS
    missing_text = [c for c in text_cols if c not in lex_tr.columns]
    if missing_text:
        raise ValueError(f"Text feature columns not found in {lexical_csv}: {missing_text}")

    embed_cols = _embedding_cols_from_df(emb_tr, exclude=set(text_cols))
    if not embed_cols:
        raise ValueError(
            "No embedding columns found for text_lexical_embed_lr. "
            "Expected columns starting with 'emb_' or 'sbert_'."
        )

    X_tr, X_te = _fit_text_embedding_reducer(
        lex_tr,
        lex_te,
        emb_tr,
        emb_te,
        text_cols=text_cols,
        embed_cols=embed_cols,
        n_components=n_components,
        seed=seed,
    )
    y_tr = train_df["label"].to_numpy(dtype=int)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        X_fold_tr, X_fold_va = _fit_text_embedding_reducer(
            lex_tr.iloc[tr_idx].reset_index(drop=True),
            lex_tr.iloc[va_idx].reset_index(drop=True),
            emb_tr.iloc[tr_idx].reset_index(drop=True),
            emb_tr.iloc[va_idx].reset_index(drop=True),
            text_cols=text_cols,
            embed_cols=embed_cols,
            n_components=n_components,
            seed=seed,
        )
        m = make_lr(seed)
        m.fit(X_fold_tr, y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_fold_va)[:, 1]

    return {"txt": p_test}, {"txt": p_oof}


def run_audio_lr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inner_splits_idx: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    feature_cols: Optional[List[str]] = None,
    *,
    audio_csv: str,
    **kwargs,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cols = feature_cols or infer_audio_cols(audio_csv)

    train_feats = load_features_for_split(audio_csv, train_df["clip_id"])
    test_feats = load_features_for_split(audio_csv, test_df["clip_id"])

    missing_cols = [c for c in cols if c not in train_feats.columns]
    if missing_cols:
        raise ValueError(f"Audio feature columns not found in {audio_csv}: {missing_cols}")

    X_tr = train_feats[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = test_feats[cols].to_numpy(dtype=float)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    # OOF for threshold selection (inner CV on already-grouped indices)
    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"aud": p_test}, {"aud": p_oof}


def run_text_lr_speaker(train_df, test_df, inner_splits_idx, seed, lexical_csv=None, feature_cols=None, **kwargs):
    cols = feature_cols or DEFAULT_TEXT_COLS

    train_feats = load_features_for_split(lexical_csv, train_df["clip_id"])
    test_feats = load_features_for_split(lexical_csv, test_df["clip_id"])

    speaker_tr = _aggregate_features_to_speaker(train_df, train_feats, cols)
    speaker_te = _aggregate_features_to_speaker(test_df, test_feats, cols)
    sp_cols = _speaker_feature_cols(speaker_tr)

    X_tr = speaker_tr[sp_cols].to_numpy(dtype=float)
    y_tr = speaker_tr["label"].to_numpy(dtype=int)
    X_te = speaker_te[sp_cols].to_numpy(dtype=float)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = _map_speaker_probs_to_clips(test_df, speaker_te, model.predict_proba(X_te)[:, 1])

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[va_idx].reset_index(drop=True)
        fold_train_feats = train_feats.iloc[tr_idx].reset_index(drop=True)
        fold_val_feats = train_feats.iloc[va_idx].reset_index(drop=True)

        fold_speaker_tr = _aggregate_features_to_speaker(fold_train_df, fold_train_feats, cols)
        fold_speaker_va = _aggregate_features_to_speaker(fold_val_df, fold_val_feats, cols)
        fold_cols = _speaker_feature_cols(fold_speaker_tr)

        m = make_lr(seed)
        m.fit(
            fold_speaker_tr[fold_cols].to_numpy(dtype=float),
            fold_speaker_tr["label"].to_numpy(dtype=int),
        )
        p_va_speaker = m.predict_proba(fold_speaker_va[fold_cols].to_numpy(dtype=float))[:, 1]
        p_oof[va_idx] = _map_speaker_probs_to_clips(fold_val_df, fold_speaker_va, p_va_speaker)

    return {"txt": p_test}, {"txt": p_oof}


def run_audio_lr_speaker(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inner_splits_idx: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    feature_cols: Optional[List[str]] = None,
    *,
    audio_csv: str,
    **kwargs,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cols = feature_cols or infer_audio_cols(audio_csv)

    train_feats = load_features_for_split(audio_csv, train_df["clip_id"])
    test_feats = load_features_for_split(audio_csv, test_df["clip_id"])

    speaker_tr = _aggregate_features_to_speaker(train_df, train_feats, cols)
    speaker_te = _aggregate_features_to_speaker(test_df, test_feats, cols)
    sp_cols = _speaker_feature_cols(speaker_tr)

    model = make_lr(seed)
    model.fit(
        speaker_tr[sp_cols].to_numpy(dtype=float),
        speaker_tr["label"].to_numpy(dtype=int),
    )
    p_test = _map_speaker_probs_to_clips(
        test_df,
        speaker_te,
        model.predict_proba(speaker_te[sp_cols].to_numpy(dtype=float))[:, 1],
    )

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[va_idx].reset_index(drop=True)
        fold_train_feats = train_feats.iloc[tr_idx].reset_index(drop=True)
        fold_val_feats = train_feats.iloc[va_idx].reset_index(drop=True)

        fold_speaker_tr = _aggregate_features_to_speaker(fold_train_df, fold_train_feats, cols)
        fold_speaker_va = _aggregate_features_to_speaker(fold_val_df, fold_val_feats, cols)
        fold_cols = _speaker_feature_cols(fold_speaker_tr)

        m = make_lr(seed)
        m.fit(
            fold_speaker_tr[fold_cols].to_numpy(dtype=float),
            fold_speaker_tr["label"].to_numpy(dtype=int),
        )
        p_va_speaker = m.predict_proba(fold_speaker_va[fold_cols].to_numpy(dtype=float))[:, 1]
        p_oof[va_idx] = _map_speaker_probs_to_clips(fold_val_df, fold_speaker_va, p_va_speaker)

    return {"aud": p_test}, {"aud": p_oof}


def run_text_lr_timeaware(
    train_df,
    test_df,
    inner_splits_idx,
    seed,
    lexical_csv=None,
    feature_cols=None,
    near_dx_years=5.0,
    far_positive_weight=0.5,
    **kwargs,
):
    cols = feature_cols or DEFAULT_TEXT_COLS

    train_feats = load_features_for_split(lexical_csv, train_df["clip_id"])
    test_feats = load_features_for_split(lexical_csv, test_df["clip_id"])

    X_tr = train_feats[cols].to_numpy(dtype=float)
    X_te = test_feats[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    w_tr = _make_timeaware_sample_weights(
        train_df,
        near_dx_years=near_dx_years,
        far_positive_weight=far_positive_weight,
    )

    model = make_lr(seed)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx], clf__sample_weight=w_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"txt": p_test}, {"txt": p_oof}


def run_audio_lr_timeaware(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inner_splits_idx: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    feature_cols: Optional[List[str]] = None,
    *,
    audio_csv: str,
    near_dx_years=5.0,
    far_positive_weight=0.5,
    **kwargs,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cols = feature_cols or infer_audio_cols(audio_csv)

    train_feats = load_features_for_split(audio_csv, train_df["clip_id"])
    test_feats = load_features_for_split(audio_csv, test_df["clip_id"])

    X_tr = train_feats[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = test_feats[cols].to_numpy(dtype=float)
    w_tr = _make_timeaware_sample_weights(
        train_df,
        near_dx_years=near_dx_years,
        far_positive_weight=far_positive_weight,
    )

    model = make_lr(seed)
    model.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx], clf__sample_weight=w_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"aud": p_test}, {"aud": p_oof}

def run_late_fusion_lr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inner_splits_idx: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    text_cols: Optional[List[str]] = None,
    audio_cols: Optional[List[str]] = None,
    *,
    lexical_csv: str,
    audio_csv: str,
    **kwargs,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    # 1) Load features first (so we can infer columns safely)
    train_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    test_txt  = load_features_for_split(lexical_csv, test_df["clip_id"])

    train_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    test_aud  = load_features_for_split(audio_csv, test_df["clip_id"])

    # 2) Choose columns
    tcols = text_cols or DEFAULT_TEXT_COLS

    if audio_cols is None:
        acols = [
            c for c in train_aud.columns
            if c != "clip_id" and pd.api.types.is_numeric_dtype(train_aud[c])
        ]
    else:
        acols = audio_cols

    if len(acols) == 0:
        raise ValueError(f"Inferred 0 audio feature columns from {audio_csv}. Check dtypes / parsing.")

    # 3) Validate requested columns exist
    missing_t = [c for c in tcols if c not in train_txt.columns]
    missing_a = [c for c in acols if c not in train_aud.columns]
    if missing_t:
        raise ValueError(f"Text feature columns not found in {lexical_csv}: {missing_t}")
    if missing_a:
        raise ValueError(f"Audio feature columns not found in {audio_csv}: {missing_a}")

    # 4) Build matrices
    X_txt_tr = train_txt[tcols].to_numpy(dtype=float)
    X_aud_tr = train_aud[acols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)

    X_txt_te = test_txt[tcols].to_numpy(dtype=float)
    X_aud_te = test_aud[acols].to_numpy(dtype=float)

    # OOF head predictions for blender training / thresholding
    p_txt_oof = np.zeros(len(train_df), dtype=float)
    p_aud_oof = np.zeros(len(train_df), dtype=float)

    for tr_idx, va_idx in inner_splits_idx:
        mt = make_lr(seed)
        ma = make_lr(seed)
        mt.fit(X_txt_tr[tr_idx], y_tr[tr_idx])
        ma.fit(X_aud_tr[tr_idx], y_tr[tr_idx])

        p_txt_oof[va_idx] = mt.predict_proba(X_txt_tr[va_idx])[:, 1]
        p_aud_oof[va_idx] = ma.predict_proba(X_aud_tr[va_idx])[:, 1]

    Z_oof = np.column_stack([p_txt_oof, p_aud_oof])

    blender = make_lr(seed)
    blender.fit(Z_oof, y_tr)
    p_fus_oof = blender.predict_proba(Z_oof)[:, 1]

    # refit heads on full outer-train
    mt_full = make_lr(seed)
    ma_full = make_lr(seed)
    mt_full.fit(X_txt_tr, y_tr)
    ma_full.fit(X_aud_tr, y_tr)

    p_txt_te = mt_full.predict_proba(X_txt_te)[:, 1]
    p_aud_te = ma_full.predict_proba(X_aud_te)[:, 1]

    # fit blender on full-train head preds (optional, keeps consistent)
    p_txt_tr_full = mt_full.predict_proba(X_txt_tr)[:, 1]
    p_aud_tr_full = ma_full.predict_proba(X_aud_tr)[:, 1]
    Z_full = np.column_stack([p_txt_tr_full, p_aud_tr_full])

    blender_full = make_lr(seed)
    blender_full.fit(Z_full, y_tr)

    Z_te = np.column_stack([p_txt_te, p_aud_te])
    p_fus_te = blender_full.predict_proba(Z_te)[:, 1]

    test_probs = {"txt": p_txt_te, "aud": p_aud_te, "fus": p_fus_te}
    oof_probs = {"txt": p_txt_oof, "aud": p_aud_oof, "fus": p_fus_oof}
    return test_probs, oof_probs


def run_early_fusion_lr_speaker(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inner_splits_idx: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    text_cols: Optional[List[str]] = None,
    audio_cols: Optional[List[str]] = None,
    *,
    lexical_csv: str,
    audio_csv: str,
    **kwargs,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = text_cols or DEFAULT_TEXT_COLS
    acols = audio_cols or infer_numeric_cols(tr_aud)

    speaker_tr_txt = _aggregate_features_to_speaker(train_df, tr_txt, tcols)
    speaker_te_txt = _aggregate_features_to_speaker(test_df, te_txt, tcols)
    speaker_tr_aud = _aggregate_features_to_speaker(train_df, tr_aud, acols)
    speaker_te_aud = _aggregate_features_to_speaker(test_df, te_aud, acols)

    X_tr = np.concatenate(
        [
            speaker_tr_txt[_speaker_feature_cols(speaker_tr_txt)].to_numpy(dtype=float),
            speaker_tr_aud[_speaker_feature_cols(speaker_tr_aud)].to_numpy(dtype=float),
        ],
        axis=1,
    )
    y_tr = speaker_tr_txt["label"].to_numpy(dtype=int)
    X_te = np.concatenate(
        [
            speaker_te_txt[_speaker_feature_cols(speaker_te_txt)].to_numpy(dtype=float),
            speaker_te_aud[_speaker_feature_cols(speaker_te_aud)].to_numpy(dtype=float),
        ],
        axis=1,
    )

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = _map_speaker_probs_to_clips(test_df, speaker_te_txt, model.predict_proba(X_te)[:, 1])

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[va_idx].reset_index(drop=True)
        fold_tr_txt = tr_txt.iloc[tr_idx].reset_index(drop=True)
        fold_va_txt = tr_txt.iloc[va_idx].reset_index(drop=True)
        fold_tr_aud = tr_aud.iloc[tr_idx].reset_index(drop=True)
        fold_va_aud = tr_aud.iloc[va_idx].reset_index(drop=True)

        fold_sp_tr_txt = _aggregate_features_to_speaker(fold_train_df, fold_tr_txt, tcols)
        fold_sp_va_txt = _aggregate_features_to_speaker(fold_val_df, fold_va_txt, tcols)
        fold_sp_tr_aud = _aggregate_features_to_speaker(fold_train_df, fold_tr_aud, acols)
        fold_sp_va_aud = _aggregate_features_to_speaker(fold_val_df, fold_va_aud, acols)

        X_fold_tr = np.concatenate(
            [
                fold_sp_tr_txt[_speaker_feature_cols(fold_sp_tr_txt)].to_numpy(dtype=float),
                fold_sp_tr_aud[_speaker_feature_cols(fold_sp_tr_aud)].to_numpy(dtype=float),
            ],
            axis=1,
        )
        X_fold_va = np.concatenate(
            [
                fold_sp_va_txt[_speaker_feature_cols(fold_sp_va_txt)].to_numpy(dtype=float),
                fold_sp_va_aud[_speaker_feature_cols(fold_sp_va_aud)].to_numpy(dtype=float),
            ],
            axis=1,
        )
        m = make_lr(seed)
        m.fit(X_fold_tr, fold_sp_tr_txt["label"].to_numpy(dtype=int))
        p_va_speaker = m.predict_proba(X_fold_va)[:, 1]
        p_oof[va_idx] = _map_speaker_probs_to_clips(fold_val_df, fold_sp_va_txt, p_va_speaker)

    return {"fus": p_test}, {"fus": p_oof}


def run_early_fusion_lr_timeaware(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inner_splits_idx: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    text_cols: Optional[List[str]] = None,
    audio_cols: Optional[List[str]] = None,
    *,
    lexical_csv: str,
    audio_csv: str,
    near_dx_years=5.0,
    far_positive_weight=0.5,
    **kwargs,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = text_cols or DEFAULT_TEXT_COLS
    acols = audio_cols or infer_numeric_cols(tr_aud)
    X_tr = np.concatenate(
        [tr_txt[tcols].to_numpy(dtype=float), tr_aud[acols].to_numpy(dtype=float)],
        axis=1,
    )
    X_te = np.concatenate(
        [te_txt[tcols].to_numpy(dtype=float), te_aud[acols].to_numpy(dtype=float)],
        axis=1,
    )
    y_tr = train_df["label"].to_numpy(dtype=int)
    w_tr = _make_timeaware_sample_weights(
        train_df,
        near_dx_years=near_dx_years,
        far_positive_weight=far_positive_weight,
    )

    model = make_lr(seed)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(
            X_tr[tr_idx],
            y_tr[tr_idx],
            sample_weight=w_tr[tr_idx],
        )
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"fus": p_test}, {"fus": p_oof}

def _run_single_modality(
    train_df,
    test_df,
    inner_splits_idx,
    seed,
    *,
    features_csv,
    head_name,
    feature_cols=None,
    model_kind="lr",
):
    feats_tr = load_features_for_split(features_csv, train_df["clip_id"])
    feats_te = load_features_for_split(features_csv, test_df["clip_id"])

    cols = feature_cols or infer_numeric_cols(feats_tr)

    X_tr = build_X_from_loaded(feats_tr, cols)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = build_X_from_loaded(feats_te, cols)

    if model_kind == "lr":
        model = make_lr(seed)
    elif model_kind == "svm_cal":
        model = make_svm_cal(seed)
    elif model_kind == "mlp":
        model = make_mlp(seed)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    # fit full model
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    # OOF for threshold selection
    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        if model_kind == "lr":
            m = make_lr(seed)
        elif model_kind == "svm_cal":
            m = make_svm_cal(seed)
        elif model_kind == "mlp":
            m = make_mlp(seed)

        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {head_name: p_test}, {head_name: p_oof}

def run_text_svm_cal(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, lexical_csv, **kwargs):
    return _run_single_modality(
        train_df, test_df, inner_splits_idx, seed,
        features_csv=lexical_csv,
        head_name="txt",
        feature_cols=feature_cols or DEFAULT_TEXT_COLS,  # if you have this
        model_kind="svm_cal",
    )

def run_text_mlp(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, lexical_csv, **kwargs):
    return _run_single_modality(
        train_df, test_df, inner_splits_idx, seed,
        features_csv=lexical_csv,
        head_name="txt",
        feature_cols=feature_cols or DEFAULT_TEXT_COLS,
        model_kind="mlp",
    )

def run_audio_svm_cal(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, audio_csv, **kwargs):
    return _run_single_modality(
        train_df, test_df, inner_splits_idx, seed,
        features_csv=audio_csv,
        head_name="aud",
        feature_cols=feature_cols,  # let it infer by default
        model_kind="svm_cal",
    )

def run_audio_mlp(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, audio_csv, **kwargs):
    return _run_single_modality(
        train_df, test_df, inner_splits_idx, seed,
        features_csv=audio_csv,
        head_name="aud",
        feature_cols=feature_cols,
        model_kind="mlp",
    )

def _run_early_fusion(
    train_df,
    test_df,
    inner_splits_idx,
    seed,
    *,
    lexical_csv,
    audio_csv,
    text_cols=None,
    audio_cols=None,
    model_kind="lr",
):
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = text_cols or DEFAULT_TEXT_COLS
    acols = audio_cols or infer_numeric_cols(tr_aud)

    X_tr = np.concatenate(
        [build_X_from_loaded(tr_txt, tcols), build_X_from_loaded(tr_aud, acols)],
        axis=1,
    )
    y_tr = train_df["label"].to_numpy(dtype=int)

    X_te = np.concatenate(
        [build_X_from_loaded(te_txt, tcols), build_X_from_loaded(te_aud, acols)],
        axis=1,
    )

    if model_kind == "lr":
        model = make_lr(seed)
    elif model_kind == "svm_cal":
        model = make_svm_cal(seed)
    elif model_kind == "mlp":
        model = make_mlp(seed)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        if model_kind == "lr":
            m = make_lr(seed)
        elif model_kind == "svm_cal":
            m = make_svm_cal(seed)
        elif model_kind == "mlp":
            m = make_mlp(seed)

        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"fus": p_test}, {"fus": p_oof}

def run_early_fusion_lr(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    return _run_early_fusion(train_df, test_df, inner_splits_idx, seed,
                             lexical_csv=lexical_csv, audio_csv=audio_csv, model_kind="lr")

def run_early_fusion_svm_cal(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    return _run_early_fusion(train_df, test_df, inner_splits_idx, seed,
                             lexical_csv=lexical_csv, audio_csv=audio_csv, model_kind="svm_cal")

def run_early_fusion_mlp(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    return _run_early_fusion(train_df, test_df, inner_splits_idx, seed,
                             lexical_csv=lexical_csv, audio_csv=audio_csv, model_kind="mlp")

def make_catboost(
    seed,
    *,
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    auto_class_weights=None,
    random_strength=0.0,
    bagging_temperature=0.0,
):
    # CPU-only, deterministic-ish with random_seed; silent training.
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        auto_class_weights=auto_class_weights,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


CATBOOST_PARAM_CONFIGS: Dict[str, Dict[str, object]] = {
    "cfg1": {
        "depth": 4,
        "learning_rate": 0.05,
        "iterations": 3000,
        "l2_leaf_reg": 10,
        "auto_class_weights": "Balanced",
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
    "cfg2": {
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 5000,
        "l2_leaf_reg": 10,
        "auto_class_weights": "Balanced",
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
    "cfg3": {
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 5000,
        "l2_leaf_reg": 30,
        "auto_class_weights": "Balanced",
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
}


def fit_catboost_predict_proba(model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    # CatBoost returns probs for both classes; [:,1] is positive class
    return model.predict_proba(X_te)[:, 1]


def _run_single_modality_catboost(
    train_df,
    test_df,
    inner_splits_idx,
    seed,
    *,
    features_csv,
    head_name,
    feature_cols=None,
    cat_params=None,
):
    feats_tr = load_features_for_split(features_csv, train_df["clip_id"])
    feats_te = load_features_for_split(features_csv, test_df["clip_id"])

    if feature_cols is None:
        cols = infer_numeric_cols(feats_tr)  # sorted numeric, excludes clip_id
    else:
        cols = feature_cols

    # sanity
    missing = [c for c in cols if c not in feats_tr.columns]
    if missing:
        raise ValueError(f"Columns not found in {features_csv}: {missing}")

    X_tr = feats_tr[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = feats_te[cols].to_numpy(dtype=float)

    params = cat_params or {}
    model = make_catboost(seed, **params)
    p_test = fit_catboost_predict_proba(model, X_tr, y_tr, X_te)

    # OOF for thresholding (inner CV)
    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_catboost(seed, **params)
        p_oof[va_idx] = fit_catboost_predict_proba(
            m,
            X_tr[tr_idx],
            y_tr[tr_idx],
            X_tr[va_idx],
        )

    return {head_name: p_test}, {head_name: p_oof}


def _run_early_fusion_catboost(
    train_df,
    test_df,
    inner_splits_idx,
    seed,
    *,
    lexical_csv,
    audio_csv,
    text_cols=None,
    audio_cols=None,
    cat_params=None,
):
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])

    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = text_cols or DEFAULT_TEXT_COLS
    if audio_cols is None:
        acols = infer_numeric_cols(tr_aud)  # sorted numeric
    else:
        acols = audio_cols

    # check cols exist
    missing_t = [c for c in tcols if c not in tr_txt.columns]
    missing_a = [c for c in acols if c not in tr_aud.columns]
    if missing_t:
        raise ValueError(f"Text cols not found in {lexical_csv}: {missing_t}")
    if missing_a:
        raise ValueError(f"Audio cols not found in {audio_csv}: {missing_a}")

    X_tr = np.concatenate(
        [tr_txt[tcols].to_numpy(dtype=float), tr_aud[acols].to_numpy(dtype=float)],
        axis=1,
    )
    y_tr = train_df["label"].to_numpy(dtype=int)

    X_te = np.concatenate(
        [te_txt[tcols].to_numpy(dtype=float), te_aud[acols].to_numpy(dtype=float)],
        axis=1,
    )

    params = cat_params or {}
    model = make_catboost(seed, **params)
    p_test = fit_catboost_predict_proba(model, X_tr, y_tr, X_te)

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_catboost(seed, **params)
        p_oof[va_idx] = fit_catboost_predict_proba(
            m,
            X_tr[tr_idx],
            y_tr[tr_idx],
            X_tr[va_idx],
        )

    return {"fus": p_test}, {"fus": p_oof}

def run_text_catboost(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, lexical_csv, **kwargs):
    cols = feature_cols or DEFAULT_TEXT_COLS
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=lexical_csv,
        head_name="txt",
        feature_cols=cols,
        cat_params=None,
    )


def run_audio_catboost(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, audio_csv, **kwargs):
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=audio_csv,
        head_name="aud",
        feature_cols=feature_cols,  # None => infer numeric
        cat_params=None,
    )


def run_early_fusion_catboost(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    return _run_early_fusion_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        lexical_csv=lexical_csv,
        audio_csv=audio_csv,
        text_cols=DEFAULT_TEXT_COLS,
        audio_cols=None,
        cat_params=None,
    )


def run_text_catboost_cfg1(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, lexical_csv, **kwargs):
    cols = feature_cols or DEFAULT_TEXT_COLS
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=lexical_csv,
        head_name="txt",
        feature_cols=cols,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg1"],
    )


def run_text_catboost_cfg2(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, lexical_csv, **kwargs):
    cols = feature_cols or DEFAULT_TEXT_COLS
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=lexical_csv,
        head_name="txt",
        feature_cols=cols,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg2"],
    )


def run_text_catboost_cfg3(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, lexical_csv, **kwargs):
    cols = feature_cols or DEFAULT_TEXT_COLS
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=lexical_csv,
        head_name="txt",
        feature_cols=cols,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg3"],
    )


def run_audio_catboost_cfg1(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, audio_csv, **kwargs):
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=audio_csv,
        head_name="aud",
        feature_cols=feature_cols,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg1"],
    )


def run_audio_catboost_cfg2(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, audio_csv, **kwargs):
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=audio_csv,
        head_name="aud",
        feature_cols=feature_cols,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg2"],
    )


def run_audio_catboost_cfg3(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, audio_csv, **kwargs):
    return _run_single_modality_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        features_csv=audio_csv,
        head_name="aud",
        feature_cols=feature_cols,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg3"],
    )


def run_early_fusion_catboost_cfg1(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    return _run_early_fusion_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        lexical_csv=lexical_csv,
        audio_csv=audio_csv,
        text_cols=DEFAULT_TEXT_COLS,
        audio_cols=None,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg1"],
    )


def run_early_fusion_catboost_cfg2(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    return _run_early_fusion_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        lexical_csv=lexical_csv,
        audio_csv=audio_csv,
        text_cols=DEFAULT_TEXT_COLS,
        audio_cols=None,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg2"],
    )


def run_early_fusion_catboost_cfg3(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    return _run_early_fusion_catboost(
        train_df,
        test_df,
        inner_splits_idx,
        seed,
        lexical_csv=lexical_csv,
        audio_csv=audio_csv,
        text_cols=DEFAULT_TEXT_COLS,
        audio_cols=None,
        cat_params=CATBOOST_PARAM_CONFIGS["cfg3"],
    )


def run_late_fusion_blender_catboost(
    train_df,
    test_df,
    inner_splits_idx,
    seed,
    text_cols=None,
    audio_cols=None,
    *,
    lexical_csv,
    audio_csv,
    **kwargs,
):
    # load feature tables
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = text_cols or DEFAULT_TEXT_COLS
    if audio_cols is None:
        acols = infer_numeric_cols(tr_aud)
    else:
        acols = audio_cols

    X_txt_tr = tr_txt[tcols].to_numpy(dtype=float)
    X_aud_tr = tr_aud[acols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)

    X_txt_te = te_txt[tcols].to_numpy(dtype=float)
    X_aud_te = te_aud[acols].to_numpy(dtype=float)

    # OOF head predictions
    p_txt_oof = np.zeros(len(train_df), dtype=float)
    p_aud_oof = np.zeros(len(train_df), dtype=float)

    for tr_idx, va_idx in inner_splits_idx:
        mt = make_catboost(seed)
        ma = make_catboost(seed)

        p_txt_oof[va_idx] = fit_catboost_predict_proba(mt, X_txt_tr[tr_idx], y_tr[tr_idx], X_txt_tr[va_idx])
        p_aud_oof[va_idx] = fit_catboost_predict_proba(ma, X_aud_tr[tr_idx], y_tr[tr_idx], X_aud_tr[va_idx])

    Z_oof = np.column_stack([p_txt_oof, p_aud_oof])

    # blender on OOF
    blender = make_catboost(seed, depth=3, iterations=2000, learning_rate=0.05)
    blender.fit(Z_oof, y_tr)
    p_fus_oof = blender.predict_proba(Z_oof)[:, 1]

    # refit heads on full outer-train
    mt_full = make_catboost(seed)
    ma_full = make_catboost(seed)

    mt_full.fit(X_txt_tr, y_tr)
    ma_full.fit(X_aud_tr, y_tr)

    p_txt_te = mt_full.predict_proba(X_txt_te)[:, 1]
    p_aud_te = ma_full.predict_proba(X_aud_te)[:, 1]

    # fit blender on full-train head preds for test-time inference
    p_txt_tr_full = mt_full.predict_proba(X_txt_tr)[:, 1]
    p_aud_tr_full = ma_full.predict_proba(X_aud_tr)[:, 1]
    Z_full = np.column_stack([p_txt_tr_full, p_aud_tr_full])

    blender_full = make_catboost(seed, depth=3, iterations=2000, learning_rate=0.05)
    blender_full.fit(Z_full, y_tr)

    Z_te = np.column_stack([p_txt_te, p_aud_te])
    p_fus_te = blender_full.predict_proba(Z_te)[:, 1]

    test_probs = {"txt": p_txt_te, "aud": p_aud_te, "fus": p_fus_te}
    oof_probs = {"txt": p_txt_oof, "aud": p_aud_oof, "fus": p_fus_oof}
    return test_probs, oof_probs

def run_audio_core_lr(train_df, test_df, inner_splits_idx, seed, *, audio_csv, **kwargs):
    train_feats = load_features_for_split(audio_csv, train_df["clip_id"])
    test_feats  = load_features_for_split(audio_csv, test_df["clip_id"])

    cols = get_core_audio_cols(train_feats)

    X_tr = train_feats[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = test_feats[cols].to_numpy(dtype=float)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"aud": p_test}, {"aud": p_oof}


def run_audio_core_catboost(train_df, test_df, inner_splits_idx, seed, *, audio_csv, **kwargs):
    train_feats = load_features_for_split(audio_csv, train_df["clip_id"])
    test_feats  = load_features_for_split(audio_csv, test_df["clip_id"])

    cols = get_core_audio_cols(train_feats)

    X_tr = train_feats[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = test_feats[cols].to_numpy(dtype=float)

    model = make_catboost(seed, **CATBOOST_PARAM_CONFIGS["cfg1"])
    p_test = fit_catboost_predict_proba(model, X_tr, y_tr, X_te)

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_catboost(seed, **CATBOOST_PARAM_CONFIGS["cfg1"])
        p_oof[va_idx] = fit_catboost_predict_proba(m, X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx])

    return {"aud": p_test}, {"aud": p_oof}


def run_early_fusion_coreaudio_catboost(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, audio_csv, **kwargs):
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = DEFAULT_TEXT_COLS
    acols = get_core_audio_cols(tr_aud)

    X_tr = np.concatenate(
        [tr_txt[tcols].to_numpy(dtype=float), tr_aud[acols].to_numpy(dtype=float)],
        axis=1
    )
    y_tr = train_df["label"].to_numpy(dtype=int)

    X_te = np.concatenate(
        [te_txt[tcols].to_numpy(dtype=float), te_aud[acols].to_numpy(dtype=float)],
        axis=1
    )

    model = make_catboost(seed, **CATBOOST_PARAM_CONFIGS["cfg1"])
    p_test = fit_catboost_predict_proba(model, X_tr, y_tr, X_te)

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_catboost(seed, **CATBOOST_PARAM_CONFIGS["cfg1"])
        p_oof[va_idx] = fit_catboost_predict_proba(m, X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx])

    return {"fus": p_test}, {"fus": p_oof}

def run_early_fusion_audio_pca_catboost(
    train_df, test_df, inner_splits_idx, seed,
    *, lexical_csv, audio_csv, n_components=50, **kwargs
):
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = DEFAULT_TEXT_COLS
    acols = infer_numeric_cols(tr_aud)  # all numeric audio cols

    X_txt_tr = tr_txt[tcols].to_numpy(dtype=float)
    X_txt_te = te_txt[tcols].to_numpy(dtype=float)

    X_aud_tr = tr_aud[acols].to_numpy(dtype=float)
    X_aud_te = te_aud[acols].to_numpy(dtype=float)

    Z_tr, Z_te = fit_audio_pca_transform(X_aud_tr, X_aud_te, n_components=n_components, seed=seed)

    X_tr = np.concatenate([X_txt_tr, Z_tr], axis=1)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = np.concatenate([X_txt_te, Z_te], axis=1)

    model = make_catboost(seed, **CATBOOST_PARAM_CONFIGS["cfg1"])
    p_test = fit_catboost_predict_proba(model, X_tr, y_tr, X_te)

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_catboost(seed, **CATBOOST_PARAM_CONFIGS["cfg1"])
        p_oof[va_idx] = fit_catboost_predict_proba(m, X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx])

    return {"fus": p_test}, {"fus": p_oof}

def run_early_fusion_audio_pca_lr(
    train_df, test_df, inner_splits_idx, seed,
    *, lexical_csv, audio_csv, n_components=50, **kwargs
):
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = DEFAULT_TEXT_COLS
    acols = infer_numeric_cols(tr_aud)  # all numeric audio cols

    X_txt_tr = tr_txt[tcols].to_numpy(dtype=float)
    X_txt_te = te_txt[tcols].to_numpy(dtype=float)

    X_aud_tr = tr_aud[acols].to_numpy(dtype=float)
    X_aud_te = te_aud[acols].to_numpy(dtype=float)

    Z_tr, Z_te = fit_audio_pca_transform(X_aud_tr, X_aud_te, n_components=n_components, seed=seed)

    X_tr = np.concatenate([X_txt_tr, Z_tr], axis=1)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = np.concatenate([X_txt_te, Z_te], axis=1)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"fus": p_test}, {"fus": p_oof}

def run_embed_lr(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, embed_csv, **kwargs):
    if not embed_csv:
        raise ValueError("embed_csv is required for embed_lr. Pass --embed_csv path/to/embeddings.csv")

    feats_tr = load_features_for_split(embed_csv, train_df["clip_id"])
    feats_te = load_features_for_split(embed_csv, test_df["clip_id"])

    cols = feature_cols or infer_numeric_cols(feats_tr)

    X_tr = feats_tr[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = feats_te[cols].to_numpy(dtype=float)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    # return as "txt" so it lands in p_txt column in predictions.csv
    return {"txt": p_test}, {"txt": p_oof}


def run_embed_catboost_cfg1(train_df, test_df, inner_splits_idx, seed, feature_cols=None, *, embed_csv, **kwargs):
    if not embed_csv:
        raise ValueError("embed_csv is required for embed_catboost_cfg1. Pass --embed_csv path/to/embeddings.csv")

    feats_tr = load_features_for_split(embed_csv, train_df["clip_id"])
    feats_te = load_features_for_split(embed_csv, test_df["clip_id"])

    cols = feature_cols or infer_numeric_cols(feats_tr)

    X_tr = feats_tr[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = feats_te[cols].to_numpy(dtype=float)

    params = CATBOOST_PARAM_CONFIGS["cfg1"]
    model = make_catboost(seed, **params)
    p_test = fit_catboost_predict_proba(model, X_tr, y_tr, X_te)

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_catboost(seed, **params)
        p_oof[va_idx] = fit_catboost_predict_proba(
            m, X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx]
        )

    return {"txt": p_test}, {"txt": p_oof}

def run_text_lr_no_entropy(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, **kwargs):
    return run_text_lr(train_df, test_df, inner_splits_idx, seed,
                       lexical_csv=lexical_csv,
                       feature_cols=["mattr", "ttr", "mean_sentence_len", "pronoun_ratio", "sem_drift"])

def run_text_lr_no_drift(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, **kwargs):
    return run_text_lr(train_df, test_df, inner_splits_idx, seed,
                       lexical_csv=lexical_csv,
                       feature_cols=["mattr", "ttr", "mean_sentence_len", "pronoun_ratio", "lex_entropy"])

def run_text_lr_traditional_only(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, **kwargs):
    return run_text_lr(train_df, test_df, inner_splits_idx, seed,
                       lexical_csv=lexical_csv,
                       feature_cols=["mattr", "ttr", "mean_sentence_len", "pronoun_ratio"])

def run_text_lr_novel_only(train_df, test_df, inner_splits_idx, seed, *, lexical_csv, **kwargs):
    return run_text_lr(train_df, test_df, inner_splits_idx, seed,
                       lexical_csv=lexical_csv,
                       feature_cols=["lex_entropy", "sem_drift"])

def run_audio_core_lr(train_df, test_df, inner_splits_idx, seed, *, audio_csv, **kwargs):
    train_feats = load_features_for_split(audio_csv, train_df["clip_id"])
    test_feats  = load_features_for_split(audio_csv, test_df["clip_id"])

    cols = get_core_audio_cols(train_feats)

    X_tr = train_feats[cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = test_feats[cols].to_numpy(dtype=float)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"aud": p_test}, {"aud": p_oof}

def run_early_fusion_audio_pca_lr(
    train_df, test_df, inner_splits_idx, seed,
    *, lexical_csv, audio_csv, n_components=50, **kwargs
):
    tr_txt = load_features_for_split(lexical_csv, train_df["clip_id"])
    te_txt = load_features_for_split(lexical_csv, test_df["clip_id"])
    tr_aud = load_features_for_split(audio_csv, train_df["clip_id"])
    te_aud = load_features_for_split(audio_csv, test_df["clip_id"])

    tcols = DEFAULT_TEXT_COLS
    acols = infer_numeric_cols(tr_aud)  # all numeric audio cols

    X_txt_tr = tr_txt[tcols].to_numpy(dtype=float)
    X_txt_te = te_txt[tcols].to_numpy(dtype=float)

    X_aud_tr = tr_aud[acols].to_numpy(dtype=float)
    X_aud_te = te_aud[acols].to_numpy(dtype=float)

    Z_tr, Z_te = fit_audio_pca_transform(X_aud_tr, X_aud_te, n_components=n_components, seed=seed)

    X_tr = np.concatenate([X_txt_tr, Z_tr], axis=1)
    y_tr = train_df["label"].to_numpy(dtype=int)
    X_te = np.concatenate([X_txt_te, Z_te], axis=1)

    model = make_lr(seed)
    model.fit(X_tr, y_tr)
    p_test = model.predict_proba(X_te)[:, 1]

    p_oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in inner_splits_idx:
        m = make_lr(seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    return {"fus": p_test}, {"fus": p_oof}

# registry for experiments as dict
EXPERIMENTS: Dict[str, Callable[..., tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]] = {
    "text_lr": run_text_lr,
    "text_lexical_embed_lr": run_text_lexical_embed_lr,
    "text_lr_speaker": run_text_lr_speaker,
    "text_lr_timeaware": run_text_lr_timeaware,
    "audio_lr": run_audio_lr,
    "audio_lr_speaker": run_audio_lr_speaker,
    "audio_lr_timeaware": run_audio_lr_timeaware,
    "late_fusion_lr": run_late_fusion_lr,
    "early_fusion_lr_speaker": run_early_fusion_lr_speaker,
    "early_fusion_lr_timeaware": run_early_fusion_lr_timeaware,
    "text_svm_cal": run_text_svm_cal,
    "text_mlp": run_text_mlp,
    "audio_svm_cal": run_audio_svm_cal,
    "audio_mlp": run_audio_mlp,
    "early_fusion_lr": run_early_fusion_lr,
    "early_fusion_svm_cal": run_early_fusion_svm_cal,
    "early_fusion_mlp": run_early_fusion_mlp,
    "text_catboost": run_text_catboost,
    "audio_catboost": run_audio_catboost,
    "early_fusion_catboost": run_early_fusion_catboost,
    "text_catboost_cfg1": run_text_catboost_cfg1,
    "text_catboost_cfg2": run_text_catboost_cfg2,
    "text_catboost_cfg3": run_text_catboost_cfg3,
    "audio_catboost_cfg1": run_audio_catboost_cfg1,
    "audio_catboost_cfg2": run_audio_catboost_cfg2,
    "audio_catboost_cfg3": run_audio_catboost_cfg3,
    "early_fusion_catboost_cfg1": run_early_fusion_catboost_cfg1,
    "early_fusion_catboost_cfg2": run_early_fusion_catboost_cfg2,
    "early_fusion_catboost_cfg3": run_early_fusion_catboost_cfg3,
    "late_fusion_blender_catboost": run_late_fusion_blender_catboost,
    "audio_core_lr": run_audio_core_lr,
    "audio_core_catboost": run_audio_core_catboost,
    "early_fusion_coreaudio_catboost": run_early_fusion_coreaudio_catboost,
    "early_fusion_audio_pca_lr": run_early_fusion_audio_pca_lr,
    "early_fusion_audio_pca_catboost": run_early_fusion_audio_pca_catboost,
    "embed_lr": run_embed_lr,
    "embed_catboost_cfg1": run_embed_catboost_cfg1,
    "text_lr_no_entropy":       run_text_lr_no_entropy,
    "text_lr_no_drift":         run_text_lr_no_drift,
    "text_lr_traditional_only": run_text_lr_traditional_only,
    "text_lr_novel_only":       run_text_lr_novel_only,
    "audio_lr": run_audio_lr,
    "audio_core_lr": run_audio_core_lr,
    "early_fusion_audio_pca_lr": run_early_fusion_audio_pca_lr,
}
