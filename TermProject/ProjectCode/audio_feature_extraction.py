#!/usr/bin/env python
from __future__ import annotations
import os
from pathlib import Path

import librosa
import numpy as np
import opensmile
import pandas as pd
from tqdm import tqdm


"""
eGeMAPS functional features
n_sil        =    # pauses ≥ 0.2 s
mean_sil     =    mean pause duration (s)
silence_pct  =    fraction of clip that is silent
voiced_sec   =    total voiced speech seconds
wpm          =    words per minute (transcript‑based if possible)
clip_id      =    wav path (relative)
"""

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Compute pauses
def pause_stats(
    y: np.ndarray,
    sr: int,
    *,
    db_thresh: float = -35.0,
    min_silence_dur: float = 0.2,
) -> dict[str, float]:

    nonsil = librosa.effects.split(y, top_db=abs(db_thresh))

    total_sec = len(y) / sr
    if nonsil.size == 0:
        return {
            "n_sil": 0,
            "mean_sil": 0.0,
            "silence_pct": 1.0,
            "voiced_sec": 0.0,
        }

    voiced_lengths = nonsil[:, 1] - nonsil[:, 0]
    voiced_sec = voiced_lengths.sum() / sr

    if len(nonsil) > 1:
        sil_durs = (nonsil[1:, 0] - nonsil[:-1, 1]) / sr
        sil_durs = sil_durs[sil_durs >= min_silence_dur]
    else:
        sil_durs = np.asarray([])

    return {
        "n_sil": int(len(sil_durs)),
        "mean_sil": float(sil_durs.mean()) if sil_durs.size else 0.0,
        "silence_pct": float(1.0 - voiced_sec / total_sec),
        "voiced_sec": float(voiced_sec),
    }

# Audio file iterator
root = Path("data")
wav_files = list(root.glob("*/**/*.wav"))
rows: list[dict[str, float]] = []

for wav in tqdm(wav_files, desc="audio", unit="clip"):
    try:
        sm_df = smile.process_file(wav)

        y, sr = librosa.load(wav, sr=None, mono=True)
        tempo = pause_stats(y, sr)

        txt_path = wav.with_suffix(".txt")
        if txt_path.exists():
            words = len(txt_path.read_text(errors="ignore").split())
            total_sec = (len(y) / sr) or 1e-9
            wpm = words / (total_sec / 60.0)
        else:
            voiced_wps = 2.5
            total_sec = (len(y) / sr) or 1e-9
            wpm = (tempo["voiced_sec"] * voiced_wps) / (total_sec / 60.0)

        # 3d. merge dictionaries
        feats = sm_df.iloc[0].to_dict()
        feats.update(tempo)
        feats["wpm"] = float(wpm)
        feats["clip_id"] = wav.as_posix()

        rows.append(feats)

    except Exception as e:
        print("ERR:", wav, "->", e)


df = pd.DataFrame(rows)
df.to_csv("audio_features.csv", index=False)