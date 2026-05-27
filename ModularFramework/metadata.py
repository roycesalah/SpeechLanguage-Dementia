# build_metadata.py
"""
Build metadata.csv from text_features_lexical.csv.

Input
==================
clip_id

  data/dementia/B B King/bbking_15.wav
  data/dementia/David Prowse/DavidProwse_0.wav
  data/dementia/George Klein/georgeklein_5_1.wav
  data/nodementia/Barbra Streisand/BarbraStreisand_1.wav

Output
==================
metadata.csv with columns:
  clip_id, condition, label, speaker, years_pre_dx, clip_num

Usage
==================
python build_metadata.py --lexical_csv path/to/text_features_lexical.csv --out_csv path/to/metadata.csv
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def parse_clip_id(clip_id: str):
    s = str(clip_id).strip().replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    if len(parts) < 4:
        raise ValueError(f"clip_id path too short to parse: {clip_id}")

    condition = parts[1].lower()
    if condition not in {"dementia", "nodementia"}:
        raise ValueError(f"Unknown condition segment '{parts[1]}' in clip_id: {clip_id}")

    label = 1 if condition == "dementia" else 0
    speaker = parts[2]

    filename = parts[-1]
    stem = re.sub(r"\.[^.]+$", "", filename)
    tokens = stem.split("_")

    years_pre_dx = float("nan")
    clip_num = None

    if condition == "dementia":
        if len(tokens) < 2:
            raise ValueError(f"Unexpected dementia filename format: {clip_id}")

        # If end of string is numeric and one before is numeric then parse (years, clipnum)
        if tokens[-1].isdigit() and tokens[-2].isdigit():
            years_pre_dx = float(tokens[-2])
            clip_num = int(tokens[-1])
        # Else if last token is numeric then num yrs
        elif tokens[-1].isdigit():
            years_pre_dx = float(tokens[-1])
            clip_num = 1
        else:
            raise ValueError(f"Could not parse dementia years_pre_dx from filename: {clip_id}")

    else:
        # nodementia []_<clipnum> (clipnum numeric at end)
        if len(tokens) >= 2 and tokens[-1].isdigit():
            clip_num = int(tokens[-1])
        else:
            clip_num = None
        years_pre_dx = float("nan")

    return condition, label, speaker, years_pre_dx, clip_num



def build_metadata(lexical_csv: str) -> pd.DataFrame:
    df = pd.read_csv(lexical_csv)

    first_col = df.columns[0]
    clip_ids = df[first_col].astype(str).tolist()

    rows = []
    for cid in clip_ids:
        condition, label, speaker, years_pre_dx, clip_num = parse_clip_id(cid)
        rows.append(
            {
                "clip_id": cid,
                "condition": condition,
                "label": label,
                "speaker": speaker,
                "years_pre_dx": years_pre_dx,
                "clip_num": clip_num,
            }
        )

    meta = pd.DataFrame(rows)

    # Check for duplicates
    if meta["clip_id"].duplicated().any():
        dups = meta.loc[meta["clip_id"].duplicated(), "clip_id"].head(10).tolist()
        raise ValueError(f"Duplicate clip_id values found (showing up to 10): {dups}")

    # Ensure type
    meta["label"] = meta["label"].astype(int)
    # years_pre_dx may be NaN for controls; keep float
    meta["years_pre_dx"] = pd.to_numeric(meta["years_pre_dx"], errors="coerce")
    meta["clip_num"] = pd.to_numeric(meta["clip_num"], errors="coerce")

    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lexical_csv", required=True, help="Path to text_features_lexical.csv")
    p.add_argument("--out_csv", required=True, help="Path to write metadata.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = build_metadata(args.lexical_csv)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    meta.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(meta)} rows)")
    print(meta.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
