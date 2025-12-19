#!/usr/bin/env python
"""
Recursively transcribes all .wav files under data\,
saving a .txt transcript next to each clip.
"""
import argparse, os
from pathlib import Path
import whisper, torch
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--model", default="base",
                   help="tiny, base, small, medium, large")
    p.add_argument("--device", default="auto",
                   help="cpu, cuda, or auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def pick_device(d):
    return "cuda" if d == "auto" and torch.cuda.is_available() else d

def main():
    args   = get_args()
    device = pick_device(args.device)
    model  = whisper.load_model(args.model, device=device)
    wavs   = list(Path(args.data_root).glob("*/**/*.wav"))
    if not wavs:
        raise SystemExit("No .wav files found!")
    for wav in tqdm(wavs, unit="clip"):
        txt = wav.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            continue
        try:
            out = model.transcribe(str(wav), fp16=(device=="cuda"))
            txt.write_text(out["text"].strip(), encoding="utf-8")
        except Exception as e:
            print(f"Failed on {wav}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
