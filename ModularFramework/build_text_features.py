#!/usr/bin/env python
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_sentencizer(nlp):
    if (
        "parser" not in nlp.pipe_names
        and "senter" not in nlp.pipe_names
        and "sentencizer" not in nlp.pipe_names
    ):
        nlp.add_pipe("sentencizer")
    return nlp


def split_sent(doc):
    sents = []
    try:
        for s in doc.sents:
            t = s.text.strip()
            if t:
                sents.append(t)
    except Exception:
        pass
    return sents


def sem_drift(vecs):
    if vecs is None or len(vecs) < 2:
        return 0.0
    norms = np.linalg.norm(vecs, axis=1)
    denom = norms[:-1] * norms[1:]
    denom = np.where(denom == 0, 1e-8, denom)
    cos = np.sum(vecs[:-1] * vecs[1:], axis=1) / denom
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.mean(1.0 - cos))


@torch.no_grad()
def lex_entropy(text, tok, lm, max_len=512):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
    if ids.input_ids.shape[1] < 3:
        return 0.0
    logits = lm(**ids).logits[:, :-1]
    probs = torch.softmax(logits.float(), dim=-1).clamp_(min=1e-8)
    return float((-probs * torch.log2(probs)).sum(-1).mean().item())


def mattr(tokens, window=50):
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return float(len(set(tokens)) / len(tokens))
    scores = []
    uniq = Counter()
    for i, tok in enumerate(tokens):
        uniq[tok] += 1
        if i >= window:
            prev = tokens[i - window]
            uniq[prev] -= 1
            if uniq[prev] == 0:
                del uniq[prev]
        if i >= window - 1:
            scores.append(len(uniq) / window)
    return float(np.mean(scores)) if scores else 0.0


def to_project_clip_id(txt_path, data_root, clip_suffix=".wav"):
    # txt_path: ...\data\dementia\Speaker Name\FirstnameLastname_5_2.txt
    # we want:  data/dementia/Speaker Name/FirstnameLastname_5_2.wav
    rel = txt_path.relative_to(data_root).as_posix()  # dementia/.../file.txt
    rel_wav = rel[:-4] + clip_suffix  # swap .txt -> .wav
    return "data/" + rel_wav


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Path to the data folder that contains dementia/ and nodementia/")
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    p.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    p.add_argument("--max_chars", type=int, default=30000, help="Max chars to read from each transcript")
    p.add_argument("--max_lm_len", type=int, default=512, help="Max tokens for GPT2 entropy")
    p.add_argument("--keep_emb_dims", type=int, default=0, help="0=don’t save embeddings; otherwise save first N dims")
    p.add_argument("--batch_size", type=int, default=32, help="SBERT batch size")
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    # spaCy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp = ensure_sentencizer(nlp)

    # SBERT
    sbert = SentenceTransformer(args.embed_model, device=DEVICE)

    # GPT2 for entropy (you can swap later, but keep stable for now)
    tok = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()

    rows = []
    txt_files = list(data_root.glob("*/**/*.txt"))

    for txt_path in tqdm(txt_files, unit="clip"):
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")[: args.max_chars]

        doc = nlp(raw)

        tokens = [
            t.text.lower()
            for t in doc
            if (not t.is_space) and (not t.is_punct) and t.is_alpha
        ]

        ttr_val = float(len(set(tokens)) / len(tokens)) if tokens else 0.0
        mattr_val = mattr(tokens)

        # sentence lengths (in tokens, excluding spaces/punct)
        sent_lens = []
        for s in doc.sents:
            sent_lens.append(sum(1 for t in s if (not t.is_space) and (not t.is_punct)))
        mean_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0

        pron = sum(1 for t in doc if t.pos_ == "PRON")
        nouns = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
        pronoun_ratio = float(pron / (pron + nouns)) if (pron + nouns) else 0.0

        # SBERT sentence vectors (for drift)
        sentences = split_sent(doc)
        if sentences:
            vecs = sbert.encode(
                sentences,
                convert_to_numpy=True,
                batch_size=args.batch_size,
                show_progress_bar=False
            )
            drift_val = sem_drift(vecs)
            doc_emb = vecs.mean(axis=0)
        else:
            drift_val = 0.0
            doc_emb = np.zeros(sbert.get_sentence_embedding_dimension(), dtype=float)

        entropy_val = lex_entropy(raw, tok, lm, max_len=args.max_lm_len)

        clip_id = to_project_clip_id(txt_path, data_root)

        row = {
            "clip_id": clip_id,
            "sem_drift": drift_val,
            "lex_entropy": entropy_val,
            "ttr": ttr_val,
            "mattr": mattr_val,
            "mean_sentence_len": mean_sent_len,
            "pronoun_ratio": pronoun_ratio,
        }

        if args.keep_emb_dims and args.keep_emb_dims > 0:
            k = int(args.keep_emb_dims)
            if k > len(doc_emb):
                k = len(doc_emb)
            for i in range(k):
                row[f"emb_{i}"] = float(doc_emb[i])

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  rows={len(out)}  embed_model={args.embed_model}  keep_emb_dims={args.keep_emb_dims}")


if __name__ == "__main__":
    main()
