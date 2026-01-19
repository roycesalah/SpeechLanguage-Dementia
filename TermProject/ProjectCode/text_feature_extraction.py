#!/usr/bin/env python
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, torch
from tqdm import tqdm

import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

_tok  = AutoTokenizer.from_pretrained("gpt2")
_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2",
                                             torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
                                             ).to(DEVICE).eval()

NLP = spacy.load("en_core_web_sm", disable=["ner"])

# helper functions
if (
    "parser" not in NLP.pipe_names
    and "senter" not in NLP.pipe_names
    and "sentencizer" not in NLP.pipe_names
):
    NLP.add_pipe("sentencizer")

def split_sent(doc: "spacy.tokens.Doc"):
    try:
        sents = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
    except Exception:
        sents = []
    return sents

def sem_drift(vecs: np.ndarray) -> float:
    if len(vecs) < 2: return 0.0
    norms = np.linalg.norm(vecs, axis=1)
    cos   = np.sum(vecs[:-1] * vecs[1:], axis=1) / (norms[:-1] * norms[1:])

    return float(np.mean(1 - cos))

@torch.no_grad()
def lex_entropy(text: str, max_len=512) -> float:
    ids = _tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
    if ids.input_ids.shape[1] < 3: return 0.0
    logits = _gpt2(**ids).logits[:, :-1]
    probs  = torch.softmax(logits.float(), dim=-1).clamp_(min=1e-8)

    return float((-probs * torch.log2(probs)).sum(-1).mean().item())

def mattr(tokens, window=50):
    if not tokens: return 0.0
    if len(tokens) < window: return len(set(tokens)) / len(tokens)
    scores, uniq = [], Counter()
    for i, tok in enumerate(tokens):
        uniq[tok] += 1
        if i >= window:
            prev = tokens[i-window]
            uniq[prev] -= 1
            if uniq[prev] == 0: del uniq[prev]
        if i >= window-1: scores.append(len(uniq)/window)
        
    return float(np.mean(scores))

rows = []
for txt_path in tqdm(Path("data").glob("*/**/*.txt"), unit="clip"):
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")[:30_000]

    doc    = NLP(raw)
    tokens = [t.text.lower() for t in doc if t.is_punct == False and t.is_alpha and not t.is_space]

    # classic lexical metrics
    ttr_val   = len(set(tokens))/len(tokens) if tokens else 0.0
    mattr_val = mattr(tokens)
    sent_lens = [len([t for t in sent if not t.is_space]) for sent in doc.sents]
    mean_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0
    pron  = sum(1 for t in doc if t.pos_ == "PRON")
    nouns = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
    pronoun_ratio = pron/(pron+nouns) if (pron+nouns) else 0.0

    # SBERT sentence vectors
    sentences = split_sent(raw)
    if sentences:
        vecs = sbert.encode(sentences, convert_to_numpy=True)
        drift_val = sem_drift(vecs)
        doc_emb_full = vecs.mean(axis=0)  # 384-dim pooled embedding
    else:
        drift_val = 0.0
        doc_emb_full = np.zeros(sbert.get_sentence_embedding_dimension())

    # keep first 50 dims for compaction
    doc_emb_50 = doc_emb_full[:50]

    entropy_val = lex_entropy(raw)

    row = {
        "clip_id"          : txt_path.with_suffix(".wav").as_posix(),
        "sem_drift"        : drift_val,
        "lex_entropy"      : entropy_val,
        "ttr"              : ttr_val,
        "mattr"            : mattr_val,
        "mean_sentence_len": mean_sent_len,
        "pronoun_ratio"    : pronoun_ratio,
    }
    for i, val in enumerate(doc_emb_50):
        row[f"sbert_{i}"] = float(val)  # SBERT append

    rows.append(row)

pd.DataFrame(rows).to_csv("text_features_lexical.csv", index=False)
