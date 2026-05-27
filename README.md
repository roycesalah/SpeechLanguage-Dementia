# Speech-Language Dementia Screening

A multimodal machine learning framework for dementia risk screening from 
naturalistic speech, combining lexical, computational, and acoustic features 
with temporal stratification by years prior to diagnosis.

This codebase accompanies a paper currently under review

---

## Overview

The pipeline processes audio recordings through parallel text and acoustic 
feature extraction paths, then evaluates single-modality and fused classifiers 
under speaker-disjoint nested cross-validation. A key feature of this framework 
is time-to-diagnosis stratified evaluation, enabling analysis of how 
discriminative signal varies from the year of diagnosis to 15 years prior.

---

## Repository Structure
SpeechLanguage-Dementia/
- transcriber.py (Whisper ASR transcription)
- audio_feature_extraction.py (openSMILE eGeMAPSv02 + pause metrics + WPM)
- text_feature_extraction.py (spaCy lexical features + SBERT + GPT-2 entropy)
- ModularFramework/
-   build_text_features.py      # Text feature pipeline entry point
-     metadata.py                 # Builds metadata.csv from clip_id paths
-     experiments.py              # Experiment registry (LR, SVM, MLP, CatBoost, early/late fusion, ablations)
-     main_eval.py                # Nested CV evaluation runner
-     analyze.py                  # Post-processing, leaderboard, bin analysis
-     metrics.py                  # AUC, F1, sensitivity, specificity, Brier
-     utils.py                    # Shared utilities
- out (Experiment outputs and analysis results)

---

## Features

**Text features**
- `lex_entropy` — Mean per-token Shannon entropy from GPT-2 next-word distributions
- `sem_drift` — Mean cosine distance between adjacent sentence embeddings (SBERT)
- `mattr` — Moving-average type-token ratio
- `ttr` — Global type-token ratio
- `mean_sentence_len` — Mean token count per sentence
- `pronoun_ratio` — Pronoun-to-noun ratio
  

**Acoustic features**
- eGeMAPSv02 functionals (~88 dimensions) via openSMILE
- Voiced duration, words per minute, silence percentage, mean silence duration,
  silence count

---

## Data

This framework is designed for use with the 
[DementiaNet corpus](https://github.com/shreyasgite/dementianet), a publicly 
available longitudinal dataset of naturalistic speech from public figures with 
retrospectively assigned dementia labels.

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Core dependencies: `opensmile`, `whisper`, `sentence-transformers`, 
`transformers`, `spacy`, `scikit-learn`, `catboost`, `pandas`, `numpy`

---

## Usage

**1. Transcribe audio**
```bash
python transcriber.py --data_root data/ --out_dir data/
```

**2. Extract text features**
```bash
python ModularFramework/build_text_features.py --data_root data/ --out_csv data/text_features_lexical.csv
```

**3. Extract acoustic features**
```bash
python audio_feature_extraction.py --data_root data/ --out_csv data/audio_features.csv
```

**4. Build metadata**
```bash
python ModularFramework/metadata.py --lexical_csv data/text_features_lexical.csv --out_csv data/metadata.csv
```

**5. Run experiments**
```bash
python ModularFramework/main_eval.py --metadata data/metadata.csv --lexical_csv data/text_features_lexical.csv --out_dir out/table3_core_targetSens80 --experiments late_fusion_lr early_fusion_lr text_lr audio_lr text_svm_cal audio_core_lr
```

**6. Aggregate results**
```bash
python ModularFramework/analyze.py --results_dirs out/ --out_dir out/_analysis
```

---

## Evaluation

- Nested 5×5 StratifiedGroupKFold with speakers as groups (no speaker leakage)
- Target-sensitivity threshold policy (80% sensitivity on OOF training scores)
- Metrics: AUC, F1, sensitivity, specificity, Brier score
- Reported at both clip level and speaker level
- Temporal bin analysis across =0, (0,5], (5,10], (10,15] years pre-diagnosis

---

## Citation

If you use this code or framework, please cite the accompanying paper 
(citation details to be added upon completion of review).

---

## License

This repository is provided for research purposes. The DementiaNet corpus is 
subject to its own licensing terms at 
[github.com/shreyasgite/dementianet](https://github.com/shreyasgite/dementianet).
