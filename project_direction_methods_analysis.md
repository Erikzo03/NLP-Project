# Project Direction, Methods, and Analysis Plan

## 1) Project Direction (Final Choice)

### Main direction
Cross-lingual transfer for Danish NER, using English labeled data as the source and Danish as the target language.

### Core research question
How well can an English-trained NER model transfer to Danish, and which lightweight transfer strategy gives the best Danish F1 under a realistic semester-project budget?

### Scope decision
Use one project idea only (cross-lingual transfer). Do not combine with a separate full data-augmentation project.

## 2) Data and Label Setup

### Source data
- English EWT NER files in this repo:
  - en_ewt-ud-train.iob2
  - en_ewt-ud-dev.iob2
  - en_ewt-ud-test-masked.iob2

### Target data
- Primary Danish dataset: DaNE (Hugging Face: alexandrainst/dane).
- Fallback: Danish WikiANN subset.

### Label schema
Use consistent BIO labels with PER/ORG/LOC and O.

## 3) Candidate Methods We May Use

## Method A: Zero-shot multilingual transfer (required baseline)
Train on English only and evaluate directly on Danish.

- Model options:
  - bert-base-multilingual-cased (mBERT)
  - xlm-roberta-base (XLM-R)
- Why include:
  - Establishes the core cross-lingual baseline.
  - Simple and reproducible.

## Method B: Few-shot continued fine-tuning on Danish (recommended main method)
Start from Method A model, then continue training with small Danish subsets.

- Suggested Danish fractions: 10%, 25%, 50%, 100%.
- Why include:
  - Answers data-efficiency question.
  - Usually provides large gains over pure zero-shot.

## Method C: Projection/translation-assisted transfer (optional extension)
Generate weak Danish labels via translation/projection and train on projected data.

- Inspired by projection and CROP-style pipelines.
- Why include:
  - Strong cross-lingual literature connection.
  - Good novelty if implementation is manageable.

## Method D: Gazetteer-enhanced transfer (optional extension)
Inject Danish entity lists (PER/ORG/LOC) from external resources.

- Use as extra features or post-processing support.
- Why include:
  - Practical low-cost boost in low-resource transfer settings.

## Recommended semester-safe method stack
1. Method A (must-have baseline)
2. Method B (main contribution)
3. One of C or D only if time allows

## 4) Proposed Experiment Matrix

1. E1: mBERT English-only -> Danish test/dev (zero-shot).
2. E2: XLM-R English-only -> Danish test/dev (zero-shot).
3. E3: Best zero-shot model + 10% Danish fine-tuning.
4. E4: Best zero-shot model + 25% Danish fine-tuning.
5. E5: Best zero-shot model + 50% Danish fine-tuning.
6. E6: Best zero-shot model + 100% Danish fine-tuning.
7. E7 (optional): Add projection or gazetteer enhancement.

## 5) Evaluation and Analysis to Perform

## Primary metrics
- Strict span-level precision/recall/F1 (CoNLL style).
- Use repo evaluators:
  - conlleval.pl
  - span_f1.py

## Secondary metrics
- Per-class F1 (PER, ORG, LOC).
- Unlabeled span F1.
- Loose overlap F1.

## Required analyses
1. Learning curve analysis:
- Plot F1 vs Danish fine-tuning fraction (10/25/50/100).

2. Error type analysis:
- Boundary errors vs label errors.
- Cases where span is correct but entity type is wrong.

3. Per-class behavior:
- Which entities transfer best from English to Danish (PER/ORG/LOC)?

4. Qualitative error analysis:
- 20 to 50 sampled errors with short categories (name variants, capitalization, multi-token spans, etc.).

5. Ablation (at least one):
- mBERT vs XLM-R under same training setup.
- Optional second ablation: with/without optional enhancement (C or D).

## 6) Reproducibility Requirements

- Fixed random seeds.
- Fixed train/dev/test split definitions.
- Keep prediction output format aligned with scorer scripts.
- Document exact hyperparameters and model checkpoints.

## 7) Risks and Mitigations

1. Label/schema mismatch between datasets.
- Mitigation: explicit mapping checks before training.

2. Domain mismatch between English and Danish corpora.
- Mitigation: report zero-shot gap and rely on few-shot adaptation curve.

3. Time/compute limits.
- Mitigation: prioritize A+B and one clean analysis package before optional methods.

## 8) Deliverable Priorities

1. Strong, reproducible baseline (A).
2. Clear improvement story via few-shot transfer (B).
3. Solid analysis section with tables/plots and error discussion.
4. Optional extension only if baseline and analysis are complete.
