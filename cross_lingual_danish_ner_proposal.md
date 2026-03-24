# Cross-Lingual Transfer for Danish NER from English Supervision

## Title
Cross-Lingual Transfer for Danish Named Entity Recognition with English-Supervised Transformers

## Team
[Add names and emails]

## Motivation and Context
Named Entity Recognition (NER) systems perform well in high-resource settings, but many languages have less annotated data. Danish is lower-resource than English for NER, and cross-lingual transfer with multilingual encoders is a practical way to reduce annotation needs. Our project studies how far we can transfer an English-trained NER model to Danish, and how much Danish supervision is needed to close the gap.

## Research Question
How effective is cross-lingual transfer from English to Danish for NER with PER/ORG/LOC labels, and what is the gain from adding limited Danish fine-tuning data?

## Hypotheses
1. Zero-shot transfer (train on English only, test on Danish) will outperform trivial baselines but remain clearly below Danish-supervised performance.
2. Continued fine-tuning with a small Danish subset will significantly improve Danish F1 over zero-shot.
3. XLM-R-base will outperform mBERT in zero-shot and few-shot settings.

## Data
### Source language (training)
- English Universal NER EWT files already in this repo:
  - en_ewt-ud-train.iob2
  - en_ewt-ud-dev.iob2
  - en_ewt-ud-test-masked.iob2
- Labels in source data: PER, ORG, LOC (BIO tags).

### Target language (Danish)
Primary dataset:
- DaNE (Danish NER resource): https://aclanthology.org/2020.lrec-1.565/
- Hugging Face dataset ID: alexandrainst/dane
- License reported by dataset card: CC BY-SA 4.0

Fallback dataset if access issues occur:
- WikiANN Danish (da) via multilingual WikiANN resources
- Mentioned in DaNLP dataset docs as available for Danish and compatible PER/ORG/LOC setting.

## Method
### Baseline model
- Token-classification fine-tuning with multilingual encoder.
- Initial baseline: mBERT (bert-base-multilingual-cased).
- Stronger variant: XLM-R-base.

### Transfer setups
1. Zero-shot:
- Train on English only.
- Evaluate directly on Danish test/dev.

2. Few-shot transfer:
- Train on English, then continue fine-tuning on 10%, 25%, 50%, and 100% Danish train split.
- Evaluate on fixed Danish dev/test.

3. Monolingual Danish reference (upper-bound within project scope):
- Train and evaluate on Danish only using same architecture and protocol.

### Evaluation
- Primary metric: strict span-level F1 (CoNLL style).
- Secondary metrics: precision, recall, per-label F1 (PER/ORG/LOC).
- Use project evaluators where possible:
  - conlleval.pl
  - span_f1.py

## Analysis Plan
1. Learning-curve analysis over Danish fine-tuning fraction (10/25/50/100%).
2. Error analysis:
- Boundary errors vs label errors.
- Per-class performance (PER/ORG/LOC).
- Typical confusion patterns and short qualitative examples.
3. Ablation:
- Compare mBERT vs XLM-R under identical training settings.

## Expected Contribution
- A reproducible cross-lingual NER pipeline from English to Danish.
- Clear estimate of data-efficiency: how much Danish data is needed to approach monolingual performance.
- Practical guidance for low-resource NER in related Germanic languages.

## Feasibility and Timeline
Week 1:
- Finalize dataset loading and label mapping.
- Run English baseline and zero-shot Danish evaluation.

Week 2:
- Run few-shot Danish transfer experiments.
- Start learning-curve and per-class analysis.

Week 3:
- Complete ablation (mBERT vs XLM-R).
- Finalize qualitative error analysis and write report.

## Risks and Mitigations
- Risk: Dataset format mismatch (IOB2 vs other format).
  - Mitigation: Build robust converter to the project evaluation format and validate on sample sentences.
- Risk: Domain mismatch between English and Danish corpora.
  - Mitigation: Report and analyze errors; include few-shot adaptation to reduce mismatch.
- Risk: Compute limits.
  - Mitigation: Use base-size models and fixed hyperparameter budget.

## Core References
- Hvingelby et al. 2020, DaNE: https://aclanthology.org/2020.lrec-1.565/
- Conneau et al. 2020, XLM-R: https://aclanthology.org/2020.acl-main.747/
- Pan et al. 2017, WikiANN: https://aclanthology.org/P17-1178/
