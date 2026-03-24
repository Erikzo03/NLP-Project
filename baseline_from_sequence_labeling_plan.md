# How to Reuse sequence_labeling.ipynb for This Project

Notebook reviewed:
- /Users/erikaymerich/Downloads/sequence_labeling.ipynb

## What is directly reusable
1. Tokenization + label alignment logic (`tokenize_and_align_labels`).
2. Data collator and PyTorch DataLoader setup.
3. Training loop skeleton (TODO B block).
4. seqeval metric computation structure (`get_labels`, `compute_metrics`).

## What must be changed for your project
1. Replace dataset loader:
- Current notebook loads `conll2003` via `load_dataset`.
- Your project uses local IOB2 files and later Danish dataset files.

2. Remove HPC shell cells:
- `ssh`, `srun`, `conda create` cells are shell commands, not notebook Python code for local execution.

3. Match project evaluation format:
- Keep sentence boundaries and token order.
- Write predictions in same CoNLL-like format expected by `span_f1.py` and `conlleval.pl`.

4. Keep label set consistent:
- English source labels are PER/ORG/LOC in BIO tags.
- Ensure Danish labels map to exactly the same scheme before training/evaluation.

## Minimum baseline implementation order
1. Build local data reader for IOB2 files.
2. Build Hugging Face DatasetDict from local splits.
3. Reuse tokenizer/label alignment function.
4. Implement TODO B training loop.
5. Implement TODO C `get_labels` function.
6. Run dev evaluation with seqeval.
7. Export prediction file and evaluate with `span_f1.py` + `conlleval.pl`.

## Recommended first baseline config
- Model: `bert-base-multilingual-cased` (for cross-lingual path)
- Learning rate: 2e-5
- Epochs: 3 to 5
- Batch size: 8 or 16
- Max length: 128
- Seed: 42

## Cross-lingual experiment sequence
1. Train on English only -> evaluate Danish (zero-shot).
2. Continue fine-tuning with 10/25/50/100% Danish train splits.
3. Compare against Danish-only training.
