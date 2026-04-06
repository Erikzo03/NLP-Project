import os
from train_ner_baseline import main

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    main(
        {
            "train_file": os.path.join(BASE_DIR, "Datasets/English/en_ewt-ud-train.iob2"),
            "dev_file": os.path.join(BASE_DIR, "Datasets/English/en_ewt-ud-dev.iob2"),
            "test_file": os.path.join(BASE_DIR, "Datasets/English/en_ewt-ud-test-masked.iob2"),
            "model_name": "bert-base-multilingual-cased",
            "output_dir": os.path.join(BASE_DIR, "outputs/english_bert_baseline"),
            "label_report_path": os.path.join(BASE_DIR, "Baselines/reports/english_labels.json"),
        }
    )