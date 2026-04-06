import os
from train_ner_baseline import main

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    main(
        {
            "train_file": os.path.join(BASE_DIR, "Datasets/Norwegian/nno_norne-ud-train.iob2"),
            "dev_file": os.path.join(BASE_DIR, "Datasets/Norwegian/nno_norne-ud-dev.iob2"),
            "test_file": os.path.join(BASE_DIR, "Datasets/Norwegian/nno_norne-ud-test.iob2"),
            "model_name": "xlm-roberta-base",
            "output_dir": os.path.join(BASE_DIR, "outputs/norwegian_xlmr_baseline"),
            "label_report_path": os.path.join(BASE_DIR, "Baselines/reports/norwegian_labels_xlmr.json"),
        }
    )