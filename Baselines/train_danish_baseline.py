import os
from train_ner_baseline import main

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    main(
        {
            "train_file": os.path.join(BASE_DIR,"Datasets/Danish/da_ddt-ud-train.iob2"),
            "dev_file": os.path.join(BASE_DIR,"Datasets/Danish/da_ddt-ud-dev.iob2"),
            "test_file": os.path.join(BASE_DIR,"Datasets/Danish/da_ddt-ud-test.iob2"),
            "model_name": "bert-base-multilingual-cased",
            "output_dir": os.path.join(BASE_DIR,"outputs/danish_bert_baseline"),
            "label_report_path": os.path.join(BASE_DIR,"Baselines/reports/danish_labels.json"),
        }
    )