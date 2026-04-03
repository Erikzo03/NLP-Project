from train_ner_baseline import main


if __name__ == "__main__":
    main(
        {
            "train_file": "Datasets/Danish/da_ddt-ud-train.iob2",
            "dev_file": "Datasets/Danish/da_ddt-ud-dev.iob2",
            "test_file": "Datasets/Danish/da_ddt-ud-test.iob2",
            "model_name": "bert-base-multilingual-cased",
            "output_dir": "outputs/danish_bert_baseline",
            "label_report_path": "Baselines/reports/danish_labels.json",
        }
    )