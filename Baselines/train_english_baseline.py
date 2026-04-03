from train_ner_baseline import main


if __name__ == "__main__":
    main(
        {
            "train_file": "Datasets/English/en_ewt-ud-train.iob2",
            "dev_file": "Datasets/English/en_ewt-ud-dev.iob2",
            "test_file": "Datasets/English/en_ewt-ud-test-masked.iob2",
            "model_name": "bert-base-multilingual-cased",
            "output_dir": "outputs/english_bert_baseline",
            "label_report_path": "Baselines/reports/english_labels.json",
        }
    )