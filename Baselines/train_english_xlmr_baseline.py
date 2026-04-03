from train_ner_baseline import main


if __name__ == "__main__":
    main(
        {
            "train_file": "Datasets/English/en_ewt-ud-train.iob2",
            "dev_file": "Datasets/English/en_ewt-ud-dev.iob2",
            "test_file": "Datasets/English/en_ewt-ud-test-masked.iob2",
            "model_name": "xlm-roberta-base",
            "output_dir": "outputs/english_xlmr_baseline",
            "label_report_path": "Baselines/reports/english_labels_xlmr.json",
        }
    )