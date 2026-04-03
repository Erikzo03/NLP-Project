from train_ner_baseline import main


if __name__ == "__main__":
    main(
        {
            "train_file": "Datasets/Norwegian/nno_norne-ud-train.iob2",
            "dev_file": "Datasets/Norwegian/nno_norne-ud-dev.iob2",
            "test_file": "Datasets/Norwegian/nno_norne-ud-test.iob2",
            "model_name": "bert-base-multilingual-cased",
            "output_dir": "outputs/norwegian_bert_baseline",
            "label_report_path": "Baselines/reports/norwegian_labels.json",
        }
    )