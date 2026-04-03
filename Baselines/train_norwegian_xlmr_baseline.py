from train_ner_baseline import main


if __name__ == "__main__":
    main(
        {
            "train_file": "Datasets/Norwegian/nno_norne-ud-train.iob2",
            "dev_file": "Datasets/Norwegian/nno_norne-ud-dev.iob2",
            "test_file": "Datasets/Norwegian/nno_norne-ud-test.iob2",
            "model_name": "xlm-roberta-base",
            "output_dir": "outputs/norwegian_xlmr_baseline",
            "label_report_path": "Baselines/reports/norwegian_labels_xlmr.json",
        }
    )