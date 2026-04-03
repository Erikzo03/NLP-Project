import json
import os
from typing import Dict, Set


DATASETS = {
    "english": {
        "train": "Datasets/English/en_ewt-ud-train.iob2",
        "dev": "Datasets/English/en_ewt-ud-dev.iob2",
    },
    "danish": {
        "train": "Datasets/Danish/da_ddt-ud-train.iob2",
        "dev": "Datasets/Danish/da_ddt-ud-dev.iob2",
    },
    "norwegian": {
        "train": "Datasets/Norwegian/nno_norne-ud-train.iob2",
        "dev": "Datasets/Norwegian/nno_norne-ud-dev.iob2",
    },
}


def labels_from_iob2(path: str) -> Set[str]:
    labels: Set[str] = set()
    with open(path, "r", encoding="utf-8") as infile:
        for raw in infile:
            line = raw.strip()
            if not line or (line.startswith("#") and "\t" not in line):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                labels.add(parts[2])
    return labels


def main() -> None:
    language_labels: Dict[str, Set[str]] = {}

    for language, splits in DATASETS.items():
        labels: Set[str] = set()
        for split_path in splits.values():
            labels.update(labels_from_iob2(split_path))
        language_labels[language] = labels

    reference = language_labels["english"]
    report = {
        "labels_by_language": {k: sorted(v) for k, v in language_labels.items()},
        "num_labels_by_language": {k: len(v) for k, v in language_labels.items()},
        "consistent_with_english": {
            k: v == reference for k, v in language_labels.items() if k != "english"
        },
        "differences_vs_english": {
            k: {
                "missing_from_language": sorted(reference - v),
                "extra_in_language": sorted(v - reference),
            }
            for k, v in language_labels.items()
            if k != "english"
        },
    }

    os.makedirs("Baselines/reports", exist_ok=True)
    out_path = "Baselines/reports/label_consistency_report.json"
    with open(out_path, "w", encoding="utf-8") as outfile:
        json.dump(report, outfile, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()