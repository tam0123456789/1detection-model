import csv
from pathlib import Path


def save_metrics_csv(save_path, metrics_dict):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics_dict.items():
            writer.writerow([k, v])