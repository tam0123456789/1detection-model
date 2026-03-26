from ultralytics import YOLO
from src.models.register import register_uib_modules
from src.utils.metrics import save_metrics_csv
import torch


def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    register_uib_modules()
    model = YOLO("outputs/logs/yolo11n_uib/weights/best.pt")

    results = model.val(
        data="configs/dataset/vrtsd.yaml",
        split="test",
        imgsz=640,
        batch=16 if torch.cuda.is_available() else 4,
        device=device,
        project="outputs/logs",
        name="eval_yolo11n_uib",
        exist_ok=True
    )

    metrics = {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    save_metrics_csv("outputs/metrics/yolo11n_uib_test_metrics.csv", metrics)
    print(metrics)


if __name__ == "__main__":
    main()