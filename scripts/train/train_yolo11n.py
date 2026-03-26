from ultralytics import YOLO
import torch


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="configs/dataset/vrtsd.yaml",
        epochs=100,
        imgsz=640,
        batch=16 if torch.cuda.is_available() else 4,
        device=device,
        workers=4 if torch.cuda.is_available() else 2,
        project="outputs/logs",
        name="yolo11n_baseline",
        pretrained=True,
        exist_ok=True
    )

    print("Training completed:", results)


if __name__ == "__main__":
    main()