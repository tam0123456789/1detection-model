import argparse
from ultralytics import YOLO
from src.models.register import register_uib_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Image / folder / video path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="predict")
    return parser.parse_args()


def main():
    args = parse_args()

    if "uib" in args.weights.lower():
        register_uib_modules()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project="outputs/predictions",
        name=args.name,
        exist_ok=True
    )


if __name__ == "__main__":
    main()