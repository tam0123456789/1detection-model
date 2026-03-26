import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.register import register_uib_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="", help="Path to output video")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="cuda device, e.g. 0 or cpu")
    parser.add_argument("--show", action="store_true", help="Show live window")
    return parser.parse_args()


def make_output_path(source_path: str, output_path: str) -> str:
    if output_path:
        return output_path

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    stem = Path(source_path).stem
    return os.path.join(out_dir, f"{stem}_pred.mp4")


def get_color(class_id: int):
    palette = [
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
        (52, 69, 147),
        (100, 115, 255),
        (0, 24, 236),
        (132, 56, 255),
        (82, 0, 133),
        (203, 56, 255),
        (255, 149, 200),
        (255, 55, 199),
    ]
    return palette[class_id % len(palette)]


def draw_label(img, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    th_total = th + baseline + 8

    y_top = max(0, y1 - th_total)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 5, y1 - 6),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_box(img, box, cls_id, conf, class_name):
    x1, y1, x2, y2 = map(int, box)
    color = get_color(cls_id)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = f"{class_name} {conf:.2f}"
    draw_label(img, label, x1, y1, color)


def draw_fps_panel(img, fps, num_det):
    text1 = f"FPS: {fps:.1f}"
    text2 = f"Detections: {num_det}"

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (220, 70), (25, 25, 25), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    cv2.putText(img, text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    args = parse_args()

    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(PROJECT_ROOT, weights_path)

    source_path = args.source
    if not os.path.isabs(source_path):
        source_path = os.path.join(PROJECT_ROOT, source_path)

    output_path = make_output_path(source_path, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "uib" in weights_path.lower():
        register_uib_modules()

    print("WEIGHTS:", weights_path)
    print("SOURCE :", source_path)
    print("OUTPUT :", output_path)

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được video output: {output_path}")

    prev_time = time.time()
    fps_smooth = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )

        result = results[0]
        annotated = frame.copy()

        num_det = 0
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            num_det = len(boxes)

            for box, conf, cls_id in zip(boxes, confs, clss):
                class_name = model.names[int(cls_id)] if hasattr(model, "names") else str(cls_id)
                draw_box(annotated, box, int(cls_id), float(conf), class_name)

        curr_time = time.time()
        instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        if fps_smooth == 0:
            fps_smooth = instant_fps
        else:
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps

        annotated = draw_fps_panel(annotated, fps_smooth, num_det)

        writer.write(annotated)

        if args.show:
            cv2.imshow("YOLO Video Prediction", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames | FPS ~ {fps_smooth:.2f}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    main()