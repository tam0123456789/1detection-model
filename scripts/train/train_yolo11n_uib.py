import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gc
import random
import numpy as np
import torch
from ultralytics import YOLO

from src.models.register import register_uib_modules


def suggest_cfg():
    if not torch.cuda.is_available():
        return {"device": "cpu", "imgsz": 512, "batch": 2, "workers": 0}

    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # RTX 3060 12GB: nên ưu tiên an toàn
    if mem_gb >= 11:
        return {"device": 0, "imgsz": 640, "batch": 4, "workers": 0}
    elif mem_gb >= 8:
        return {"device": 0, "imgsz": 640, "batch": 2, "workers": 0}
    else:
        return {"device": 0, "imgsz": 512, "batch": 2, "workers": 0}


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    gc.collect()

    register_uib_modules()
    cfg = suggest_cfg()

    MODEL_YAML = os.path.join(PROJECT_ROOT, "configs", "models", "yolo11n_uib_backbone.yaml")
    DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "dataset", "vrtsd.yaml")
    PRETRAINED_PT = os.path.join(PROJECT_ROOT, "yolo11n.pt")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("MODEL_YAML   =", MODEL_YAML)
    print("DATA_YAML    =", DATA_YAML)
    print("PRETRAINED_PT=", PRETRAINED_PT)
    print("OUTPUT_DIR   =", OUTPUT_DIR)
    print("TRAIN CFG    =", cfg)

    if not os.path.exists(MODEL_YAML):
        raise FileNotFoundError(f"Không tìm thấy file model yaml: {MODEL_YAML}")
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Không tìm thấy file data yaml: {DATA_YAML}")
    if not os.path.exists(PRETRAINED_PT):
        raise FileNotFoundError(f"Không tìm thấy pretrained weight: {PRETRAINED_PT}")

    model = YOLO(MODEL_YAML)
    model.load(PRETRAINED_PT)

    results = model.train(
        data=DATA_YAML,
        epochs=200,
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg["workers"],

        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        weight_decay=5e-4,
        momentum=0.937,
        warmup_epochs=3.0,
        cos_lr=True,

        # augment giảm nhẹ để bớt tải
        hsv_h=0.012,
        hsv_s=0.50,
        hsv_v=0.30,
        degrees=5.0,
        translate=0.08,
        scale=0.25,
        shear=1.0,
        perspective=0.0002,
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=10,
        fliplr=0.0,
        flipud=0.0,

        multi_scale=False,
        cache=False,          
        amp=True if torch.cuda.is_available() else False,
        val=True,
        save=True,
        plots=True,
        patience=40,
        save_period=25,
        project=OUTPUT_DIR,
        name="yolo11n_uib_safe",
        exist_ok=True,
        seed=seed,
    )

    print("Training completed:", results)


if __name__ == "__main__":
    main()