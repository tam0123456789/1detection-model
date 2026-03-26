import os
import re
import time
import inspect
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO
from twilio.rest import Client
import torch


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "logs" / "yolo11n_uib" / "weights" / "best.pt"
ALT_WEIGHTS = PROJECT_ROOT / "outputs" / "logs" / "yolo11n_uib_safe" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Traffic Sign Detection Demo", layout="wide")
st.title("Traffic Sign Detection Demo")
st.write("Demo nhận diện biển báo giao thông bằng YOLO11n / YOLO11n-UIB")


# =========================
# UIB MODULES
# =========================
class ConvBNAct(torch.nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        p = k // 2
        self.conv = torch.nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = torch.nn.BatchNorm2d(c2)
        self.act = torch.nn.SiLU(inplace=True) if act else torch.nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(torch.nn.Module):
    def __init__(self, c, r=0.25):
        super().__init__()
        hidden = max(8, int(c * r))
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(c, hidden, 1, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, c, 1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class UIB(torch.nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=6.0):
        super().__init__()
        hidden = max(16, int(round(c1 * e)))
        self.use_res = (s == 1 and c1 == c2)
        self.block = torch.nn.Sequential(
            ConvBNAct(c1, hidden, k=1, s=1, act=True),
            ConvBNAct(hidden, hidden, k=k, s=s, g=hidden, act=True),
            SqueezeExcite(hidden, r=0.25),
            ConvBNAct(hidden, c2, k=1, s=1, act=False),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class UIBDown(torch.nn.Module):
    def __init__(self, c1, c2, k=3, s=2, e=6.0):
        super().__init__()
        hidden = max(16, int(round(c1 * e)))
        self.block = torch.nn.Sequential(
            ConvBNAct(c1, hidden, k=1, s=1, act=True),
            ConvBNAct(hidden, hidden, k=k, s=s, g=hidden, act=True),
            SqueezeExcite(hidden, r=0.25),
            ConvBNAct(hidden, c2, k=1, s=1, act=False),
        )

    def forward(self, x):
        return self.block(x)


def register_uib_modules():
    import ultralytics.nn.tasks as tasks

    tasks.UIB = UIB
    tasks.UIBDown = UIBDown

    src = inspect.getsource(tasks.parse_model)
    if "base_modules = set(base_modules) | {UIB, UIBDown}" in src:
        tasks.parse_model.__globals__["UIB"] = UIB
        tasks.parse_model.__globals__["UIBDown"] = UIBDown
        tasks.__dict__["UIB"] = UIB
        tasks.__dict__["UIBDown"] = UIBDown
        return

    lines = src.splitlines()
    out = []
    inserted = False
    in_base = False
    balance = 0
    indent_after = ""

    def delta_balance(line: str):
        return (line.count("{") - line.count("}")) + (line.count("(") - line.count(")"))

    for line in lines:
        out.append(line)

        if (not inserted) and (not in_base) and re.match(r"^\s*base_modules\s*=", line):
            in_base = True
            balance = delta_balance(line)
            indent_after = re.match(r"^(\s*)", line).group(1)
            if balance == 0:
                out.append(indent_after + "base_modules = set(base_modules) | {UIB, UIBDown}")
                out.append(indent_after + "base_modules = frozenset(base_modules)")
                inserted = True
                in_base = False
        elif in_base:
            balance += delta_balance(line)
            if balance == 0:
                out.append(indent_after + "base_modules = set(base_modules) | {UIB, UIBDown}")
                out.append(indent_after + "base_modules = frozenset(base_modules)")
                inserted = True
                in_base = False

    if not inserted:
        raise RuntimeError("Không patch được parse_model()")

    patched = "\n".join(out)
    g = dict(tasks.__dict__)
    g["UIB"] = UIB
    g["UIBDown"] = UIBDown
    exec(patched, g)
    tasks.parse_model = g["parse_model"]

    tasks.parse_model.__globals__["UIB"] = UIB
    tasks.parse_model.__globals__["UIBDown"] = UIBDown
    tasks.__dict__["UIB"] = UIB
    tasks.__dict__["UIBDown"] = UIBDown


# =========================
# HELPERS
# =========================
def get_default_weights() -> str:
    if DEFAULT_WEIGHTS.exists():
        return str(DEFAULT_WEIGHTS)
    if ALT_WEIGHTS.exists():
        return str(ALT_WEIGHTS)
    return ""


def get_color(class_id: int):
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    ]
    return palette[class_id % len(palette)]


def draw_label(img, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y_top = max(0, y1 - th - baseline - 8)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, text, (x1 + 4, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_fps_panel(img, fps, num_det):
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (230, 75), (25, 25, 25), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Detections: {num_det}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    if "uib" in weights_path.lower():
        register_uib_modules()
    return YOLO(weights_path)


@st.cache_resource(show_spinner=False)
def get_twilio_ice_servers():
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        return None

    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers


def detect_on_frame(model, frame_bgr, imgsz=512, conf=0.25, iou=0.45, device="0"):
    results = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    result = results[0]
    annotated = frame_bgr.copy()
    num_det = 0

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        num_det = len(boxes)

        for box, score, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            color = get_color(int(cls_id))
            name = model.names[int(cls_id)] if hasattr(model, "names") else str(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            draw_label(annotated, f"{name} {score:.2f}", x1, y1, color)

    return annotated, num_det


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Cấu hình")
    mode = st.selectbox("Chọn chế độ", ["Webcam", "Image", "Video"])
    weights_path = st.text_input("Đường dẫn weights", value=get_default_weights())
    imgsz = st.selectbox("Image size", [416, 512, 640], index=1)
    conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU", 0.10, 0.95, 0.45, 0.05)
    device = st.selectbox("Device", ["0", "cpu"], index=0)

if not weights_path or not os.path.exists(weights_path):
    st.error("Không tìm thấy file weights.")
    st.stop()

model = load_model(weights_path)


# =========================
# IMAGE
# =========================
if mode == "Image":
    uploaded_image = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        t0 = time.time()
        annotated, num_det = detect_on_frame(model, image_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device)
        elapsed = time.time() - t0

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Kết quả")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.write(f"Số detections: **{num_det}**")
        st.write(f"Thời gian xử lý: **{elapsed:.3f}s**")


# =========================
# VIDEO
# =========================
elif mode == "Video":
    uploaded_video = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        input_path = OUTPUT_DIR / "temp_input_video.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(str(input_path))

        if st.button("Chạy nhận diện video"):
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                st.error("Không mở được video.")
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            if fps_in <= 0:
                fps_in = 25.0

            out_path = OUTPUT_DIR / "video_result.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_in,
                (width, height),
            )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            preview = st.empty()

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                annotated, num_det = detect_on_frame(model, frame, imgsz=imgsz, conf=conf, iou=iou, device=device)
                fps_cur = 1.0 / max(time.time() - t0, 1e-6)
                annotated = draw_fps_panel(annotated, fps_cur, num_det)
                writer.write(annotated)

                frame_idx += 1
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))
                if frame_idx % 10 == 0:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            cap.release()
            writer.release()
            st.success("Xong.")
            st.video(str(out_path))


# =========================
# WEBCAM
# =========================
else:
    st.subheader("Webcam realtime")

    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if ice_servers is None:
        st.warning("Chưa có TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN trên server.")
        st.stop()

    class SignVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_time = time.time()
            self.fps_smooth = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, num_det = detect_on_frame(model, img, imgsz=imgsz, conf=conf, iou=iou, device=device)

            now = time.time()
            fps = 1.0 / max(now - self.last_time, 1e-6)
            self.last_time = now
            self.fps_smooth = fps if self.fps_smooth == 0 else 0.9 * self.fps_smooth + 0.1 * fps

            annotated = draw_fps_panel(annotated, self.fps_smooth, num_det)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="traffic-sign-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignVideoProcessor,
        rtc_configuration={"iceServers": ice_servers},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )