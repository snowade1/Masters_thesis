import sys
import time
from collections import deque

import cv2
import numpy as np


# Конфігурація
VIDEO_SOURCE = "input_video_d.mp4"
OUTPUT_VIDEO = "result.mp4"

# Пристрій
DEVICE = "auto"
USE_HALF = True  # FP16 на GPU для прискорення

# Модель YOLO
YOLO_MODEL = "yolov8m.pt"
CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.65
YOLO_IMGSZ = 1920

# Класи COCO
TARGET_CLASSES = [0, 1, 2]
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car"}

# ByteTrack - збільшений буфер для стійкості до оклюзії
BYTETRACK_TRACK_THRESH = 0.5
BYTETRACK_MATCH_THRESH = 0.8
BYTETRACK_TRACK_BUFFER = 60

# Гомографія
DEFAULT_SRC_POINTS = np.array([
    [323, 408],
    [757, 390],
    [1840, 950],
    [771, 1078],
], dtype=np.float32)
REAL_WIDTH_M = 18.6
REAL_HEIGHT_M = 63.0

# Трекінг
TRAIL_LENGTH = 60
PREDICTION_FRAMES = 15
MIN_TRACK_LENGTH = 8
SPEED_SMOOTHING = 0.6          # ЕМА для плавного відображення
SPEED_WINDOW_POINTS = 16       # скільки точок брати для регресії швидкості
DIR_MIN_DISP_PX = 3.0
MAX_LOST_SECONDS = 3

# Параметри фільтрації шуму швидкості
ANCHOR_SMOOTH_ALPHA = 0.5          # EMA-згладжування анкера (до 1), чим більше, тим швидша реакція
STATIC_NOISE_BBOX_FRAC = 0.05      # зміщення < цього % діагоналі bbox = "стоїть"
SPEED_MIN_KMH = 2.0                # нижче цього - вважаємо шумом
SPEED_ZERO_SNAP_KMH = 0.3          # після ЕМА: якщо менше - snap до 0

# Оклюзія
OCCLUSION_MAX_FRAMES = 45
OCCLUSION_DASH_LEN = 10
OCCLUSION_GAP_LEN = 6

# Оптичний потік
OF_MAX_CORNERS = 50
OF_QUALITY_LEVEL = 0.3
OF_MIN_DISTANCE = 7
OF_WIN_SIZE = (21, 21)
OF_MAX_LEVEL = 3

# Графіка
SHOW_WINDOW = True
SHOW_OPTICAL_FLOW = False
SHOW_PREDICTION = True
SHOW_TRAIL = True
SHOW_CALIB_AREA = True
SHOW_OCCLUSION = True
BBOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Використовуємо CPU або GPU
def resolve_device():
    if DEVICE == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        sys.exit("[ERROR] Встановіть PyTorch")

    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            sys.exit("[ERROR] CUDA недоступна")
        return "cuda:0"

    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    print("[Device] CPU")
    return "cpu"


class Detector:
    def __init__(self):
        from ultralytics import YOLO
        self.device = resolve_device()
        self.half = USE_HALF and self.device.startswith("cuda")
        self.model = YOLO(YOLO_MODEL)
        self.model.to(self.device)

        if self.device.startswith("cuda"):
            dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
            for _ in range(3):
                self.model.predict(dummy, imgsz=YOLO_IMGSZ, device=self.device,
                                   half=self.half, verbose=False)
        print(f"[Detector] device={self.device}, half={self.half}")

    def detect(self, frame):
        res = self.model.predict(
            frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
            classes=TARGET_CLASSES, imgsz=YOLO_IMGSZ,
            device=self.device, half=self.half, verbose=False,
        )
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        b = res[0].boxes
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        return np.hstack([xyxy, conf[:, None], cls[:, None]]).astype(np.float32)


# Трекер ByteTrack
class Tracker:
    def __init__(self, fps=30):
        import supervision as sv
        self.sv = sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=BYTETRACK_TRACK_THRESH,
            minimum_matching_threshold=BYTETRACK_MATCH_THRESH,
            lost_track_buffer=BYTETRACK_TRACK_BUFFER,
            frame_rate=int(fps),
        )

    def update(self, dets):
        sv = self.sv
        if dets is None or len(dets) == 0:
            sv_det = sv.Detections.empty()
        else:
            sv_det = sv.Detections(
                xyxy=dets[:, :4].astype(np.float32),
                confidence=dets[:, 4].astype(np.float32),
                class_id=dets[:, 5].astype(int),
            )
        try:
            tr = self.tracker.update_with_detections(sv_det)
        except Exception as e:
            print(f"[Tracker] {e}")
            return []

        if tr.tracker_id is None or len(tr) == 0:
            return []

        out = []
        for i in range(len(tr)):
            tid = int(tr.tracker_id[i])
            if tid < 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in tr.xyxy[i]]
            out.append({
                "track_id": tid,
                "bbox": [x1, y1, x2, y2],
                "class_id": int(tr.class_id[i]) if tr.class_id is not None else 2,
            })
        return out


# Калібрування
class Calibration:
    def __init__(self, src=None):
        self.src = src if src is not None else DEFAULT_SRC_POINTS.copy()
        self.dst = np.array([
            [0, 0],
            [REAL_WIDTH_M, 0],
            [REAL_WIDTH_M, REAL_HEIGHT_M],
            [0, REAL_HEIGHT_M],
        ], dtype=np.float32)
        self._update()

    def _update(self):
        self.H = cv2.getPerspectiveTransform(self.src, self.dst)
        self.H_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def pixel_to_meters(self, x, y):
        p = np.array([[[x, y]]], dtype=np.float32)
        w = cv2.perspectiveTransform(p, self.H)
        return float(w[0, 0, 0]), float(w[0, 0, 1])

    def meters_to_pixel(self, x, y):
        p = np.array([[[x, y]]], dtype=np.float32)
        w = cv2.perspectiveTransform(p, self.H_inv)
        return float(w[0, 0, 0]), float(w[0, 0, 1])

    def draw(self, frame):
        pts = self.src.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        for i, p in enumerate(self.src.astype(int)):
            cv2.circle(frame, tuple(p), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"P{i}", (p[0] + 8, p[1] - 8),
                        FONT, 0.5, (0, 255, 255), 1)


# Оптичний потік
class OpticalFlow:
    def __init__(self):
        self.prev_gray = None
        self.lk_params = dict(
            winSize=OF_WIN_SIZE, maxLevel=OF_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feat_params = dict(
            maxCorners=OF_MAX_CORNERS, qualityLevel=OF_QUALITY_LEVEL,
            minDistance=OF_MIN_DISTANCE, blockSize=7)

    def compute_in_bbox(self, frame, bbox):
        "Середній вектор руху (dx, dy) у пікселях в bbox."
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            return 0.0, 0.0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0, 0.0

        roi = self.prev_gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(roi, mask=None, **self.feat_params)
        if pts is None or len(pts) < 3:
            return 0.0, 0.0

        pts_g = pts + np.array([[x1, y1]], dtype=np.float32)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts_g, None, **self.lk_params)
        if new_pts is None:
            return 0.0, 0.0

        status = status.reshape(-1).astype(bool)
        if status.sum() < 3:
            return 0.0, 0.0

        old = pts_g[status].reshape(-1, 2)
        new = new_pts[status].reshape(-1, 2)
        disp = new - old

        mag = np.linalg.norm(disp, axis=1)
        lo, hi = np.percentile(mag, 10), np.percentile(mag, 90)
        keep = (mag >= lo) & (mag <= hi)
        if keep.sum() < 3:
            return float(np.mean(disp[:, 0])), float(np.mean(disp[:, 1]))
        return float(np.mean(disp[keep, 0])), float(np.mean(disp[keep, 1]))

    def update_prev(self, frame):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def draw_vector(frame, bbox, dx, dy, scale=5.0):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if abs(dx) + abs(dy) < 0.5:
            return
        ex, ey = int(cx + dx * scale), int(cy + dy * scale)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)


# Історія трека та аналіз
class TrackHistory:
    def __init__(self, track_id, class_id):
        self.track_id = track_id
        self.class_id = class_id
        self.centroids_px = deque(maxlen=TRAIL_LENGTH)
        self.centroids_m = deque(maxlen=TRAIL_LENGTH)
        self.timestamps = deque(maxlen=TRAIL_LENGTH)
        self.bbox_sizes = deque(maxlen=TRAIL_LENGTH)
        self.speed_kmh = 0.0
        self.direction = "-"
        self.lost_frames = 0
        self.last_bbox = None
        # Згладжений анкер (EMA у пікселях) - знижує шум bbox
        self._smoothed_px = None

    def add(self, raw_cx_px, raw_cy_px, t_sec, bbox_wh, bbox, calib):
        
        # Приймає "сирий" анкер у пікселях, робить EMA-згладжування, конвертує в метри і додає в історію.
        
        if self._smoothed_px is None:
            sx, sy = raw_cx_px, raw_cy_px
        else:
            a = ANCHOR_SMOOTH_ALPHA
            sx = a * raw_cx_px + (1 - a) * self._smoothed_px[0]
            sy = a * raw_cy_px + (1 - a) * self._smoothed_px[1]
        self._smoothed_px = (sx, sy)

        cx_m, cy_m = calib.pixel_to_meters(sx, sy)

        self.centroids_px.append((sx, sy))
        self.centroids_m.append((cx_m, cy_m))
        self.timestamps.append(t_sec)
        self.bbox_sizes.append(bbox_wh)
        self.last_bbox = bbox
        self.lost_frames = 0


class MotionAnalyzer:
    def __init__(self, calib, fps):
        self.calib = calib
        self.fps = fps
        self.histories = {}

    @staticmethod
    def _anchor_px(bbox, class_id):
        """
        Анкер = низ bbox для всіх класів, бо гомографія калібрована
        саме на площину дороги. Центроїд був би в повітрі і давав би
        неправильні метри після pixel_to_meters().
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = y2
        return cx, cy

    @staticmethod
    def estimate_direction(h):
        if len(h.centroids_px) < 2:
            return "-"
        x0, y0 = h.centroids_px[0]
        x1, y1 = h.centroids_px[-1]
        dx, dy = x1 - x0, y1 - y0
        if np.hypot(dx, dy) < DIR_MIN_DISP_PX:
            return "stopped"
        ang = np.degrees(np.arctan2(-dy, dx))
        if ang < 0:
            ang += 360
        bins = [(22.5, "E"), (67.5, "NE"), (112.5, "N"), (157.5, "NW"),
                (202.5, "W"), (247.5, "SW"), (292.5, "S"), (337.5, "SE")]
        for up, name in bins:
            if ang < up:
                return name
        return "E"

    def estimate_speed_kmh(self, h):
        """
        Стабільна швидкість через:
          1) Перевірку "чи ворушиться об'єкт у пікселях" (захист дальніх
             об'єктів від шуму перспективи).
          2) Лінійну регресію x(t), y(t) у метрах - усереднює напрямок,
             стабільніша за різницю двох медіан.
          3) Поріг SPEED_MIN_KMH обнуляє мілкий шум.
          4) EMA для плавного відображення + snap до 0 нижче порогу.
        """
        n = len(h.centroids_m)
        if n < MIN_TRACK_LENGTH:
            return 0.0

        k = min(SPEED_WINDOW_POINTS, n)
        pts_px = np.array(list(h.centroids_px)[-k:])
        pts_m = np.array(list(h.centroids_m)[-k:])
        ts = np.array(list(h.timestamps)[-k:])

        if ts[-1] - ts[0] < 1e-3:
            return h.speed_kmh

        # 1) Чи рухається у пікселях відносно розміру bbox
        # Пікселі не залежать від перспективного спотворення, тому
        # шум для дальніх об'єктів тут не перебільшується.
        px_disp = float(np.linalg.norm(pts_px[-1] - pts_px[0]))
        bw, bh = h.bbox_sizes[-1]
        bbox_diag = float(np.hypot(bw, bh))
        if bbox_diag > 1e-3 and px_disp < STATIC_NOISE_BBOX_FRAC * bbox_diag:
            # Згасання поточної швидкості + snap до 0
            h.speed_kmh *= SPEED_SMOOTHING
            if h.speed_kmh < SPEED_ZERO_SNAP_KMH:
                h.speed_kmh = 0.0
            return h.speed_kmh

        # 2) Лінійна регресія по метрам
        # Підганяємо x(t) = ax*t + bx, y(t) = ay*t + by
        # Швидкість = sqrt(ax^2 + ay^2) м/с
        ax = np.polyfit(ts, pts_m[:, 0], 1)[0]
        ay = np.polyfit(ts, pts_m[:, 1], 1)[0]
        v_ms = float(np.hypot(ax, ay))
        v_kmh = v_ms * 3.6

        # 3) Поріг шуму
        if v_kmh < SPEED_MIN_KMH:
            v_kmh = 0.0

        # 4) EMA для плавності
        h.speed_kmh = SPEED_SMOOTHING * h.speed_kmh + (1 - SPEED_SMOOTHING) * v_kmh
        if h.speed_kmh < SPEED_ZERO_SNAP_KMH:
            h.speed_kmh = 0.0
        return h.speed_kmh

    def predict_future(self, h, frames_ahead=PREDICTION_FRAMES):
        if len(h.centroids_m) < MIN_TRACK_LENGTH:
            return None
        k = min(10, len(h.centroids_m))
        pts = np.array(list(h.centroids_m)[-k:])
        ts = np.array(list(h.timestamps)[-k:])
        if ts[-1] - ts[0] < 1e-3:
            return None
        t0 = ts[0]
        tr = ts - t0
        ax, bx = np.polyfit(tr, pts[:, 0], 1)
        ay, by = np.polyfit(tr, pts[:, 1], 1)
        t_fut = (ts[-1] - t0) + frames_ahead / self.fps
        xm, ym = ax * t_fut + bx, ay * t_fut + by
        return self.calib.meters_to_pixel(float(xm), float(ym))

    def predict_occluded_bbox(self, h, frames_ahead):
        if len(h.centroids_m) < MIN_TRACK_LENGTH or h.last_bbox is None:
            return None
        k = min(10, len(h.centroids_m))
        pts = np.array(list(h.centroids_m)[-k:])
        ts = np.array(list(h.timestamps)[-k:])
        if ts[-1] - ts[0] < 1e-3:
            return None

        t0 = ts[0]
        tr = ts - t0
        ax, bx = np.polyfit(tr, pts[:, 0], 1)
        ay, by = np.polyfit(tr, pts[:, 1], 1)
        t_fut = (ts[-1] - t0) + frames_ahead / self.fps
        xm, ym = ax * t_fut + bx, ay * t_fut + by

        cx_px, cy_px = self.calib.meters_to_pixel(float(xm), float(ym))

        sizes = np.array(list(h.bbox_sizes)[-k:])
        avg_w = float(np.mean(sizes[:, 0]))
        avg_h = float(np.mean(sizes[:, 1]))

        x1 = cx_px - avg_w / 2
        x2 = cx_px + avg_w / 2
        y2 = cy_px
        y1 = cy_px - avg_h
        return (x1, y1, x2, y2)

    def update(self, tracks, frame_idx):
        t_sec = frame_idx / self.fps
        active = set()

        for tr in tracks:
            tid = tr["track_id"]
            x1, y1, x2, y2 = tr["bbox"]
            cx_px, cy_px = self._anchor_px(tr["bbox"], tr["class_id"])
            bbox_wh = (x2 - x1, y2 - y1)

            if tid not in self.histories:
                self.histories[tid] = TrackHistory(tid, tr["class_id"])
            h = self.histories[tid]
            # Згладжування і конвертація відбуваються всередині add()
            h.add(cx_px, cy_px, t_sec, bbox_wh, tr["bbox"], self.calib)
            h.direction = self.estimate_direction(h)
            self.estimate_speed_kmh(h)
            active.add(tid)

        max_lost = int(MAX_LOST_SECONDS * self.fps)
        for tid in list(self.histories.keys()):
            if tid not in active:
                self.histories[tid].lost_frames += 1
                if self.histories[tid].lost_frames > max_lost:
                    del self.histories[tid]

        return active

    def get_occluded_tracks(self, active_ids):
        occluded = []
        for tid, h in self.histories.items():
            if tid in active_ids:
                continue
            if h.lost_frames == 0 or h.lost_frames > OCCLUSION_MAX_FRAMES:
                continue
            if len(h.centroids_m) < MIN_TRACK_LENGTH:
                continue
            pred_bbox = self.predict_occluded_bbox(h, h.lost_frames)
            if pred_bbox is not None:
                occluded.append({
                    "track_id": tid,
                    "class_id": h.class_id,
                    "bbox": pred_bbox,
                    "history": h,
                })
        return occluded

    def get(self, track_id):
        return self.histories.get(track_id)


# Візуалізація
_color_cache = {}


def color_for(track_id, class_id):
    if track_id in _color_cache:
        return _color_cache[track_id]
    if class_id == 0:
        base = np.array([255, 100, 100])
    elif class_id == 1:
        base = np.array([100, 255, 100])
    else:
        base = np.array([100, 150, 255])
    rng = np.random.default_rng(track_id * 17 + 3)
    jitter = rng.integers(-40, 40, size=3)
    c = np.clip(base + jitter, 50, 240).astype(int)
    col = (int(c[0]), int(c[1]), int(c[2]))
    _color_cache[track_id] = col
    return col


def draw_dashed_rect(frame, pt1, pt2, color, thickness=2,
                     dash_len=OCCLUSION_DASH_LEN, gap_len=OCCLUSION_GAP_LEN):
    x1, y1 = pt1
    x2, y2 = pt2

    def dashed_line(p1, p2):
        dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist == 0:
            return
        step = dash_len + gap_len
        for i in range(0, dist, step):
            ratio_a = i / dist
            ratio_b = min((i + dash_len) / dist, 1.0)
            ax = int(p1[0] + (p2[0] - p1[0]) * ratio_a)
            ay = int(p1[1] + (p2[1] - p1[1]) * ratio_a)
            bx = int(p1[0] + (p2[0] - p1[0]) * ratio_b)
            by = int(p1[1] + (p2[1] - p1[1]) * ratio_b)
            cv2.line(frame, (ax, ay), (bx, by), color, thickness)

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def draw_track(frame, track, history, prediction_pt=None):
    x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
    tid = track["track_id"]
    cid = track["class_id"]
    col = color_for(tid, cid)
    name = CLASS_NAMES.get(cid, "obj")

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, BBOX_THICKNESS)

    label = f"ID:{tid} {name}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    if history:
        # Якщо швидкість = 0, показуємо "stopped" замість цифри
        if history.speed_kmh < 0.1:
            lbl = f"stopped"
        else:
            lbl = f"{history.speed_kmh:5.1f} km/h  {history.direction}"
        cv2.putText(frame, lbl, (x1, y2 + 18), FONT, 0.55, col, 2, cv2.LINE_AA)

        if SHOW_TRAIL and len(history.centroids_px) > 1:
            pts = np.array(history.centroids_px, dtype=np.int32)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thick = max(1, int(3 * alpha))
                cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), col, thick)

    if prediction_pt is not None and SHOW_PREDICTION and history and history.speed_kmh > 1.0:
        # Не малюємо прогноз для нерухомих - там лише шум
        px, py = int(prediction_pt[0]), int(prediction_pt[1])
        cx, cy = (x1 + x2) // 2, y2
        cv2.arrowedLine(frame, (cx, cy), (px, py), (0, 0, 255), 2, tipLength=0.25)
        cv2.circle(frame, (px, py), 6, (0, 0, 255), 2)
        cv2.putText(frame, "pred", (px + 8, py), FONT, 0.45, (0, 0, 255), 1, cv2.LINE_AA)


def draw_occluded(frame, occ):
    x1, y1, x2, y2 = [int(v) for v in occ["bbox"]]
    tid = occ["track_id"]
    cid = occ["class_id"]
    hist = occ["history"]
    col = color_for(tid, cid)
    name = CLASS_NAMES.get(cid, "obj")

    draw_dashed_rect(frame, (x1, y1), (x2, y2), col, BBOX_THICKNESS)

    label = f"ID:{tid} {name} (occluded)"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    if SHOW_TRAIL and len(hist.centroids_px) > 1:
        pts = np.array(hist.centroids_px, dtype=np.int32)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thick = max(1, int(3 * alpha))
            cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), col, thick)


def draw_hud(frame, frame_idx, fps_real, n_tracks, n_occluded):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), (0, 0, 0), -1)
    text = (f"ByteTrack | frame {frame_idx} | FPS {fps_real:4.1f} | "
            f"tracks: {n_tracks} | occluded: {n_occluded}")
    cv2.putText(frame, text, (10, 28), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


### Головна петля

def main():
    src = 0 if str(VIDEO_SOURCE) == "0" else VIDEO_SOURCE
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Не вдалось відкрити: {VIDEO_SOURCE}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[Main] {W}x{H} @ {FPS:.1f} FPS")

    detector = Detector()
    tracker = Tracker(fps=int(round(FPS)))
    calib = Calibration()
    motion = MotionAnalyzer(calib, FPS)
    flow = OpticalFlow()

    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    frame_idx = 0
    t_start = time.time()

    print("[Main] Старт.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            dets = detector.detect(frame)
            tracks = tracker.update(dets)

            flow_map = {}
            if SHOW_OPTICAL_FLOW:
                for tr in tracks:
                    dxy = flow.compute_in_bbox(frame, tr["bbox"])
                    flow_map[tr["track_id"]] = dxy
                    flow.draw_vector(frame, tr["bbox"], *dxy)
                flow.update_prev(frame)

            active_ids = motion.update(tracks, frame_idx)
            occluded = motion.get_occluded_tracks(active_ids) if SHOW_OCCLUSION else []

            for tr in tracks:
                h_rec = motion.get(tr["track_id"])
                pred = motion.predict_future(h_rec) if h_rec else None
                draw_track(frame, tr, h_rec, pred)

            for occ in occluded:
                draw_occluded(frame, occ)

            if SHOW_CALIB_AREA:
                calib.draw(frame)

            elapsed = time.time() - t_start
            real_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            draw_hud(frame, frame_idx, real_fps, len(tracks), len(occluded))

            writer.write(frame)
            frame_idx += 1

            if SHOW_WINDOW:
                cv2.imshow("ByteTrack - Speed Tracker", frame)
                cv2.waitKey(1)
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[Main] Готово. Кадрів: {frame_idx}")


if __name__ == "__main__":
    main()