"""
High-Precision Asynchronous Facial Expression & 3D Head Pose Worker.
Dual-Stage Pipeline (pure OpenCV + ONNX Runtime, ~40-60 FPS on CPU):
  Stage 1: SCRFD ONNX detects the face bounding box robustly (even in low light).
  Stage 2: MediaPipe 468-point 3D FaceMesh ONNX extracts exact lip, eyebrow,
           eye (EAR), nose, chin, and cheek contours.
  Stage 3: OpenCV solvePnP computes true 3D Roll/Yaw/Pitch + 3D nose axes,
           filtered with velocity-adaptive One-Euro Filters.

Supports:
  - Smile           -> FLAT car
  - Open Mouth      -> NARROW car
  - Eyebrow Raise   -> Jump trigger
  - Asymmetric Wink -> Energy Shield + Nitro Boost trigger
  - 3D Head Pose    -> Lateral Steering
  - 3-Step Personalized Calibration Wizard (NEUTRAL -> SMILE -> OPEN)

Thread-safe: performs NO Pygame calls in the background thread (returns raw RGB bytes).
"""

import threading
import time
import os
import platform
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from game.config import PIP_WIDTH, PIP_HEIGHT
from game.filters import OneEuroFilter
from game.pose_3d import HeadPose3D


# MediaPipe 468-point landmark contour indices
OUTER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_BROW_IDX = [70, 63, 105, 66, 107]
RIGHT_BROW_IDX = [336, 296, 334, 293, 300]


class SCRFDDetector:
    """Fast Stage-1 Face Detector using SCRFD ONNX via OpenCV DNN."""
    def __init__(self, onnx_path, conf_thresh=0.35, nms_thresh=0.45):
        self.inp_w = 320
        self.inp_h = 320
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.net = cv2.dnn.readNet(onnx_path)
        self.feat_strides = [8, 16, 32]
        self.num_anchors = 2

    def resize_image(self, img):
        h, w = img.shape[:2]
        scale = min(self.inp_h / h, self.inp_w / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        padded = cv2.copyMakeBorder(
            resized, 0, self.inp_h - new_h, 0, self.inp_w - new_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded, scale

    def distance2bbox(self, points, distance):
        distance = distance.reshape(-1, 4)
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance):
        distance = distance.reshape(-1, 10)
        preds = []
        for i in range(0, 10, 2):
            preds.append(points[:, 0] + distance[:, i])
            preds.append(points[:, 1] + distance[:, i + 1])
        return np.stack(preds, axis=-1)

    def detect(self, img):
        padded, scale = self.resize_image(img)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 128.0, (self.inp_w, self.inp_h),
            (127.5, 127.5, 127.5), swapRB=True
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        by_shape = {}
        for out in outs:
            arr = out[0] if out.ndim == 3 else out
            by_shape[(arr.shape[0], arr.shape[-1])] = arr

        scores_list, bboxes_list, kpss_list = [], [], []

        for stride in self.feat_strides:
            h = blob.shape[2] // stride
            w = blob.shape[3] // stride
            expected_n = h * w * self.num_anchors

            scores = by_shape.get((expected_n, 1))
            bbox_raw = by_shape.get((expected_n, 4))
            kps_raw = by_shape.get((expected_n, 10))
            if scores is None or bbox_raw is None or kps_raw is None:
                continue

            scores = scores.reshape(-1)
            pos_inds = np.where(scores >= self.conf_thresh)[0]
            if len(pos_inds) == 0:
                continue

            anchor_centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            anchor_centers = np.repeat((anchor_centers * stride).reshape((-1, 2)), self.num_anchors, axis=0)

            bboxes = self.distance2bbox(anchor_centers[pos_inds], bbox_raw[pos_inds] * stride)
            kpss = self.distance2kps(anchor_centers[pos_inds], kps_raw[pos_inds] * stride).reshape((-1, 5, 2))

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        if not scores_list:
            return np.array([]), np.array([])

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)

        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        bboxes /= scale
        kpss /= scale

        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh)
        if len(indices) == 0:
            return np.array([]), np.array([])

        indices = np.array(indices).flatten()
        return bboxes[indices], kpss[indices]


class FaceMesh468:
    """Stage-2 468-Point 3D Facial Landmark Extractor using ONNX Runtime (~2ms/frame)."""
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.inp_size = 192

    def predict(self, frame_bgr, bbox):
        fh, fw = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        cx = x + bw * 0.5
        cy = y + bh * 0.54  # Slight downward shift keeps dropped jaw/chin inside ROI
        side = max(bw, bh) * 1.55

        x1 = int(cx - side * 0.5)
        y1 = int(cy - side * 0.5)
        x2 = int(cx + side * 0.5)
        y2 = int(cy + side * 0.5)
        crop_side = max(x2 - x1, 1)

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - fw)
        pad_b = max(0, y2 - fh)

        roi = frame_bgr[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
        if roi.size == 0:
            return None

        if pad_l or pad_t or pad_r or pad_b:
            roi = cv2.copyMakeBorder(roi, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        roi_resized = cv2.resize(roi, (self.inp_size, self.inp_size))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp = (roi_rgb / 127.5) - 1.0  # MediaPipe [-1, 1] normalization

        out = self.sess.run(None, {self.input_name: inp[None, ...]})[0].reshape(468, 3)
        pts = out[:, :2].copy()

        scale_back = crop_side / float(self.inp_size)
        pts[:, 0] = pts[:, 0] * scale_back + x1
        pts[:, 1] = pts[:, 1] * scale_back + y1
        return pts


def _eye_aspect_ratio(pts, eye_idx):
    """Computes standard Eye Aspect Ratio (EAR) from 6 eye contour points."""
    p = pts[eye_idx]
    v1 = float(np.linalg.norm(p[1] - p[5]))
    v2 = float(np.linalg.norm(p[2] - p[4]))
    h = max(float(np.linalg.norm(p[0] - p[3])), 1.0)
    return (v1 + v2) / (2.0 * h)


class ExpressionWorker:
    """
    Background worker thread running SCRFD + 468-point FaceMesh + 3D solvePnP + One-Euro Filter.
    Feeds thread-safe pose, morph shape, eyebrow-jump, and wink-shield signals to the game.
    """
    def __init__(
        self,
        model_path="Face_Pose/models/scrfd_500m_kps.onnx",
        mesh_path="Face_Pose/models/face_mesh.onnx",
        camera_id=0
    ):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.camera_id = camera_id
        self.model_path = model_path
        self.mesh_path = mesh_path

        self.detector = None
        self.mesh = None
        self.pose_3d = HeadPose3D()
        self.cap = None
        self.camera_available = False

        # Outputs read by the game
        self.steer_x = 0.0
        self.current_shape = "NORMAL"
        self.face_detected = False
        self.smile_intensity = 0.0
        self.open_intensity = 0.0
        self.brow_intensity = 0.0
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.fps = 0.0

        # Edge-triggered action flags (consumed by game loop)
        self._jump_pending = False
        self._wink_pending = False
        self._last_jump_time = 0.0
        self._last_wink_time = 0.0
        self._wink_hold_frames = 0
        self._brow_armed = True

        # 3-Step Calibration Wizard state: None | "NEUTRAL" | "SMILE" | "OPEN"
        self.calibration_stage = "NEUTRAL"
        self.is_calibrating = True
        self._cal_samples = {"NEUTRAL": [], "SMILE": [], "OPEN": []}
        self.calibration_target_count = 25

        # Neutral baselines (realistic 468-point defaults)
        self.base_mouth_w_ratio = 0.62
        self.base_corner_lift = 0.015
        self.base_inner_h = 0.018
        self.base_mar = 0.030
        self.base_outer_h = 0.220
        self.base_jaw_h = 0.940
        self.base_brow_h = 0.280
        self.base_left_ear = 0.28
        self.base_right_ear = 0.28
        self.base_yaw = 0.0
        self.base_roll = 0.0
        self.base_yaw_3d = 0.0

        # Personalized dynamic spans (refined by Calibration Steps 2 & 3!)
        self.smile_width_span = 0.070
        self.smile_lift_span = 0.038
        self.open_inner_span = 0.070
        self.open_mar_span = 0.125
        self.open_outer_span = 0.095
        self.open_jaw_span = 0.105

        # Velocity-adaptive One-Euro Filters (zero jitter when still, snappy on quick moves!)
        self._f_steer = OneEuroFilter(min_cutoff=1.4, beta=0.18, d_cutoff=1.0)
        self._f_smile = OneEuroFilter(min_cutoff=1.8, beta=0.25, d_cutoff=1.0)
        self._f_open = OneEuroFilter(min_cutoff=1.8, beta=0.25, d_cutoff=1.0)
        self._f_brow = OneEuroFilter(min_cutoff=1.6, beta=0.20, d_cutoff=1.0)
        self._f_yaw = OneEuroFilter(min_cutoff=1.2, beta=0.10, d_cutoff=1.0)
        self._f_pitch = OneEuroFilter(min_cutoff=1.2, beta=0.10, d_cutoff=1.0)
        self._f_roll = OneEuroFilter(min_cutoff=1.2, beta=0.10, d_cutoff=1.0)

        self._active_shape = "NORMAL"

        # Runtime tunable sensitivity multipliers (Pause menu)
        self.smile_sensitivity = 1.0
        self.open_sensitivity = 1.0
        self.steer_sensitivity = 1.0

        # Raw RGB bytes for PiP display
        self.pip_rgb_bytes = None

    def set_sensitivities(self, smile=None, open_m=None, steer=None):
        with self.lock:
            if smile is not None:
                self.smile_sensitivity = float(np.clip(smile, 0.5, 2.0))
            if open_m is not None:
                self.open_sensitivity = float(np.clip(open_m, 0.5, 2.0))
            if steer is not None:
                self.steer_sensitivity = float(np.clip(steer, 0.5, 2.0))

    def get_sensitivities(self):
        with self.lock:
            return {
                "smile_sensitivity": self.smile_sensitivity,
                "open_sensitivity": self.open_sensitivity,
                "steer_sensitivity": self.steer_sensitivity,
            }

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

    def trigger_calibration(self, stage="NEUTRAL"):
        """Starts or advances the 3-Step Calibration Wizard ('NEUTRAL', 'SMILE', 'OPEN', or None)."""
        with self.lock:
            self.calibration_stage = stage
            self.is_calibrating = stage is not None
            if stage in self._cal_samples:
                self._cal_samples[stage].clear()

    def finalize_calibration_stage(self, stage):
        """Commits whatever samples were gathered for `stage` so wizard advances cleanly."""
        with self.lock:
            samples = list(self._cal_samples.get(stage, []))
        if len(samples) >= 5:
            self._apply_calibration_samples(stage, np.array(samples))

    def _apply_calibration_samples(self, stage, arr):
        """Applies robust median/85th-percentile personalized calibration from captured samples."""
        if stage == "NEUTRAL":
            self.base_mouth_w_ratio = float(np.median(arr[:, 0]))
            self.base_corner_lift = float(np.median(arr[:, 1]))
            self.base_inner_h = float(np.median(arr[:, 2]))
            self.base_mar = float(np.median(arr[:, 3]))
            self.base_outer_h = float(np.median(arr[:, 4]))
            self.base_jaw_h = float(np.median(arr[:, 5]))
            self.base_brow_h = float(np.median(arr[:, 6]))
            self.base_left_ear = max(0.14, float(np.median(arr[:, 7])))
            self.base_right_ear = max(0.14, float(np.median(arr[:, 8])))
            self.base_yaw = float(np.median(arr[:, 9]))
            self.base_roll = float(np.median(arr[:, 10]))
            self.base_yaw_3d = float(np.median(arr[:, 11]))
            print(
                f"[Calibration NEUTRAL] w={self.base_mouth_w_ratio:.3f}, inner={self.base_inner_h:.3f}, "
                f"brow={self.base_brow_h:.3f}, EAR=({self.base_left_ear:.2f},{self.base_right_ear:.2f})",
                flush=True
            )
        elif stage == "SMILE":
            peak_w = float(np.percentile(arr[:, 0], 85))
            peak_lift = float(np.percentile(arr[:, 1], 85))
            dw = peak_w - self.base_mouth_w_ratio
            dl = peak_lift - self.base_corner_lift
            # Keep spans within safe anatomical bounds so a weak smile still works
            self.smile_width_span = float(np.clip(dw * 0.85, 0.045, 0.14))
            self.smile_lift_span = float(np.clip(dl * 0.85, 0.024, 0.075))
            print(
                f"[Calibration SMILE] span_w={self.smile_width_span:.3f}, span_lift={self.smile_lift_span:.3f}",
                flush=True
            )
        elif stage == "OPEN":
            peak_inner = float(np.percentile(arr[:, 2], 85))
            peak_mar = float(np.percentile(arr[:, 3], 85))
            peak_outer = float(np.percentile(arr[:, 4], 85))
            peak_jaw = float(np.percentile(arr[:, 5], 85))
            self.open_inner_span = float(np.clip((peak_inner - self.base_inner_h) * 0.78, 0.042, 0.14))
            self.open_mar_span = float(np.clip((peak_mar - self.base_mar) * 0.78, 0.080, 0.22))
            self.open_outer_span = float(np.clip((peak_outer - self.base_outer_h) * 0.78, 0.060, 0.16))
            self.open_jaw_span = float(np.clip((peak_jaw - self.base_jaw_h) * 0.78, 0.065, 0.18))
            print(
                f"[Calibration OPEN] span_inner={self.open_inner_span:.3f}, span_outer={self.open_outer_span:.3f}, "
                f"span_jaw={self.open_jaw_span:.3f}",
                flush=True
            )

    def _open_camera(self):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(self.camera_id)

    def _extract_mesh_features(self, pts):
        """
        Computes scale-invariant and roll-invariant facial expression & eye/brow metrics
        from 468-point 3D FaceMesh landmarks.
        """
        p_le = pts[33]
        p_re = pts[263]
        eye_vec = p_re - p_le
        eye_dist = max(float(np.linalg.norm(eye_vec)), 1.0)
        roll_rad = float(np.arctan2(eye_vec[1], eye_vec[0]))
        roll_deg = roll_rad * 180.0 / np.pi

        c, s = np.cos(-roll_rad), np.sin(-roll_rad)
        origin = pts[1]
        shifted = pts - origin
        rx = shifted[:, 0] * c - shifted[:, 1] * s
        ry = shifted[:, 0] * s + shifted[:, 1] * c

        # 1. Mouth Width
        mouth_w = max(float(abs(rx[291] - rx[61])), 1.0)
        mouth_w_ratio = mouth_w / eye_dist

        # 2. Inner Lip Gap & MAR
        gap_center = abs(ry[14] - ry[13])
        gap_left = abs(ry[87] - ry[82])
        gap_right = abs(ry[317] - ry[312])
        mouth_open_px = float((gap_center + gap_left + gap_right) / 3.0)
        inner_h = mouth_open_px / eye_dist
        mar = mouth_open_px / mouth_w

        # 3. Outer Lip Vertical Height & Jaw Drop
        outer_h = float(abs(ry[17] - ry[0]) / eye_dist)
        jaw_h = float(abs(ry[152] - ry[1]) / eye_dist)

        # 4. Smile Corner Lift anchored to UPPER lip (13)
        corner_y = (ry[61] + ry[291]) * 0.5
        corner_lift = float((ry[13] - corner_y) / eye_dist)

        # 5. Eyebrow Raise height (eye center Y minus brow Y -> higher when brows lift!)
        left_brow_y = float(np.mean(ry[LEFT_BROW_IDX]))
        right_brow_y = float(np.mean(ry[RIGHT_BROW_IDX]))
        left_eye_y = float((ry[159] + ry[145]) * 0.5)
        right_eye_y = float((ry[386] + ry[374]) * 0.5)
        brow_h = float(((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) * 0.5 / eye_dist)

        # 6. Left & Right Eye Aspect Ratios (for Wink Detection)
        left_ear = _eye_aspect_ratio(pts, LEFT_EYE_IDX)
        right_ear = _eye_aspect_ratio(pts, RIGHT_EYE_IDX)

        # 7. Cheek-to-nose Yaw fallback
        dist_l = max(float(rx[1] - rx[234]), 1.0)
        dist_r = max(float(rx[454] - rx[1]), 1.0)
        yaw = (dist_l - dist_r) / (dist_l + dist_r)

        return (
            mouth_w_ratio, corner_lift, inner_h, mar, outer_h, jaw_h,
            brow_h, left_ear, right_ear, yaw, roll_deg
        )

    def _run_loop(self):
        try:
            if os.path.exists(self.model_path):
                self.detector = SCRFDDetector(self.model_path)
            else:
                print(f"[ExpressionWorker] SCRFD model not found at {self.model_path}", flush=True)
        except Exception as e:
            print(f"[ExpressionWorker] SCRFD load error: {e}", flush=True)

        try:
            if ort is not None and os.path.exists(self.mesh_path):
                self.mesh = FaceMesh468(self.mesh_path)
                print("[ExpressionWorker] Loaded 468-point FaceMesh + 3D solvePnP + One-Euro engine!", flush=True)
        except Exception as e:
            print(f"[ExpressionWorker] FaceMesh load error: {e}", flush=True)
            self.mesh = None

        self.cap = self._open_camera()
        if not self.cap or not self.cap.isOpened():
            print(f"[ExpressionWorker] Camera {self.camera_id} unavailable. Keyboard fallback active.", flush=True)
            with self.lock:
                self.camera_available = False
                self.is_calibrating = False
                self.calibration_stage = None
        else:
            with self.lock:
                self.camera_available = True

        last_time = time.perf_counter()

        while self.running:
            if not self.cap or not self.cap.isOpened() or self.detector is None:
                with self.lock:
                    self.is_calibrating = False
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if float(frame.mean()) < 75.0:
                frame = cv2.convertScaleAbs(frame, alpha=1.28, beta=18)

            now = time.perf_counter()
            current_fps = 1.0 / max(now - last_time, 1e-4)
            last_time = now

            try:
                bboxes, _ = self.detector.detect(frame)
            except Exception:
                bboxes = np.array([])

            face_found = len(bboxes) > 0
            smooth_steer = 0.0
            smooth_smile = 0.0
            smooth_open = 0.0
            smooth_brow = 0.0
            yaw_3d, pitch_3d, roll_3d = 0.0, 0.0, 0.0

            if face_found:
                centers = np.column_stack([
                    bboxes[:, 0] + bboxes[:, 2] * 0.5 - w * 0.5,
                    bboxes[:, 1] + bboxes[:, 3] * 0.5 - h * 0.5
                ])
                best_idx = int(np.argmin(np.sum(np.abs(centers), axis=1)))
                bbox = bboxes[best_idx]

                mesh_pts = self.mesh.predict(frame, bbox) if self.mesh is not None else None

                if mesh_pts is not None:
                    (
                        mouth_w_ratio, corner_lift, inner_h, mar, outer_h, jaw_h,
                        brow_h, left_ear, right_ear, yaw_2d, roll_2d
                    ) = self._extract_mesh_features(mesh_pts)

                    # True 3D Head Pose via cv2.solvePnP
                    pose_res = self.pose_3d.estimate(mesh_pts, w, h)
                    rvec = tvec = cam_mat = None
                    if pose_res is not None:
                        raw_roll_3d, raw_yaw_3d, raw_pitch_3d, rvec, tvec, cam_mat = pose_res
                        roll_3d = self._f_roll.filter(raw_roll_3d, now)
                        yaw_3d = self._f_yaw.filter(raw_yaw_3d, now)
                        pitch_3d = self._f_pitch.filter(raw_pitch_3d, now)
                    else:
                        roll_3d = roll_2d
                        yaw_3d = yaw_2d * 35.0
                        pitch_3d = 0.0

                    # 3-Step Calibration Sample Collection
                    with self.lock:
                        cal_stage = self.calibration_stage
                    if cal_stage in self._cal_samples:
                        sample = (
                            mouth_w_ratio, corner_lift, inner_h, mar, outer_h, jaw_h,
                            brow_h, left_ear, right_ear, yaw_2d, roll_2d, yaw_3d
                        )
                        self._cal_samples[cal_stage].append(sample)
                        if len(self._cal_samples[cal_stage]) >= self.calibration_target_count:
                            self._apply_calibration_samples(cal_stage, np.array(self._cal_samples[cal_stage]))
                            if cal_stage == "NEUTRAL":
                                # Wait for game wizard (or finish if standalone auto-cal)
                                pass

                    with self.lock:
                        smile_sens = self.smile_sensitivity
                        open_sens = self.open_sensitivity
                        steer_sens = self.steer_sensitivity

                    # 1. Smile Score (Width Expansion + Upper-Lip Corner Lift)
                    width_score = float(np.clip((mouth_w_ratio - self.base_mouth_w_ratio) / max(self.smile_width_span, 0.04), 0.0, 1.0))
                    lift_score = float(np.clip((corner_lift - self.base_corner_lift) / max(self.smile_lift_span, 0.02), 0.0, 1.0))

                    # 2. Multi-Signal Open Mouth Score (Inner, MAR, Outer, Jaw Drop, O-Pucker)
                    inner_score = float(np.clip((inner_h - self.base_inner_h) / max(self.open_inner_span, 0.04), 0.0, 1.0))
                    mar_score = float(np.clip((mar - self.base_mar) / max(self.open_mar_span, 0.07), 0.0, 1.0))
                    outer_score = float(np.clip((outer_h - self.base_outer_h) / max(self.open_outer_span, 0.05), 0.0, 1.0))
                    jaw_score = float(np.clip((jaw_h - self.base_jaw_h) / max(self.open_jaw_span, 0.06), 0.0, 1.0))
                    pucker_score = float(np.clip(
                        (self.base_mouth_w_ratio - mouth_w_ratio) / max(self.base_mouth_w_ratio * 0.14, 0.04),
                        0.0, 1.0
                    ))

                    lip_open = 0.40 * inner_score + 0.30 * mar_score + 0.30 * outer_score
                    raw_open = max(lip_open, 0.65 * lip_open + 0.35 * jaw_score)
                    if pucker_score > 0.22 and (inner_score > 0.12 or outer_score > 0.15 or jaw_score > 0.20):
                        raw_open = max(raw_open, 0.55 * lip_open + 0.55 * pucker_score)

                    raw_smile = (0.58 * width_score + 0.42 * lift_score) * smile_sens
                    if width_score > 0.55 and lift_score > 0.35:
                        raw_open *= 0.40
                    elif raw_open > 0.40 and width_score < 0.35:
                        raw_smile *= 0.25

                    smile_score = float(np.clip(raw_smile, 0.0, 1.0))
                    open_score = float(np.clip(raw_open * open_sens, 0.0, 1.0))

                    # 3. Eyebrow Raise -> JUMP (edge-triggered with cooldown)
                    raw_brow = float(np.clip((brow_h - self.base_brow_h) / 0.048, 0.0, 1.0))
                    smooth_brow = float(np.clip(self._f_brow.filter(raw_brow, now), 0.0, 1.0))
                    if smooth_brow > 0.62 and self._brow_armed and (now - self._last_jump_time) > 0.85:
                        self._brow_armed = False
                        self._last_jump_time = now
                        with self.lock:
                            self._jump_pending = True
                    elif smooth_brow < 0.35:
                        self._brow_armed = True

                    # 4. Asymmetric Wink -> SHIELD + NITRO (ignores normal two-eye blinks!)
                    left_ratio = left_ear / max(self.base_left_ear, 0.10)
                    right_ratio = right_ear / max(self.base_right_ear, 0.10)
                    is_left_wink = (left_ratio < 0.68 and right_ratio > 0.82)
                    is_right_wink = (right_ratio < 0.68 and left_ratio > 0.82)
                    if is_left_wink or is_right_wink:
                        self._wink_hold_frames += 1
                        if self._wink_hold_frames >= 3 and (now - self._last_wink_time) > 2.5:
                            self._last_wink_time = now
                            self._wink_hold_frames = 0
                            with self.lock:
                                self._wink_pending = True
                    else:
                        self._wink_hold_frames = 0

                    # 5. Steering via 3D solvePnP Yaw + Cheek Yaw + Roll (filtered by One-Euro)
                    diff_yaw_3d = (yaw_3d - self.base_yaw_3d) / 22.0
                    diff_yaw_2d = (yaw_2d - self.base_yaw) * 1.5
                    blended_yaw = 0.55 * diff_yaw_3d + 0.45 * diff_yaw_2d
                    diff_roll = (roll_2d - self.base_roll) / 26.0
                    raw_steer = (blended_yaw + diff_roll * 0.55) * steer_sens
                    if abs(raw_steer) < 0.10:
                        raw_steer = 0.0
                    target_steer = float(np.clip(raw_steer, -1.0, 1.0))

                    # One-Euro Filtering for all continuous game channels
                    smooth_steer = float(np.clip(self._f_steer.filter(target_steer, now), -1.0, 1.0))
                    smooth_smile = float(np.clip(self._f_smile.filter(smile_score, now), 0.0, 1.0))
                    smooth_open = float(np.clip(self._f_open.filter(open_score, now), 0.0, 1.0))

                    self._update_shape_hysteresis(smooth_smile, smooth_open)

                    # Draw 3D axes, contours & expression HUD on PiP webcam frame
                    self._draw_pip_mesh_hud(
                        frame, bbox.astype(int), mesh_pts,
                        smooth_smile, smooth_open, smooth_brow,
                        self._active_shape, yaw_3d, pitch_3d, roll_3d,
                        rvec, tvec, cam_mat
                    )
                else:
                    self._active_shape = "NORMAL"
            else:
                self._active_shape = "NORMAL"
                cv2.putText(frame, "NO FACE DETECTED", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Keys: A/D, Down/Up, J=Jump, Shift=Shield", (15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

            pip_img = cv2.resize(frame, (PIP_WIDTH, PIP_HEIGHT))
            pip_rgb = cv2.cvtColor(pip_img, cv2.COLOR_BGR2RGB)
            rgb_bytes = pip_rgb.tobytes()

            with self.lock:
                self.face_detected = face_found
                self.steer_x = smooth_steer
                self.current_shape = self._active_shape
                self.smile_intensity = smooth_smile
                self.open_intensity = smooth_open
                self.brow_intensity = smooth_brow
                self.yaw_deg = yaw_3d
                self.pitch_deg = pitch_3d
                self.roll_deg = roll_3d
                self.fps = current_fps
                self.pip_rgb_bytes = rgb_bytes

        if self.cap:
            self.cap.release()

    def _update_shape_hysteresis(self, smile, open_m):
        """Hysteresis thresholds so shapes engage quickly (>0.42) and hold steadily (>0.28)."""
        enter_thresh = 0.42
        exit_thresh = 0.28

        if self._active_shape == "FLAT":
            if open_m > enter_thresh and open_m > smile + 0.12:
                self._active_shape = "NARROW"
            elif smile < exit_thresh:
                self._active_shape = "NARROW" if open_m > enter_thresh else "NORMAL"
        elif self._active_shape == "NARROW":
            if smile > enter_thresh and smile > open_m + 0.12:
                self._active_shape = "FLAT"
            elif open_m < exit_thresh:
                self._active_shape = "FLAT" if smile > enter_thresh else "NORMAL"
        else:
            if smile > enter_thresh and smile >= open_m:
                self._active_shape = "FLAT"
            elif open_m > enter_thresh:
                self._active_shape = "NARROW"

    def _draw_pip_mesh_hud(
        self, frame, bbox, pts, smile_val, open_val, brow_val,
        shape, yaw_d, pitch_d, roll_d, rvec, tvec, cam_mat
    ):
        """Draws 3D solvePnP axes, eye/brow/lip contours, and live gauges on PiP frame."""
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 210, 255), 1)

        # Eye contours
        for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            poly = pts[eye_idx].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], True, (0, 255, 140), 1, cv2.LINE_AA)

        # Eyebrow contours (glow gold when raised!)
        brow_col = (0, 230, 255) if brow_val > 0.55 else (180, 200, 120)
        for brow_idx in (LEFT_BROW_IDX, RIGHT_BROW_IDX):
            poly = pts[brow_idx].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], False, brow_col, 2, cv2.LINE_AA)

        # Lip contours
        lip_col = (255, 0, 200) if shape == "FLAT" else ((0, 255, 140) if shape == "NARROW" else (0, 230, 255))
        cv2.polylines(frame, [pts[OUTER_LIP_IDX].astype(np.int32).reshape((-1, 1, 2))], True, lip_col, 2, cv2.LINE_AA)
        cv2.polylines(frame, [pts[INNER_LIP_IDX].astype(np.int32).reshape((-1, 1, 2))], True, (255, 255, 255), 1, cv2.LINE_AA)

        # 3D solvePnP XYZ axes projecting from nose tip
        HeadPose3D.draw_axes(frame, pts, rvec, tvec, cam_mat, axis_len=42.0)

        # Top HUD badge
        bcolor = {"NORMAL": (255, 210, 0), "FLAT": (128, 0, 255), "NARROW": (140, 255, 0)}.get(shape, (255, 255, 255))
        cv2.rectangle(frame, (8, 8), (285, 142), (18, 16, 26), -1)
        cv2.rectangle(frame, (8, 8), (285, 142), bcolor, 1)
        cv2.putText(frame, f"CAR: {shape}", (16, 33), cv2.FONT_HERSHEY_DUPLEX, 0.68, bcolor, 2)

        # Smile, Open, Brow meters
        meters = [
            ("Smile", smile_val, (128, 0, 255), 56),
            ("Open ", open_val, (140, 255, 0), 79),
            ("Brow ", brow_val, (0, 220, 255), 102),
        ]
        for label, val, col, y_pos in meters:
            cv2.putText(frame, f"{label} {int(val * 100):>3}%", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.rectangle(frame, (128, y_pos - 11), (272, y_pos + 2), (45, 45, 55), -1)
            cv2.rectangle(frame, (128, y_pos - 11), (128 + int(144 * val), y_pos + 2), col, -1)

        # 3D Euler angle telemetry
        cv2.putText(
            frame, f"3D Y:{yaw_d:+.0f} P:{pitch_d:+.0f} R:{roll_d:+.0f}",
            (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 230, 255), 1
        )

    def get_state(self, consume_triggers=False):
        """
        Returns thread-safe snapshot of latest vision outputs.
        If `consume_triggers=True`, atomically reads and resets edge-triggered
        `jump_triggered` and `wink_triggered` flags.
        """
        with self.lock:
            jump_trig = self._jump_pending
            wink_trig = self._wink_pending
            if consume_triggers:
                self._jump_pending = False
                self._wink_pending = False
            return {
                "steer_x": self.steer_x,
                "shape": self.current_shape,
                "face_detected": self.face_detected,
                "camera_available": self.camera_available,
                "smile_intensity": self.smile_intensity,
                "open_intensity": self.open_intensity,
                "brow_intensity": self.brow_intensity,
                "jump_triggered": jump_trig,
                "wink_triggered": wink_trig,
                "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg,
                "roll_deg": self.roll_deg,
                "is_calibrating": self.is_calibrating,
                "calibration_stage": self.calibration_stage,
                "fps": self.fps,
                "pip_rgb_bytes": self.pip_rgb_bytes,
            }
