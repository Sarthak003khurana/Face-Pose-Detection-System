"""
High-Precision Asynchronous Facial Expression & Head Pose Worker.
Dual-Stage Pipeline (pure OpenCV + ONNX Runtime, ~40-60 FPS on CPU):
  Stage 1: SCRFD ONNX detects the face bounding box robustly (even in low light).
  Stage 2: MediaPipe 468-point 3D FaceMesh ONNX extracts exact lip, mouth-corner,
           eye, nose, and cheek contours for rock-solid Smile, O-Mouth, and Steering.

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


# MediaPipe 468-point landmark contour indices for PiP visualization
OUTER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


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
        """
        Crops a padded square ROI around `bbox` (x, y, w, h) and returns
        (468, 2) landmark coordinates in original frame pixel coordinates.
        """
        fh, fw = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        cx = x + bw * 0.5
        cy = y + bh * 0.5
        side = max(bw, bh) * 1.45

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

        # Map from [0..192] back to original frame coordinates
        scale_back = crop_side / float(self.inp_size)
        pts[:, 0] = pts[:, 0] * scale_back + x1
        pts[:, 1] = pts[:, 1] * scale_back + y1
        return pts


class ExpressionWorker:
    """
    Background worker thread that runs SCRFD + 468-point FaceMesh,
    computes rotation-invariant Smile, O-Mouth (MAR), and Head Steering,
    and feeds thread-safe outputs to the 60 FPS game thread.
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
        self.cap = None
        self.camera_available = False

        # Outputs read by the game
        self.steer_x = 0.0
        self.current_shape = "NORMAL"
        self.face_detected = False
        self.smile_intensity = 0.0
        self.open_intensity = 0.0
        self.fps = 0.0

        # Calibration (auto-calibrates on first face detection and when triggered)
        self.is_calibrating = True
        self.calibration_samples = []
        self.calibration_target_count = 30  # ~1 second of neutral frames

        # Realistic neutral defaults for 468-point FaceMesh
        self.base_mouth_w_ratio = 0.55
        self.base_corner_lift = 0.00
        self.base_mar = 0.055
        self.base_yaw = 0.0
        self.base_roll = 0.0

        # Smoothed signals & hysteresis state
        self._smooth_steer = 0.0
        self._smooth_smile = 0.0
        self._smooth_open = 0.0
        self._active_shape = "NORMAL"

        # Runtime tunable sensitivity multipliers (adjustable from Pause menu)
        self.smile_sensitivity = 1.0
        self.open_sensitivity = 1.0
        self.steer_sensitivity = 1.0

        # Raw RGB bytes for PiP display
        self.pip_rgb_bytes = None

    def set_sensitivities(self, smile=None, open_m=None, steer=None):
        """Updates expression and steering sensitivity multipliers safely at runtime."""
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

    def trigger_calibration(self):
        with self.lock:
            self.is_calibrating = True
            self.calibration_samples.clear()

    def _open_camera(self):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(self.camera_id)

    def _extract_mesh_features(self, pts):
        """
        Computes exact, scale-invariant and roll-invariant facial expression metrics
        from 468-point 3D FaceMesh landmarks.
        """
        # Eye outer corners (33 = left eye outer, 263 = right eye outer)
        p_le = pts[33]
        p_re = pts[263]
        eye_vec = p_re - p_le
        eye_dist = max(float(np.linalg.norm(eye_vec)), 1.0)
        roll_rad = float(np.arctan2(eye_vec[1], eye_vec[0]))
        roll_deg = roll_rad * 180.0 / np.pi

        # Rotate landmarks by -roll_rad around nose tip (1) to level the head
        c, s = np.cos(-roll_rad), np.sin(-roll_rad)
        origin = pts[1]
        shifted = pts - origin
        rx = shifted[:, 0] * c - shifted[:, 1] * s
        ry = shifted[:, 0] * s + shifted[:, 1] * c

        # 1. True Mouth Width (left corner 61 to right corner 291) normalized by eye distance
        mouth_w = max(float(abs(rx[291] - rx[61])), 1.0)
        mouth_w_ratio = mouth_w / eye_dist

        # 2. True Inner Lip Vertical Opening (average of left, center, right inner lip gaps)
        gap_center = abs(ry[14] - ry[13])
        gap_left = abs(ry[87] - ry[82])
        gap_right = abs(ry[317] - ry[312])
        mouth_open_h = float((gap_center + gap_left + gap_right) / 3.0)

        # True Mouth Aspect Ratio (MAR): vertical inner lip opening / mouth width
        mar = mouth_open_h / mouth_w

        # 3. Mouth Corner Lift (positive when corners pull UP higher than lip center = smile)
        corner_y = (ry[61] + ry[291]) * 0.5
        lip_center_y = (ry[13] + ry[14]) * 0.5
        corner_lift = float((lip_center_y - corner_y) / eye_dist)

        # 4. Head Yaw from cheek contour (234 = left cheek, 454 = right cheek, 1 = nose tip)
        dist_l = max(float(rx[1] - rx[234]), 1.0)
        dist_r = max(float(rx[454] - rx[1]), 1.0)
        yaw = (dist_l - dist_r) / (dist_l + dist_r)

        return mouth_w_ratio, corner_lift, mar, yaw, roll_deg

    def _run_loop(self):
        # Load Stage 1: SCRFD
        try:
            if os.path.exists(self.model_path):
                self.detector = SCRFDDetector(self.model_path)
            else:
                print(f"[ExpressionWorker] SCRFD model not found at {self.model_path}", flush=True)
        except Exception as e:
            print(f"[ExpressionWorker] SCRFD load error: {e}", flush=True)

        # Load Stage 2: 468-point FaceMesh ONNX
        try:
            if ort is not None and os.path.exists(self.mesh_path):
                self.mesh = FaceMesh468(self.mesh_path)
                print("[ExpressionWorker] Loaded 468-point FaceMesh ONNX engine!", flush=True)
        except Exception as e:
            print(f"[ExpressionWorker] FaceMesh load error (falling back to 5-pt): {e}", flush=True)
            self.mesh = None

        self.cap = self._open_camera()
        if not self.cap or not self.cap.isOpened():
            print(f"[ExpressionWorker] Camera {self.camera_id} unavailable. Keyboard fallback active.", flush=True)
            with self.lock:
                self.camera_available = False
                self.is_calibrating = False
        else:
            with self.lock:
                self.camera_available = True

        last_time = time.time()

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

            # Auto low-light boost for dim rooms (keeps SCRFD & FaceMesh locked on)
            if float(frame.mean()) < 75.0:
                frame = cv2.convertScaleAbs(frame, alpha=1.28, beta=18)

            now = time.time()
            current_fps = 1.0 / max(now - last_time, 1e-4)
            last_time = now

            try:
                bboxes, kpss = self.detector.detect(frame)
            except Exception:
                bboxes, kpss = np.array([]), np.array([])

            face_found = len(bboxes) > 0
            target_steer = 0.0
            smile_score = 0.0
            open_score = 0.0

            if face_found:
                # Choose face closest to center
                centers = np.column_stack([
                    bboxes[:, 0] + bboxes[:, 2] * 0.5 - w * 0.5,
                    bboxes[:, 1] + bboxes[:, 3] * 0.5 - h * 0.5
                ])
                best_idx = int(np.argmin(np.sum(np.abs(centers), axis=1)))
                bbox = bboxes[best_idx]

                mesh_pts = self.mesh.predict(frame, bbox) if self.mesh is not None else None

                if mesh_pts is not None:
                    mouth_w_ratio, corner_lift, mar, yaw, roll_deg = self._extract_mesh_features(mesh_pts)

                    # Accumulate neutral baseline calibration
                    with self.lock:
                        calibrating = self.is_calibrating
                    if calibrating:
                        self.calibration_samples.append((mouth_w_ratio, corner_lift, mar, yaw, roll_deg))
                        if len(self.calibration_samples) >= self.calibration_target_count:
                            arr = np.array(self.calibration_samples)
                            self.base_mouth_w_ratio = float(np.median(arr[:, 0]))
                            self.base_corner_lift = float(np.median(arr[:, 1]))
                            self.base_mar = float(np.median(arr[:, 2]))
                            self.base_yaw = float(np.median(arr[:, 3]))
                            self.base_roll = float(np.median(arr[:, 4]))
                            with self.lock:
                                self.is_calibrating = False
                            print(
                                f"[ExpressionWorker] 468-pt Calibrated! "
                                f"mouth_w={self.base_mouth_w_ratio:.3f}, lift={self.base_corner_lift:.3f}, MAR={self.base_mar:.3f}",
                                flush=True
                            )

                    with self.lock:
                        smile_sens = self.smile_sensitivity
                        open_sens = self.open_sensitivity
                        steer_sens = self.steer_sensitivity

                    # 1. Open Mouth ("O") Score via true inner lip MAR
                    delta_mar = mar - self.base_mar
                    o_narrow_bonus = max(0.0, (self.base_mouth_w_ratio - mouth_w_ratio) / max(self.base_mouth_w_ratio * 0.18, 0.04))
                    raw_open = (delta_mar / 0.24) + (0.25 * o_narrow_bonus if delta_mar > 0.08 else 0.0)
                    open_score = float(np.clip(raw_open * open_sens, 0.0, 1.0))

                    # 2. Smile Score via Mouth Width Expansion + Corner Lift
                    width_gain = (mouth_w_ratio - self.base_mouth_w_ratio) / max(self.base_mouth_w_ratio * 0.13, 0.04)
                    lift_gain = (corner_lift - self.base_corner_lift) / 0.042
                    width_score = float(np.clip(width_gain, 0.0, 1.0))
                    lift_score = float(np.clip(lift_gain, 0.0, 1.0))

                    raw_smile = (0.55 * width_score + 0.45 * lift_score) * smile_sens
                    if mar > self.base_mar + 0.28 and width_score < 0.35:
                        raw_smile *= 0.35
                    smile_score = float(np.clip(raw_smile, 0.0, 1.0))

                    # 3. Steering from true cheek-to-nose Yaw + eye Roll (with deadzone)
                    diff_yaw = yaw - self.base_yaw
                    diff_roll = (roll_deg - self.base_roll) / 26.0
                    raw_steer = (diff_yaw * 1.65 + diff_roll * 0.60) * steer_sens
                    if abs(raw_steer) < 0.10:
                        raw_steer = 0.0
                    target_steer = float(np.clip(raw_steer, -1.0, 1.0))

                    # Temporal smoothing (fast response, zero jitter)
                    self._smooth_steer += (target_steer - self._smooth_steer) * 0.28
                    self._smooth_smile += (smile_score - self._smooth_smile) * 0.35
                    self._smooth_open += (open_score - self._smooth_open) * 0.35

                    # Hysteresis state machine (prevents morph flicker)
                    self._update_shape_hysteresis(self._smooth_smile, self._smooth_open)

                    # Draw crisp 468-pt contours & meters on PiP frame
                    self._draw_pip_mesh_hud(frame, bbox.astype(int), mesh_pts, self._smooth_smile, self._smooth_open, self._active_shape)
                else:
                    # Fallback if mesh couldn't run
                    self._active_shape = "NORMAL"
            else:
                self._smooth_steer *= 0.85
                self._smooth_smile *= 0.80
                self._smooth_open *= 0.80
                self._active_shape = "NORMAL"
                cv2.putText(frame, "NO FACE DETECTED", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Keys: A/D Steer, Down/Up Morph", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Convert PiP image to raw RGB bytes (thread-safe!)
            pip_img = cv2.resize(frame, (PIP_WIDTH, PIP_HEIGHT))
            pip_rgb = cv2.cvtColor(pip_img, cv2.COLOR_BGR2RGB)
            rgb_bytes = pip_rgb.tobytes()

            with self.lock:
                self.face_detected = face_found
                self.steer_x = self._smooth_steer
                self.current_shape = self._active_shape
                self.smile_intensity = self._smooth_smile
                self.open_intensity = self._smooth_open
                self.fps = current_fps
                self.pip_rgb_bytes = rgb_bytes

        if self.cap:
            self.cap.release()

    def _update_shape_hysteresis(self, smile, open_m):
        """
        Hysteresis thresholds so shapes engage quickly (>0.48) and hold steadily
        until released (<0.34), prioritizing whichever expression is stronger.
        """
        enter_thresh = 0.48
        exit_thresh = 0.34

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

    def _draw_pip_mesh_hud(self, frame, bbox, pts, smile_val, open_val, shape):
        """Draws crisp lip/eye contours and live expression meters on the PiP webcam frame."""
        # Subtle face box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 210, 255), 1)

        # Draw Eye contours (green)
        for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            poly = pts[eye_idx].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], True, (0, 255, 140), 1, cv2.LINE_AA)

        # Draw Outer & Inner Lip contours (color reacts to active expression!)
        lip_col = (255, 0, 200) if shape == "FLAT" else ((0, 255, 140) if shape == "NARROW" else (0, 230, 255))
        outer_lip = pts[OUTER_LIP_IDX].astype(np.int32).reshape((-1, 1, 2))
        inner_lip = pts[INNER_LIP_IDX].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [outer_lip], True, lip_col, 2, cv2.LINE_AA)
        cv2.polylines(frame, [inner_lip], True, (255, 255, 255), 1, cv2.LINE_AA)

        # Nose dot
        cv2.circle(frame, (int(pts[1][0]), int(pts[1][1])), 3, (0, 200, 255), -1)

        # Top HUD badge (BGR colors for OpenCV)
        bcolor = {"NORMAL": (255, 210, 0), "FLAT": (128, 0, 255), "NARROW": (140, 255, 0)}.get(shape, (255, 255, 255))
        cv2.rectangle(frame, (8, 8), (275, 102), (18, 16, 26), -1)
        cv2.rectangle(frame, (8, 8), (275, 102), bcolor, 1)
        cv2.putText(frame, f"CAR: {shape}", (16, 35), cv2.FONT_HERSHEY_DUPLEX, 0.72, bcolor, 2)

        # Smile bar
        cv2.putText(frame, f"Smile {int(smile_val * 100):>3}%", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.rectangle(frame, (125, 50), (262, 64), (45, 45, 55), -1)
        cv2.rectangle(frame, (125, 50), (125 + int(137 * smile_val), 64), (128, 0, 255), -1)

        # O-Mouth bar
        cv2.putText(frame, f"Open  {int(open_val * 100):>3}%", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.rectangle(frame, (125, 76), (262, 90), (45, 45, 55), -1)
        cv2.rectangle(frame, (125, 76), (125 + int(137 * open_val), 90), (140, 255, 0), -1)

    def get_state(self):
        with self.lock:
            return {
                "steer_x": self.steer_x,
                "shape": self.current_shape,
                "face_detected": self.face_detected,
                "camera_available": self.camera_available,
                "smile_intensity": self.smile_intensity,
                "open_intensity": self.open_intensity,
                "is_calibrating": self.is_calibrating,
                "fps": self.fps,
                "pip_rgb_bytes": self.pip_rgb_bytes
            }
