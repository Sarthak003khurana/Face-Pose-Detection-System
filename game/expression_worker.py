"""
Asynchronous Camera & Facial Expression / Head Pose Worker.
Runs SCRFD face detection in a background thread to prevent game stutter.
Thread-safe: performs NO Pygame calls in the background thread (returns raw RGB frames).
"""

import threading
import time
import os
import platform
import cv2
import numpy as np

from game.config import (
    PIP_WIDTH, PIP_HEIGHT,
)


class SCRFDDetector:
    """Lightweight wrapper for SCRFD ONNX face detector using OpenCV DNN."""
    def __init__(self, onnx_path, conf_thresh=0.45, nms_thresh=0.45):
        self.inp_w = 640
        self.inp_h = 640
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.net = cv2.dnn.readNet(onnx_path)
        self.feat_strides = [8, 16, 32]
        self.fmc = 3
        self.num_anchors = 2

    def resize_image(self, img):
        h, w = img.shape[:2]
        scale = min(self.inp_h / h, self.inp_w / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_w = self.inp_w - new_w
        pad_h = self.inp_h - new_h
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded, scale

    def distance2bbox(self, points, distance):
        distance = distance.reshape(-1, 4)
        n = min(points.shape[0], distance.shape[0])
        pts = points[:n]
        dist = distance[:n]
        x1 = pts[:, 0] - dist[:, 0]
        y1 = pts[:, 1] - dist[:, 1]
        x2 = pts[:, 0] + dist[:, 2]
        y2 = pts[:, 1] + dist[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance):
        distance = distance.reshape(-1, 10)
        n = min(points.shape[0], distance.shape[0])
        pts = points[:n]
        dist = distance[:n]
        preds = []
        for i in range(0, 10, 2):
            px = pts[:, 0] + dist[:, i]
            py = pts[:, 1] + dist[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def detect(self, img):
        padded, scale = self.resize_image(img)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 128.0, (self.inp_w, self.inp_h),
            (127.5, 127.5, 127.5), swapRB=True
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self.feat_strides):
            scores = outs[idx * self.fmc][0]
            bbox_preds = outs[idx * self.fmc + 1][0] * stride
            kps_preds = outs[idx * self.fmc + 2][0] * stride

            h = blob.shape[2] // stride
            w = blob.shape[3] // stride
            anchor_centers = np.stack(
                np.mgrid[:h, :w][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=0)

            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))

            valid_len = min(len(scores), len(bboxes), len(kpss))
            scores = scores[:valid_len]
            bboxes = bboxes[:valid_len]
            kpss = kpss[:valid_len]

            pos_inds = np.where(scores >= self.conf_thresh)[0]
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        if not scores_list or all(len(s) == 0 for s in scores_list):
            return np.array([]), np.array([])

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)

        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        bboxes /= scale
        kpss /= scale

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), scores.tolist(),
            self.conf_thresh, self.nms_thresh
        )
        if len(indices) == 0:
            return np.array([]), np.array([])

        indices = np.array(indices).flatten()
        return bboxes[indices], kpss[indices]


class ExpressionWorker:
    """
    Background worker thread that captures webcam frames, runs face & expression
    detection, and provides smooth steering & morphing states to the game.
    """
    def __init__(self, model_path="Face_Pose/models/scrfd_500m_kps.onnx", camera_id=0):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.camera_id = camera_id
        self.model_path = model_path

        # Detector instance
        self.detector = None
        self.cap = None
        self.camera_available = False

        # Game state outputs
        self.steer_x = 0.0          # -1.0 (far left) to +1.0 (far right)
        self.current_shape = "NORMAL"
        self.face_detected = False
        self.smile_intensity = 0.0   # 0.0 to 1.0
        self.open_intensity = 0.0    # 0.0 to 1.0
        self.fps = 0.0

        # Calibration state
        self.is_calibrating = False
        self.calibration_samples = []
        self.calibration_target_count = 40  # ~1.5-2 seconds of frames
        self.baseline_smile = 0.82
        self.baseline_open = 0.70
        self.baseline_yaw = 0.0
        self.baseline_roll = 0.0

        # Smoothed values
        self._smooth_steer = 0.0
        self._smooth_smile = 0.0
        self._smooth_open = 0.0

        # Raw RGB bytes for PiP display (converted to Surface on main thread)
        self.pip_rgb_bytes = None

    def start(self):
        """Starts the worker thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Signals the thread to terminate."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

    def trigger_calibration(self):
        """Starts a fresh baseline calibration."""
        with self.lock:
            self.is_calibrating = True
            self.calibration_samples.clear()

    def _open_camera(self):
        """Opens webcam quickly using DirectShow on Windows with fallback."""
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(self.camera_id)

    def _run_loop(self):
        """Continuous camera capture & detection loop."""
        try:
            if os.path.exists(self.model_path):
                self.detector = SCRFDDetector(self.model_path)
            else:
                print(f"[ExpressionWorker] Model not found at {self.model_path}", flush=True)
        except Exception as e:
            print(f"[ExpressionWorker] Failed to load detector: {e}", flush=True)
            self.detector = None

        self.cap = self._open_camera()
        if not self.cap or not self.cap.isOpened():
            print(f"[ExpressionWorker] Camera {self.camera_id} unavailable. Using keyboard fallback.", flush=True)
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

            # Mirror frame for intuitive controls
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Measure vision loop FPS
            now = time.time()
            dt = max(now - last_time, 1e-4)
            current_fps = 1.0 / dt
            last_time = now

            # Detect faces
            try:
                bboxes, kpss = self.detector.detect(frame)
            except Exception:
                bboxes, kpss = np.array([]), np.array([])

            face_found = len(bboxes) > 0

            target_shape = "NORMAL"
            target_steer = 0.0
            smile_metric = 0.0
            open_metric = 0.0

            if face_found:
                # Pick the face closest to the image center
                centers = np.column_stack([
                    bboxes[:, 0] + bboxes[:, 2] / 2.0 - w / 2.0,
                    bboxes[:, 1] + bboxes[:, 3] / 2.0 - h / 2.0
                ])
                center_dists = np.sum(np.abs(centers), axis=1)
                best_idx = int(np.argmin(center_dists))

                bbox = bboxes[best_idx].astype(int)
                pts = kpss[best_idx]  # 5 landmarks: [L_eye, R_eye, Nose, L_mouth, R_mouth]

                # Geometry metrics
                p_le, p_re = pts[0], pts[1]
                p_nose = pts[2]
                p_lm, p_rm = pts[3], pts[4]

                # 1. Eye distance & tilt (Roll)
                dx_eyes = max(float(p_re[0] - p_le[0]), 1.0)
                dy_eyes = float(p_re[1] - p_le[1])
                roll_deg = np.arctan2(dy_eyes, dx_eyes) * 180.0 / np.pi

                # 2. Nose horizontal asymmetry (Yaw)
                dist_l2n = p_nose[0] - p_le[0]
                dist_n2r = p_re[0] - p_nose[0]
                yaw_ratio = (dist_l2n - dist_n2r) / dx_eyes

                # 3. Smile Ratio (mouth width / eye distance)
                dx_mouth = max(float(p_rm[0] - p_lm[0]), 1.0)
                smile_ratio = dx_mouth / dx_eyes

                # 4. Mouth Vertical Gap (nose-to-mouth midpoint / eye distance)
                mouth_y = (p_lm[1] + p_rm[1]) / 2.0
                dy_n2m = mouth_y - p_nose[1]
                open_ratio = dy_n2m / dx_eyes

                # Calibration accumulation
                with self.lock:
                    calibrating = self.is_calibrating
                if calibrating:
                    self.calibration_samples.append((smile_ratio, open_ratio, yaw_ratio, roll_deg))
                    if len(self.calibration_samples) >= self.calibration_target_count:
                        arr = np.array(self.calibration_samples)
                        self.baseline_smile = float(np.median(arr[:, 0]))
                        self.baseline_open = float(np.median(arr[:, 1]))
                        self.baseline_yaw = float(np.median(arr[:, 2]))
                        self.baseline_roll = float(np.median(arr[:, 3]))
                        with self.lock:
                            self.is_calibrating = False
                        print(f"[ExpressionWorker] Calibrated! Smile: {self.baseline_smile:.2f}, Open: {self.baseline_open:.2f}", flush=True)

                # Normalized scores
                delta_smile = smile_ratio - self.baseline_smile
                smile_metric = float(np.clip(delta_smile / max(self.baseline_smile * 0.16, 0.05), 0.0, 1.0))

                delta_open = open_ratio - self.baseline_open
                open_metric = float(np.clip(delta_open / max(self.baseline_open * 0.22, 0.05), 0.0, 1.0))

                # Steering mapping from head tilt and yaw
                diff_yaw = yaw_ratio - self.baseline_yaw
                diff_roll = (roll_deg - self.baseline_roll) / 25.0
                raw_steer = (diff_yaw * 2.2 + diff_roll * 0.8)
                target_steer = float(np.clip(raw_steer, -1.0, 1.0))

                # Shape classification
                if smile_metric > 0.58:
                    target_shape = "FLAT"
                elif open_metric > 0.58:
                    target_shape = "NARROW"
                else:
                    target_shape = "NORMAL"

                # Draw telemetry on PiP frame
                self._draw_pip_telemetry(frame, bbox, pts, smile_metric, open_metric, target_shape)
            else:
                cv2.putText(frame, "NO FACE DETECTED", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Keys: A/D Steer, Down/Up Morph", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Apply temporal smoothing
            alpha = 0.22
            self._smooth_steer += (target_steer - self._smooth_steer) * alpha
            self._smooth_smile += (smile_metric - self._smooth_smile) * alpha
            self._smooth_open += (open_metric - self._smooth_open) * alpha

            # Resize and convert to RGB raw bytes (no Pygame calls in thread!)
            pip_img = cv2.resize(frame, (PIP_WIDTH, PIP_HEIGHT))
            pip_rgb = cv2.cvtColor(pip_img, cv2.COLOR_BGR2RGB)
            rgb_bytes = pip_rgb.tobytes()

            # Update shared state
            with self.lock:
                self.face_detected = face_found
                self.steer_x = self._smooth_steer
                self.current_shape = target_shape
                self.smile_intensity = self._smooth_smile
                self.open_intensity = self._smooth_open
                self.fps = current_fps
                self.pip_rgb_bytes = rgb_bytes

        # Cleanup
        if self.cap:
            self.cap.release()

    def _draw_pip_telemetry(self, frame, bbox, pts, smile_val, open_val, shape):
        """Draws visual HUD landmarks and telemetry meters directly on camera frame."""
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 1)

        for i, pt in enumerate(pts):
            color = (0, 255, 0) if i < 2 else ((0, 200, 255) if i == 2 else (255, 0, 200))
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, color, -1)

        cv2.line(frame, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(pts[3][0]), int(pts[3][1])), (int(pts[4][0]), int(pts[4][1])), (255, 0, 200), 2)

        color_map = {
            "NORMAL": (255, 210, 0),
            "FLAT": (128, 0, 255),
            "NARROW": (140, 255, 0)
        }
        bcolor = color_map.get(shape, (255, 255, 255))
        cv2.rectangle(frame, (10, 10), (240, 42), (20, 20, 20), -1)
        cv2.putText(frame, f"MORPH: {shape}", (16, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, bcolor, 2)

        cv2.putText(frame, f"Smile: {int(smile_val * 100)}%", (12, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (125, 52), (255, 68), (40, 40, 40), -1)
        cv2.rectangle(frame, (125, 52), (125 + int(130 * smile_val), 68), (128, 0, 255), -1)

        cv2.putText(frame, f"Open:  {int(open_val * 100)}%", (12, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (125, 75), (255, 91), (40, 40, 40), -1)
        cv2.rectangle(frame, (125, 75), (125 + int(130 * open_val), 91), (140, 255, 0), -1)

    def get_state(self):
        """Returns the thread-safe copy of latest vision outputs."""
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
