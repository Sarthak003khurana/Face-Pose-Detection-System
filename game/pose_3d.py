"""
True 3D Head Pose Estimation via OpenCV solvePnP on 468-Point FaceMesh Landmarks.
Uses OpenCV's native camera coordinate convention (+X right, +Y down, +Z forward)
so frontal faces sit cleanly near (0, 0, 0) degrees with positive depth (tvec_z > 0),
completely eliminating ±180° gimbal-lock flips and unstable steering jumps.
"""

import math
import cv2
import numpy as np


# 6 Canonical 3D Anthropometric Face Model Points in OpenCV coordinates (mm):
#   +X = Right, +Y = Down, +Z = Deeper into the face (away from camera)
# Mapped to exact MediaPipe 468-point landmark indices:
#   1   = Nose tip
#   152 = Chin bottom
#   33  = Left eye outer corner
#   263 = Right eye outer corner
#   61  = Left mouth corner
#   291 = Right mouth corner
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,   0.0),     # Nose tip (1) — closest to camera
    (0.0,   63.6,  12.5),     # Chin (152) — below nose (+Y)
    (-43.3, -32.7, 26.0),     # Left eye outer corner (33) — left (-X), above (-Y)
    (43.3,  -32.7, 26.0),     # Right eye outer corner (263) — right (+X), above (-Y)
    (-28.9,  28.9, 24.1),     # Left mouth corner (61) — left (-X), below (+Y)
    (28.9,   28.9, 24.1),     # Right mouth corner (291) — right (+X), below (+Y)
], dtype=np.float64)

LANDMARK_3D_INDICES = [1, 152, 33, 263, 61, 291]


class HeadPose3D:
    """Estimates stable 3D Roll, Yaw, and Pitch using cv2.solvePnP."""
    def __init__(self):
        self._rvec = None
        self._tvec = None

    @staticmethod
    def _camera_matrix(frame_w, frame_h):
        focal = float(frame_w)
        cx = frame_w * 0.5
        cy = frame_h * 0.5
        return np.array([
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def reset(self):
        self._rvec = None
        self._tvec = None

    def estimate(self, mesh_pts_468, frame_w, frame_h):
        """
        Estimates 3D head orientation from (468, 2) image landmarks.
        Returns: (roll_deg, yaw_deg, pitch_deg, rvec, tvec, cam_matrix)
                 or None if solvePnP fails or produces an invalid behind-camera pose.
        """
        if mesh_pts_468 is None or len(mesh_pts_468) < 468:
            return None

        image_pts = np.array(
            [mesh_pts_468[idx] for idx in LANDMARK_3D_INDICES],
            dtype=np.float64
        )
        cam_matrix = self._camera_matrix(frame_w, frame_h)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Initialize frontal guess in front of camera (+Z = 600mm) so solver never flips behind camera
        if self._rvec is None or self._tvec is None or self._tvec[2, 0] <= 50.0:
            init_rvec = np.zeros((3, 1), dtype=np.float64)
            init_tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
        else:
            init_rvec = self._rvec.copy()
            init_tvec = self._tvec.copy()

        ok, rvec, tvec = cv2.solvePnP(
            MODEL_3D_POINTS,
            image_pts,
            cam_matrix,
            dist_coeffs,
            rvec=init_rvec,
            tvec=init_tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or tvec[2, 0] <= 50.0:
            self.reset()
            return None

        self._rvec = rvec
        self._tvec = tvec

        # Extract smooth, singularity-free Euler angles directly from rotation matrix R
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        if sy > 1e-6:
            pitch_rad = math.atan2(R[2, 1], R[2, 2])
            yaw_rad = math.atan2(-R[2, 0], sy)
            roll_rad = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch_rad = math.atan2(-R[1, 2], R[1, 1])
            yaw_rad = math.atan2(-R[2, 0], sy)
            roll_rad = 0.0

        pitch_deg = float(np.clip(math.degrees(pitch_rad), -75.0, 75.0))
        # Negate yaw so turning head RIGHT produces positive (+) yaw matching screen +X
        yaw_deg = float(np.clip(-math.degrees(yaw_rad), -75.0, 75.0))
        roll_deg = float(np.clip(math.degrees(roll_rad), -75.0, 75.0))

        return roll_deg, yaw_deg, pitch_deg, rvec, tvec, cam_matrix

    @staticmethod
    def draw_axes(frame, mesh_pts_468, rvec, tvec, cam_matrix, axis_len=45.0):
        """Projects 3D X (Red=Right), Y (Green=Up), Z (Cyan=Forward out of nose) axes."""
        if rvec is None or tvec is None:
            return
        axis_3d = np.array([
            (axis_len, 0.0, 0.0),    # X (right)        -> Red
            (0.0, -axis_len, 0.0),   # -Y (up)          -> Green
            (0.0, 0.0, -axis_len),   # -Z (out of nose) -> Cyan
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2)

        nose = (int(mesh_pts_468[1][0]), int(mesh_pts_468[1][1]))
        px = (int(proj[0][0]), int(proj[0][1]))
        py = (int(proj[1][0]), int(proj[1][1]))
        pz = (int(proj[2][0]), int(proj[2][1]))

        cv2.line(frame, nose, px, (40, 60, 255), 2, cv2.LINE_AA)   # X - Red
        cv2.line(frame, nose, py, (40, 255, 80), 2, cv2.LINE_AA)   # Y - Green
        cv2.line(frame, nose, pz, (255, 220, 40), 2, cv2.LINE_AA)  # Z - Cyan
