"""
True 3D Head Pose Estimation via OpenCV solvePnP on 468-Point FaceMesh Landmarks.
Computes 6-DOF rotation/translation vectors, Euler angles (Roll, Yaw, Pitch in degrees),
and projects 3D RGB coordinate axes onto the webcam preview frame.
"""

import cv2
import numpy as np


# 6 Canonical 3D Anthropometric Face Model Points (mm, right-handed face coordinates)
# Mapped to exact MediaPipe 468-point landmark indices:
#   1   = Nose tip
#   152 = Chin bottom
#   33  = Left eye outer corner
#   263 = Right eye outer corner
#   61  = Left mouth corner
#   291 = Right mouth corner
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip (1)
    (0.0,   -63.6, -12.5),     # Chin (152)
    (-43.3,  32.7, -26.0),     # Left eye outer corner (33)
    (43.3,   32.7, -26.0),     # Right eye outer corner (263)
    (-28.9, -28.9, -24.1),     # Left mouth corner (61)
    (28.9,  -28.9, -24.1),     # Right mouth corner (291)
], dtype=np.float64)

LANDMARK_3D_INDICES = [1, 152, 33, 263, 61, 291]


class HeadPose3D:
    """Estimates 3D Roll, Yaw, and Pitch using cv2.solvePnP."""
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

    def estimate(self, mesh_pts_468, frame_w, frame_h):
        """
        Estimates 3D head orientation from (468, 2) image landmarks.
        Returns: (roll_deg, yaw_deg, pitch_deg, rvec, tvec, cam_matrix)
                 or None if solvePnP fails.
        """
        if mesh_pts_468 is None or len(mesh_pts_468) < 468:
            return None

        image_pts = np.array(
            [mesh_pts_468[idx] for idx in LANDMARK_3D_INDICES],
            dtype=np.float64
        )
        cam_matrix = self._camera_matrix(frame_w, frame_h)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        use_guess = self._rvec is not None and self._tvec is not None
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_3D_POINTS,
            image_pts,
            cam_matrix,
            dist_coeffs,
            rvec=self._rvec.copy() if use_guess else None,
            tvec=self._tvec.copy() if use_guess else None,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self._rvec = None
            self._tvec = None
            return None

        self._rvec = rvec
        self._tvec = tvec

        # Convert rotation vector to 3x3 rotation matrix and decompose to Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Normalize angles to [-90, 90] cleanly
        pitch_deg = float(np.clip(angles[0], -90.0, 90.0))
        yaw_deg = float(np.clip(angles[1], -90.0, 90.0))
        roll_deg = float(np.clip(angles[2], -90.0, 90.0))

        # Wrap inverted pitch/roll when RQDecomp returns ~±180 around optical axis
        if pitch_deg > 90.0:
            pitch_deg -= 180.0
        elif pitch_deg < -90.0:
            pitch_deg += 180.0

        return roll_deg, yaw_deg, pitch_deg, rvec, tvec, cam_matrix

    @staticmethod
    def draw_axes(frame, mesh_pts_468, rvec, tvec, cam_matrix, axis_len=45.0):
        """Projects 3D X (Red), Y (Green), Z (Blue) pose axes from the nose tip."""
        if rvec is None or tvec is None:
            return
        axis_3d = np.array([
            (axis_len, 0.0, 0.0),   # X (right) -> Red
            (0.0, axis_len, 0.0),   # Y (up)    -> Green
            (0.0, 0.0, axis_len),   # Z (front) -> Blue/Cyan
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
        cv2.line(frame, nose, pz, (255, 220, 40), 2, cv2.LINE_AA)  # Z - Cyan/Blue
