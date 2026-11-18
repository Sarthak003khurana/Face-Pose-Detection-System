"""Unit tests for One-Euro temporal filter and 3D solvePnP head pose estimation."""

import cv2
import numpy as np
import pytest

from game.filters import OneEuroFilter
from game.pose_3d import HeadPose3D, MODEL_3D_POINTS, LANDMARK_3D_INDICES


def test_one_euro_filter_smooths_stationary_jitter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    rng = np.random.default_rng(42)
    raw = rng.normal(0.0, 0.25, size=60)
    filtered = [f.filter(val, timestamp=i * (1.0 / 30.0)) for i, val in enumerate(raw)]
    # Output variance must be significantly lower than raw noise variance
    assert np.std(filtered[10:]) < np.std(raw[10:]) * 0.6


def test_one_euro_filter_tracks_fast_step_quickly():
    f = OneEuroFilter(min_cutoff=1.5, beta=0.25)
    t = 0.0
    for _ in range(10):
        f.filter(0.0, timestamp=t)
        t += 1.0 / 30.0
    # Sudden jump to 1.0 should converge rapidly within ~5 frames
    out = 0.0
    for _ in range(6):
        out = f.filter(1.0, timestamp=t)
        t += 1.0 / 30.0
    assert out > 0.70


def test_one_euro_filter_reset():
    f = OneEuroFilter()
    assert f.filter(5.0, timestamp=0.0) == pytest.approx(5.0)
    f.reset()
    assert f.filter(-2.0, timestamp=1.0) == pytest.approx(-2.0)


def test_head_pose_3d_solvepnp_frontal_and_axes():
    pose = HeadPose3D()
    w, h = 640, 480
    cam_matrix = pose._camera_matrix(w, h)
    rvec_gt = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    tvec_gt = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    proj_2d, _ = cv2.projectPoints(MODEL_3D_POINTS, rvec_gt, tvec_gt, cam_matrix, dist)
    proj_2d = proj_2d.reshape(-1, 2)

    # Populate synthetic 468-point array with the 6 canonical 3D projections
    mesh_pts = np.zeros((468, 2), dtype=np.float32)
    for pt_2d, lm_idx in zip(proj_2d, LANDMARK_3D_INDICES):
        mesh_pts[lm_idx] = pt_2d

    res = pose.estimate(mesh_pts, w, h)
    assert res is not None
    roll_deg, yaw_deg, pitch_deg, rvec, tvec, cm = res
    assert abs(yaw_deg) < 5.0
    assert abs(roll_deg) < 5.0

    # Ensure 3D axis projection draws cleanly without error
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    HeadPose3D.draw_axes(canvas, mesh_pts, rvec, tvec, cm)
    assert canvas.sum() > 0
