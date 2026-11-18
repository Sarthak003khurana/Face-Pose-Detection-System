"""Unit tests for ONNX model loading, 3-step calibration, hysteresis, and triggers."""

import os
import numpy as np
import pytest

from game.expression_worker import SCRFDDetector, FaceMesh468, ExpressionWorker


def test_onnx_models_load_and_run_on_blank_frame():
    scrfd_path = "Face_Pose/models/scrfd_500m_kps.onnx"
    mesh_path = "Face_Pose/models/face_mesh.onnx"
    assert os.path.exists(scrfd_path)
    assert os.path.exists(mesh_path)

    det = SCRFDDetector(scrfd_path)
    mesh = FaceMesh468(mesh_path)

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    bboxes, kpss = det.detect(blank)
    assert len(bboxes) == 0

    pts = mesh.predict(blank, np.array([60.0, 40.0, 120.0, 120.0]))
    assert pts is not None and pts.shape == (468, 2)


def test_hysteresis_transitions_cleanly():
    w = ExpressionWorker()
    assert w._active_shape == "NORMAL"

    # Engage FLAT on smile
    w._update_shape_hysteresis(smile=0.65, open_m=0.05)
    assert w._active_shape == "FLAT"

    # Hold FLAT inside hysteresis band (between 0.28 and 0.42)
    w._update_shape_hysteresis(smile=0.35, open_m=0.05)
    assert w._active_shape == "FLAT"

    # Release back to NORMAL below exit threshold
    w._update_shape_hysteresis(smile=0.20, open_m=0.05)
    assert w._active_shape == "NORMAL"

    # Engage NARROW on open mouth
    w._update_shape_hysteresis(smile=0.05, open_m=0.70)
    assert w._active_shape == "NARROW"


def test_three_step_calibration_updates_baselines_and_spans():
    w = ExpressionWorker()

    # Step 1: Neutral samples (12-tuple)
    neutral = np.tile(
        np.array([0.60, 0.012, 0.015, 0.025, 0.210, 0.930, 0.275, 0.29, 0.28, 0.01, 0.5, 1.0]),
        (15, 1),
    )
    w._apply_calibration_samples("NEUTRAL", neutral)
    assert w.base_mouth_w_ratio == pytest.approx(0.60)
    assert w.base_brow_h == pytest.approx(0.275)
    assert w.base_left_ear == pytest.approx(0.29)

    # Step 2: Smile samples
    smile = np.tile(
        np.array([0.72, 0.055, 0.020, 0.030, 0.215, 0.935, 0.275, 0.28, 0.28, 0.0, 0.0, 0.0]),
        (15, 1),
    )
    w._apply_calibration_samples("SMILE", smile)
    assert 0.045 <= w.smile_width_span <= 0.14
    assert 0.024 <= w.smile_lift_span <= 0.075

    # Step 3: Open-mouth samples
    open_m = np.tile(
        np.array([0.54, 0.005, 0.095, 0.175, 0.320, 1.060, 0.275, 0.28, 0.28, 0.0, 0.0, 0.0]),
        (15, 1),
    )
    w._apply_calibration_samples("OPEN", open_m)
    assert 0.042 <= w.open_inner_span <= 0.14
    assert 0.060 <= w.open_outer_span <= 0.16


def test_sensitivities_and_trigger_consumption():
    w = ExpressionWorker()
    w.set_sensitivities(smile=1.5, open_m=0.2, steer=3.5)
    s = w.get_sensitivities()
    assert s["smile_sensitivity"] == pytest.approx(1.5)
    assert s["open_sensitivity"] == pytest.approx(0.5)   # Clamped min
    assert s["steer_sensitivity"] == pytest.approx(2.0)  # Clamped max

    w._jump_pending = True
    w._wink_pending = True
    st1 = w.get_state(consume_triggers=True)
    assert st1["jump_triggered"] is True
    assert st1["wink_triggered"] is True
    st2 = w.get_state(consume_triggers=True)
    assert st2["jump_triggered"] is False
    assert st2["wink_triggered"] is False
