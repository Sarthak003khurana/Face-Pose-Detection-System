# Face Pose Detection System

A real-time head pose estimation and facial expression (smile) detection system based on 2D face detectors.

Estimates 3D head orientation (**Roll**, **Pitch**, and **Yaw**) and detects facial smiles in real time from a live webcam feed or video file using lightweight, fast geometric formulas applied to 5 facial landmarks (eyes, nose, and mouth corners).

---

## Features

- **Head Pose Estimation**:
  - **Method 1 (Pixel Ratio Heuristics)**: Fast estimation based on relative landmark distances.
  - **Method 2 (Angular Degrees)**: Roll-normalized angles in degrees ($-90^\circ$ to $+90^\circ$) calibrated according to face image quality standards.
- **Smile Detection**: Expression classification based on the mouth-to-interpupillary width ratio.
- **Framing & Centering Diagnostics**: Automatic face selection (closest to frame center) and face framing validation.
- **Multiple Face Detector Backends**:
  - **MTCNN**: Multi-task Cascaded CNN detector with TensorFlow.
  - **RetinaFace**: MobileNetV2-backed RetinaFace model (SavedModel & ONNX).
  - **SCRFD**: High-efficiency ONNX face detector running via OpenCV DNN (`500m`, `2.5g`, `10g`).

---

## Directory Structure

```text
Face-Pose-Detection-System/
├── README.md
├── .gitignore
└── Face_Pose/
    ├── README.md
    ├── LICENSE.md
    ├── pose_demo.png
    ├── pose_detection_mtcnn.py
    ├── pose_detection_retinaface.py
    ├── pose_detection_scrfd.py
    ├── models/
    │   ├── scrfd_500m_kps.onnx
    │   ├── scrfd_2.5g_kps.onnx
    │   ├── scrfd_10g_kps.onnx
    │   ├── tf-serving-retinaface_mbv2.onnx
    │   └── tf_retinaface_mbv2/
    └── mtcnn/
```

---

## Quickstart

Navigate to the `Face_Pose` directory:

```bash
cd Face_Pose
```

Run any of the detection scripts:

```bash
# Using MTCNN
python pose_detection_mtcnn.py

# Using RetinaFace
python pose_detection_retinaface.py

# Using SCRFD
python pose_detection_scrfd.py
```

Press `q` to exit the live video feed.
