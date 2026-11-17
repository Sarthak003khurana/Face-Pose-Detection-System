# Face Pose & Expression Detection System

A real-time computer vision system combining head pose estimation (Roll, Pitch, Yaw), facial expression recognition (Smile & Mouth-Open), and an interactive game: **Shape-Shift Racer**!

---

## 🎮 Featured Game: Shape-Shift Racer

An interactive 2.5D arcade game where **your face controls the car**:
- **Smile Big** $\rightarrow$ Car morphs into an ultra-low **FLAT** shape to slide underneath low-hanging barriers.
- **Open Mouth ("O" / Surprise)** $\rightarrow$ Car morphs into a tall **NARROW** blade to squeeze through tight vertical pillar slits.
- **Neutral Face** $\rightarrow$ Standard balanced **CRUISER** shape.
- **Head Turn / Tilt (Yaw & Roll)** $\rightarrow$ Steers the vehicle left $\leftrightarrow$ right across lanes to dodge solid road blocks.

```
       [Webcam Video] ──> [SCRFD Face & Landmark Detector]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
         [Head Tilt / Turn]                  [Facial Expressions]
                  │                                   │
                  ▼                                   ▼
         Steers Car Left/Right           Morphs Shape (Flat / Narrow)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                      [Pygame 60 FPS Game Loop]
                      ├── 2.5D Perspective Road
                      ├── Procedural Sound Effects
                      ├── Approaching Obstacles & Gaps
                      └── Live Webcam PiP Overlay
```

### Launch the Game:
```bash
python run_game.py
```

*Note: Keyboard controls (`A/D` or Arrows to steer, `Down` to flatten, `Up` to narrow, `Space` to start/pause, `C` to recalibrate) are always active as backup.*

---

## 🔬 Core Pose & Expression Estimation

The repository also includes standalone real-time research and benchmarking scripts:
1. **Pose Detection**:
   - **Method 1 (Pixel Ratio Heuristics)**: Fast orientation tracking based on relative landmark positions.
   - **Method 2 (Angular Degrees)**: Roll-normalized angles ($-90^\circ$ to $+90^\circ$) calibrated according to NIST face quality assessment guidelines.
2. **Smile Detection**: Mouth-width-to-interpupillary distance ratio analysis.
3. **Framing & Quality Diagnostics**: Centering check and aspect-ratio validation.

### Supported Face Detectors:
| Detector | Script | Framework |
| :--- | :--- | :--- |
| **SCRFD** | `Face_Pose/pose_detection_scrfd.py` | OpenCV DNN + ONNX (`500m`, `2.5g`, `10g`) |
| **RetinaFace** | `Face_Pose/pose_detection_retinaface.py` | TensorFlow SavedModel & ONNX |
| **MTCNN** | `Face_Pose/pose_detection_mtcnn.py` | Pure Python + TensorFlow |

---

## 📁 Repository Structure

```text
Face-Pose-Detection-System/
├── run_game.py                 # Main game executable
├── requirements.txt            # Dependencies (pygame, opencv-python, numpy)
├── README.md                   # Project documentation
├── .gitignore                  # Git rules
├── game/                       # Shape-Shift Racer game engine
│   ├── config.py               # Constants, speeds, and thresholds
│   ├── expression_worker.py    # Threaded camera & face classifier
│   ├── car.py                  # Morphing vehicle & vector tweening
│   ├── obstacles.py            # Low beams, pillars, roadblocks, and hitboxes
│   ├── road.py                 # 2.5D retro perspective road rendering
│   └── sound_manager.py        # Procedural in-memory 8-bit sound effects
└── Face_Pose/                  # Standalone detection demos & models
    ├── README.md
    ├── pose_detection_scrfd.py
    ├── pose_detection_retinaface.py
    ├── pose_detection_mtcnn.py
    ├── models/                 # Pretrained ONNX & SavedModel weights
    └── mtcnn/                  # MTCNN network implementation
```

---

## 🚀 Installation & Quickstart

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Play the Game**:
   ```bash
   python run_game.py
   ```

3. **Run Standalone Face Pose Demos**:
   ```bash
   cd Face_Pose
   python pose_detection_scrfd.py
   # or
   python pose_detection_retinaface.py
   # or
   python pose_detection_mtcnn.py
   ```
