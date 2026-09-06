# EyeControl: Real-Time Webcam Eye-Tracking Mouse

An advanced, webcam-only computer vision mouse controller written in Python using **OpenCV**, **MediaPipe Face Mesh (478 Iris Landmarks)**, and **Windows Native Multi-Monitor Actuation**.

This project provides hands-free mouse navigation across single and dual-monitor workstation environments with sub-pixel stabilization and blink-based click control.

---

## Key Architectural Capabilities

### 1. Dual-Monitor Spanning with Right-Monitor Camera Anchor
- **Physical Layout Auto-Detection**:
  - **Left Monitor**: 1600 x 900 at (0, 0)
  - **Right Monitor (Webcam Mount)**: 1536 x 864 at (1600, -94)
  - **Total Virtual Desktop**: 3136 x 994 pixels
- **Anchor Center**: When looking forward at the webcam on top of the Right Monitor, resting gaze is calibrated to the center of the Right Monitor (X ~ 2368, Y ~ 338).
- **Seamless Monitor Crossing**: Glancing or turning your head towards the left smoothly carries the cursor past the border (X = 1600) onto the Left Monitor.

### 2. Multi-Stage Cursor Stabilization (Jitter & Tremor Elimination)
Human eyes have involuntary physiological tremors (30-80 Hz) and micro-saccades. To prevent shaking without causing input lag, EyeControl uses a 3-stage stabilization pipeline:
1. **Stage 1: Temporal 3-Frame Median Outlier Filter**:
   Filters raw iris and facial landmarks to reject single-frame detection flickers before coordinate mapping.
2. **Stage 2: Dual-Axis 1-Euro Filter (Casiez et al., CHI 2012)**:
   Dynamically modulates low-pass cutoff frequency based on velocity:
   - At low speeds (hovering/aiming): Cutoff drops to 0.35 - 0.55 Hz (eliminates jitter completely).
   - At high speeds (glancing across monitors): Cutoff increases dynamically (zero lag).
3. **Stage 3: Dynamic Fixation Lock (Adaptive Deadzone)**:
   When holding gaze still, the engine engages `[FIXATION LOCKED]`, holding the cursor completely still for clicking small buttons and links.

### 3. Fused Iris Gaze & Facial Head Yaw (HEAD_POSE_ASSIST)
- Moving across two wide screens (3136 px total width) using pupils alone causes eye strain.
- EyeControl fuses relative iris movement with **facial head yaw** (tracking nose tip `1` against cheek contours `234` and `454`):
  `Total_dx = Iris_dx + 0.70 * HeadYaw_dx`
- A natural slight turn of the head towards the left monitor glides the cursor across screens effortlessly.

### 4. Neural Facial Blendshapes ("Anti-Midas" Click Engine)
- **Deep Learning Probabilities**: Replaces heuristic Euclidean distance EAR with MediaPipe's neural blendshapes (`eyeBlinkLeft` and `eyeBlinkRight` scores from 0.0 to 1.0, with automatic EAR fallback).
- **Bilateral Blink Suppression**: Natural involuntary blinks (both eyes closed simultaneously) are automatically ignored.
- **Sustained Frame Threshold**: Clicks only fire after a conscious single-eye wink held for at least 5 consecutive frames (~160 ms at 30 FPS).
- **Hysteresis Re-arming**: Prevents stuttering double-clicks during a sustained wink.

### 5. Degree-2 Polynomial Ridge Regression & Posture Invariance
- **Closed-Form Non-Linear Mapping**: Maps non-linear ocular curvature across both displays via regularized least squares $\mathbf{W} = (\mathbf{\Phi}^T \mathbf{\Phi} + \alpha \mathbf{I})^{-1} \mathbf{\Phi}^T \mathbf{Y}$.
- **Posture & Depth Normalization**: Tracks 3D Inter-Ocular Distance ($d_{IOD} = \|\mathbf{p}_{33} - \mathbf{p}_{263}\|_2$) to compensate for slouching, leaning forward, or distance changes.
- **Persistent Profile**: Calibration is saved to `calibration_profile.json` so you do not need to re-calibrate on every launch.

---

## Controls & Hotkeys

| Key / Action | Function |
| :--- | :--- |
| **Look around** | Cursor glides across both displays following eyes and head. |
| `k` | **Interactive 9-Point Calibration Wizard**: Fullscreen targets across monitors to train Degree-2 Ridge Model. |
| `e` | **Accuracy Benchmark Harness**: Empirical evaluation measuring Mean Radial Error (MRE in pixels). |
| `c` | **Re-Center Calibration**: Look at the camera/center of right monitor and press `c` (Heuristic fallback). |
| `s` | **Cycle Stability Mode**: `SMOOTH` (balanced) -> `ULTRA-STABLE` (rock-solid) -> `RESPONSIVE` (fast). |
| `+` / `=` | **Increase Fixation Deadzone** (+1 px radius). |
| `-` / `_` | **Decrease Fixation Deadzone** (-1 px radius). |
| `i` | **Toggle Invert X-Axis**: Reverses horizontal tracking direction if needed. |
| `1` | **Snap to Left Monitor Center** (800, 450). |
| `2` | **Snap to Right Monitor Center** (2368, 338). |
| **Left Eye Wink** (hold ~0.2s) | **Left Mouse Click**. |
| **Right Eye Wink** (hold ~0.2s) | **Right Mouse Click**. |
| `m` | **Toggle Mouse Control** On / Off (preview mode). |
| `q` or `ESC` | **Quit** application. |

---

## Project Structure

```text
E:\EyeControl\
  |-- eye_tracking_mouse.py     # Main standalone controller & runtime loop
  |-- gaze_calibration.py      # Degree-2 Polynomial Ridge & PostureNormalizer
  |-- evaluate_accuracy.py      # Ground-truth accuracy benchmark harness
  |-- calibration_profile.json  # Persisted calibration weights (auto-generated)
  |-- tests/
  |   |-- test_gaze_calibration.py
  |   +-- test_tracker_integration.py
  |-- models/
  |   +-- face_landmarker.task  # MediaPipe vision model bundle
  |-- requirements.txt          # Python dependencies (zero bloat, pure NumPy)
  |-- run.bat                   # One-click Windows launcher
  +-- README.md                 # Project documentation
```

---

## Quickstart

### Method 1: One-Click Launcher (Windows)
Double-click `run.bat` in `E:\EyeControl\`.

### Method 2: Command Line
```bash
cd E:\EyeControl
pip install -r requirements.txt
python eye_tracking_mouse.py
```

### Empirical Accuracy Benchmark
To measure empirical pixel error before and after calibration:
```bash
python evaluate_accuracy.py --baseline  # Benchmark using baseline linear heuristic
python evaluate_accuracy.py             # Benchmark using trained polynomial ridge model
```

---

## Future Enhancement Roadmap

- [x] **Interactive Multi-Point Calibration Wizard**: Fullscreen 9-point targets for polynomial ridge gaze mapping.
- [x] **Neural Facial Blendshapes**: MediaPipe deep learning blendshape integration for false-positive-free clicking.
- [x] **Posture & Depth Normalization**: 3D Inter-Ocular Distance scaling.
- [x] **Empirical Accuracy Benchmark Harness**: Ground-truth pixel error evaluation tool.
- [ ] **Dwell Clicking**: Hover on a target for 0.8s with a circular countdown ring to trigger click without winking.
- [ ] **Edge-Scroll Gesture Detection**: Looking at the extreme top/bottom edges of the screen triggers smooth document scrolling.
- [ ] **System Tray Minimization**: Run in background with global hotkeys and tray icon.

