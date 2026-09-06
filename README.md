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

### 4. False-Positive Blink Detection ("Anti-Midas Touch")
- **6-Point Eye Aspect Ratio (EAR)**: Computes vertical-to-horizontal eyelid distance ratio for each eye independently.
- **Bilateral Blink Suppression**: Natural involuntary blinks (both eyes closed simultaneously) are automatically ignored.
- **Sustained Frame Threshold**: Clicks only fire after a conscious single-eye wink held for at least 5 consecutive frames (~160 ms at 30 FPS).
- **Hysteresis Re-arming**: Prevents stuttering double-clicks during a sustained wink.

---

## Controls & Hotkeys

| Key / Action | Function |
| :--- | :--- |
| **Look around** | Cursor glides across both displays following eyes and head. |
| `c` | **Re-Center Calibration**: Look at the camera/center of right monitor and press `c` to re-zero anchor. |
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
  |-- eye_tracking_mouse.py     # Main standalone controller
  |-- face_landmarker.task      # MediaPipe vision model bundle (offline ready)
  |-- models/
  |   +-- face_landmarker.task  # Local model cache
  |-- requirements.txt          # Python dependencies
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

---

## Future Enhancement Roadmap

- [ ] **Dwell Clicking**: Hover on a target for 0.8s with a circular countdown ring to trigger click without winking.
- [ ] **Edge-Scroll Gesture Detection**: Looking at the extreme top/bottom edges of the screen triggers smooth document scrolling.
- [ ] **Interactive 5-Point Calibration Wizard**: Fullscreen overlay showing 5 calibration targets for polynomial gaze warping.
- [ ] **System Tray Minimization**: Run in background with global hotkeys and tray icon.
