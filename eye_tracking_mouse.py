"""
================================================================================
Real-Time Dual-Monitor Eye-Tracking Mouse Controller (Webcam on Right Monitor)
================================================================================
Architecture Overview:
  1. Multi-Monitor Display Management:
     - Automatically detects dual-monitor geometry via Windows Win32 APIs:
       * Monitor 0 (Left):  1600 x 900  at (0, 0)
       * Monitor 1 (Right): 1536 x 864  at (1600, -94)
       * Virtual Desktop:   3136 x 994  spanning (0, -94) to (3136, 900)
     - Camera Anchor: Anchors resting forward gaze to the RIGHT monitor (where the
       webcam is physically mounted).
  2. Multi-Screen Gaze & Head Pose Fusion:
     - Dual-monitor spanning requires wider angular coverage than a single screen.
     - Fuses relative corneal iris tracking with facial head yaw/pitch (nose vs. cheek contours).
     - Looking straight forward at the camera = Center of Right Monitor (X ≈ 2368).
     - Glancing or turning head left = Glides smoothly across the seam (X = 1600) onto the Left Monitor!
  3. Native Multi-Monitor Cursor Actuation:
     - Uses Windows user32.SetCursorPos & user32.mouse_event for 0ms latency,
       completely avoiding PyAutoGUI FailSafe exceptions on negative coordinate monitors.
  4. Jitter Reduction:
     - Exponential Moving Average (EMA) filter: S_t = alpha * X_t + (1 - alpha) * S_{t-1}.
     - Spatial deadzone to eliminate micro-saccadic tremor when fixating on buttons.
  5. Blink Detection via 6-Point EAR (Soukupová & Čech, 2016):
     - Left-eye wink -> Left click.
     - Right-eye wink -> Right click.
  6. False-Positive Prevention (The "Midas Touch" Problem):
     - Sustained Frame Buffer: Requires 4-6 consecutive closed frames (~160ms).
     - Bilateral Blink Suppression: Natural simultaneous blinks are ignored.
     - Hysteresis Re-arming: Latches click state until eye re-opens past threshold.

Dependencies:
  pip install opencv-python mediapipe pyautogui numpy
================================================================================
"""

import os
import sys
import time
import math
import ctypes
from ctypes import wintypes
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np

# PyAutoGUI import with fail-safe safety for non-Windows fallback
try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disabled because virtual multi-monitor desktop crosses (0, 0)
    pyautogui.PAUSE = 0.0
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# MediaPipe for Face Mesh & Iris landmark detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ==============================================================================
# 1. SYSTEM CONFIGURATION (TUNE ALL PARAMETERS HERE)
# ==============================================================================
@dataclass
class EyeTrackerConfig:
    """
    Tunable configuration parameters for dual-monitor gaze tracking and blink detection.
    """
    # --------------------------------------------------------------------------
    # Multi-Monitor Setup
    # --------------------------------------------------------------------------
    ENABLE_DUAL_MONITORS: bool = True       # Spans cursor across both displays
    CAMERA_ON_MONITOR: str = "right"        # "right" or "left" - physical webcam mounting monitor
    HEAD_POSE_ASSIST: bool = True           # Fuses head yaw with iris tracking for natural multi-screen glance
    HEAD_YAW_WEIGHT: float = 0.70           # Contribution of head turning (0.0 = pure iris, 1.0 = heavy head assist)
    
    # --------------------------------------------------------------------------
    # Camera & Capture
    # --------------------------------------------------------------------------
    CAMERA_INDEX: int = 0                   # Default webcam index
    FRAME_WIDTH: int = 640                  # Capture width (640x480 gives optimal 60+ FPS)
    FRAME_HEIGHT: int = 480                 # Capture height
    MIRROR_WEBCAM: bool = True              # Looking right moves cursor right

    # --------------------------------------------------------------------------
    # Blink Detection & "Midas Touch" Prevention
    # --------------------------------------------------------------------------
    EAR_THRESHOLD: float = 0.18             # Eye Aspect Ratio threshold (below = eye closed)
    EAR_HYSTERESIS_OFFSET: float = 0.03     # Re-arm threshold = EAR_THRESHOLD + OFFSET (e.g. 0.21)
    CONSECUTIVE_FRAMES_TRIGGER: int = 5     # Consecutive closed frames before click (5 @ 30 FPS ~ 160ms)
    SUPPRESS_BILATERAL_BLINKS: bool = True  # Ignore clicks when BOTH eyes close (natural blink)

    # --------------------------------------------------------------------------
    # Multi-Stage Stabilization & Tremor Elimination (1€ Filter + Fixation Lock)
    # --------------------------------------------------------------------------
    STABILIZER_PRESET: str = "SMOOTH"       # "SMOOTH", "ULTRA-STABLE", or "RESPONSIVE"
    ONE_EURO_MIN_CUTOFF: float = 0.55       # Low-speed cutoff (Hz) (lower = rock-solid hover)
    ONE_EURO_BETA: float = 0.008            # Dynamic velocity coefficient (higher = faster saccades)
    FIXATION_DEADZONE: float = 7.0          # Pixel radius to lock cursor during target fixation

    # --------------------------------------------------------------------------
    # Gaze-to-Screen Mapping Sensitivity & Inversion
    # --------------------------------------------------------------------------
    SENSITIVITY_X: float = 3.2              # Horizontal reach gain (allows reaching far edges of both monitors)
    SENSITIVITY_Y: float = 2.8              # Vertical reach gain
    INVERT_X: bool = False                  # False = Natural (Turn Left -> Moves to Left Monitor). True = Inverted
    INVERT_Y: bool = False                  # False = Natural (Tilt Up -> Moves Up)

    # --------------------------------------------------------------------------
    # Operational Modes
    # --------------------------------------------------------------------------
    ENABLE_MOUSE_CONTROL: bool = True       # True = physically move cursor; False = preview HUD
    SHOW_DEBUG_HUD: bool = True             # Overlay real-time EAR meters and active monitor indicator
    MODEL_TASK_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")  # MediaPipe vision model bundle path


# ==============================================================================
# 2. MULTI-MONITOR DISPLAY GEOMETRY MANAGER
# ==============================================================================
@dataclass
class MonitorInfo:
    index: int
    name: str
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    center_x: int
    center_y: int
    is_primary: bool


class DisplayManager:
    """
    Automatically detects and manages multi-monitor setups on Windows.
    Enables smooth continuous cursor navigation across both monitors with accurate coordinate projection.
    """

    def __init__(self, camera_on_monitor: str = "right"):
        self.camera_on_monitor = camera_on_monitor.lower()
        self.monitors: List[MonitorInfo] = []
        self.virtual_left = 0
        self.virtual_top = 0
        self.virtual_width = 1920
        self.virtual_height = 1080
        self._detect_monitors()

    def _detect_monitors(self):
        """Queries Windows Win32 APIs for physical monitor rectangles."""
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            self.virtual_left = user32.GetSystemMetrics(76)    # SM_XVIRTUALSCREEN
            self.virtual_top = user32.GetSystemMetrics(77)     # SM_YVIRTUALSCREEN
            self.virtual_width = user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
            self.virtual_height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN

            raw_monitors = []

            def enum_proc(h_monitor, hdc_monitor, lprc_monitor, dw_data):
                rect = lprc_monitor.contents
                raw_monitors.append((rect.left, rect.top, rect.right, rect.bottom))
                return 1

            MONITORENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                wintypes.HMONITOR,
                wintypes.HDC,
                ctypes.POINTER(wintypes.RECT),
                wintypes.LPARAM
            )
            user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(enum_proc), 0)

            # Sort monitors left-to-right by X-coordinate
            raw_monitors.sort(key=lambda m: m[0])

            for idx, (l, t, r, b) in enumerate(raw_monitors):
                w = r - l
                h = b - t
                is_prim = (l == 0 and t == 0)
                name = "LEFT" if idx == 0 else ("RIGHT" if idx == len(raw_monitors) - 1 else f"MON_{idx}")
                self.monitors.append(MonitorInfo(
                    index=idx, name=name, left=l, top=t, right=r, bottom=b,
                    width=w, height=h,
                    center_x=l + w // 2, center_y=t + h // 2,
                    is_primary=is_prim
                ))

        if not self.monitors:
            # Fallback for single monitor or non-Windows
            w, h = (1920, 1080)
            if PYAUTOGUI_AVAILABLE:
                w, h = pyautogui.size()
            self.monitors.append(MonitorInfo(
                index=0, name="PRIMARY", left=0, top=0, right=w, bottom=h,
                width=w, height=h, center_x=w // 2, center_y=h // 2, is_primary=True
            ))
            self.virtual_left = 0
            self.virtual_top = 0
            self.virtual_width = w
            self.virtual_height = h

    def get_anchor_monitor(self) -> MonitorInfo:
        """Returns the monitor where the camera is physically mounted."""
        if len(self.monitors) <= 1:
            return self.monitors[0]
        if self.camera_on_monitor == "right":
            return self.monitors[-1]
        return self.monitors[0]

    def get_monitor_for_x(self, x: int) -> MonitorInfo:
        """Identifies which physical monitor a virtual screen X coordinate falls into."""
        for m in self.monitors:
            if m.left <= x < m.right:
                return m
        return self.monitors[-1] if x >= self.monitors[-1].right else self.monitors[0]


# ==============================================================================
# 3. NATIVE HARDWARE-ACCELERATED MOUSE CONTROLLER
# ==============================================================================
class MouseController:
    """
    Cross-monitor mouse controller. On Windows, uses native user32.SetCursorPos
    and user32.mouse_event to support virtual desktops with negative coordinates
    without PyAutoGUI failsafe crashes.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        if self.is_windows:
            self.user32 = ctypes.windll.user32

    def move(self, x: int, y: int):
        """Moves cursor to virtual desktop coordinate (x, y)."""
        if self.is_windows:
            self.user32.SetCursorPos(int(x), int(y))
        elif PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.moveTo(x, y)
            except Exception:
                pass

    def click(self, button: str = "left"):
        """Performs hardware-level mouse click."""
        if self.is_windows:
            if button == "left":
                self.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                self.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
            elif button == "right":
                self.user32.mouse_event(0x0008, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTDOWN
                self.user32.mouse_event(0x0010, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTUP
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.click(button=button)


# ==============================================================================
# 4. MEDIAPIPE FACE MESH LANDMARK CONSTANTS
# ==============================================================================
class Landmarks:
    """Standard MediaPipe Face Mesh landmark indices with refined irises."""
    # Subject Right Eye (camera frame left in standard view):
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_IRIS_CENTER = 468
    RIGHT_EYE_BOUNDS = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}

    # Subject Left Eye (camera frame right in standard view):
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS_CENTER = 473
    LEFT_EYE_BOUNDS = {"inner": 362, "outer": 263, "top": 386, "bottom": 374}

    # Head Pose Geometry (Nose vs. Facial Contours)
    NOSE_TIP = 1
    FACE_RIGHT_CHEEK = 234  # Subject right side (left on camera)
    FACE_LEFT_CHEEK = 454   # Subject left side (right on camera)
    FOREHEAD = 10
    CHIN = 152


# ==============================================================================
# 5. UNIVERSAL MEDIAPIPE ADAPTER
# ==============================================================================
class FaceMeshAdapter:
    """Universal compatibility wrapper for MediaPipe Tasks and Solutions APIs."""

    def __init__(self, model_asset_path: str = "face_landmarker.task"):
        self.mode = None
        self.detector = None

        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mode = "solutions"
            self.detector = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[MediaPipe Adapter] Initialized via legacy mp.solutions.face_mesh API.")
        elif hasattr(mp, "tasks"):
            self.mode = "tasks"
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python

            if not os.path.exists(model_asset_path):
                print(f"[MediaPipe Adapter] Downloading face_landmarker.task model to '{model_asset_path}'...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                os.makedirs(os.path.dirname(os.path.abspath(model_asset_path)) if os.path.dirname(model_asset_path) else ".", exist_ok=True)
                urllib.request.urlretrieve(url, model_asset_path)
                print(f"[MediaPipe Adapter] Download complete.")

            base_options = mp_python.BaseOptions(model_asset_path=model_asset_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("[MediaPipe Adapter] Initialized via modern mediapipe.tasks.vision.FaceLandmarker API.")
        else:
            raise RuntimeError("Incompatible MediaPipe version.")

    def process(self, frame_bgr: np.ndarray) -> Optional[List[Any]]:
        """Runs face mesh inference and returns 478 normalized landmarks."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.mode == "solutions":
            results = self.detector.process(rgb_frame)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None
        elif self.mode == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.face_landmarks and len(results.face_landmarks) > 0:
                return results.face_landmarks[0]
            return None
        return None

    def close(self):
        if self.detector and hasattr(self.detector, "close"):
            self.detector.close()


# ==============================================================================
# 6. ADAPTIVE 1€ (ONE EURO) FILTER & FIXATION STABILIZER
# ==============================================================================
class OneEuroFilter1D:
    """
    Casiez, Roussel, Vogel (CHI 2012) 1€ Filter.
    Dynamically modulates cutoff frequency based on input velocity:
      - Low velocity (hover/fixation) -> low cutoff (0.3 - 0.6 Hz) -> completely eliminates tremor.
      - High velocity (saccade/glance) -> high cutoff -> zero-latency tracking across monitors.
    """

    def __init__(self, min_cutoff: float = 0.55, beta: float = 0.008, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            self.t_prev = t
            return float(x)

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        # Velocity estimation
        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + self._alpha(self.d_cutoff, dt) * (dx - self.dx_prev)
        self.dx_prev = edx

        # Dynamic cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        x_hat = self.x_prev + alpha * (x - self.x_prev)
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class AdaptiveStabilizer:
    """
    Multi-stage cursor stabilization engine combining:
      1. Dual-axis 1€ (One Euro) adaptive filters (eliminates jitter while maintaining zero lag).
      2. Dynamic Fixation Lock (locks micro-drift when holding gaze on a button/target).
      3. Preset switching: SMOOTH, ULTRA-STABLE, and RESPONSIVE.
    """

    def __init__(
        self,
        min_cutoff: float = 0.55,
        beta: float = 0.008,
        deadzone: float = 7.0,
        preset: str = "SMOOTH"
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.deadzone = deadzone
        self.preset = preset.upper()

        self.fx = OneEuroFilter1D(min_cutoff=self.min_cutoff, beta=self.beta)
        self.fy = OneEuroFilter1D(min_cutoff=self.min_cutoff, beta=self.beta)

        self.locked_x: Optional[float] = None
        self.locked_y: Optional[float] = None
        self.is_fixating: bool = False

    def set_preset(self, preset_name: str):
        """Switches filter tuning presets."""
        self.preset = preset_name.upper()
        if self.preset == "ULTRA-STABLE":
            self.min_cutoff = 0.35
            self.beta = 0.005
            self.deadzone = 10.0
        elif self.preset == "SMOOTH":
            self.min_cutoff = 0.55
            self.beta = 0.008
            self.deadzone = 7.0
        elif self.preset == "RESPONSIVE":
            self.min_cutoff = 0.85
            self.beta = 0.015
            self.deadzone = 4.0

        self.fx.min_cutoff = self.min_cutoff
        self.fx.beta = self.beta
        self.fy.min_cutoff = self.min_cutoff
        self.fy.beta = self.beta
        print(f"[Stabilizer] Mode set to {self.preset} (MinCutoff={self.min_cutoff}, Deadzone={self.deadzone:.1f}px)")

    def cycle_preset(self) -> str:
        """Cycles through available stabilization presets."""
        order = ["SMOOTH", "ULTRA-STABLE", "RESPONSIVE"]
        curr_idx = order.index(self.preset) if self.preset in order else 0
        new_preset = order[(curr_idx + 1) % len(order)]
        self.set_preset(new_preset)
        return self.preset

    def adjust_deadzone(self, delta: float) -> float:
        """Dynamically tunes fixation lock deadzone radius."""
        self.deadzone = float(np.clip(self.deadzone + delta, 2.0, 25.0))
        print(f"[Stabilizer] Fixation Deadzone radius: {self.deadzone:.1f} px")
        return self.deadzone

    def update(self, x: float, y: float, t: Optional[float] = None) -> Tuple[int, int]:
        """
        Smooths coordinates with the 1€ filter and applies dynamic fixation lock.
        Returns stabilized integer screen pixel coordinates (x, y).
        """
        if t is None:
            t = time.time()

        smooth_x = self.fx.filter(x, t)
        smooth_y = self.fy.filter(y, t)

        if self.locked_x is None or self.locked_y is None:
            self.locked_x, self.locked_y = smooth_x, smooth_y
            self.is_fixating = False
            return int(round(smooth_x)), int(round(smooth_y))

        displacement = math.hypot(smooth_x - self.locked_x, smooth_y - self.locked_y)
        if displacement < self.deadzone:
            # Fixating on target: lock cursor in place to absorb micro-saccades
            self.is_fixating = True
            return int(round(self.locked_x)), int(round(self.locked_y))
        else:
            # Active movement
            self.is_fixating = False
            self.locked_x = smooth_x
            self.locked_y = smooth_y
            return int(round(smooth_x)), int(round(smooth_y))

    def reset(self):
        """Clears filter state."""
        self.fx.reset()
        self.fy.reset()
        self.locked_x = None
        self.locked_y = None
        self.is_fixating = False


# Backward compatibility alias
EMAFilter = AdaptiveStabilizer


# ==============================================================================
# 7. BLINK & CLICK DETECTOR (EAR STATE MACHINE)
# ==============================================================================
class BlinkDetector:
    """Computes Eye Aspect Ratio (EAR) with anti-Midas guardrails."""

    def __init__(self, config: EyeTrackerConfig):
        self.config = config
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        self.left_click_latched = False
        self.right_click_latched = False

    @staticmethod
    def calculate_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
        """6-point Euclidean EAR formula (Soukupová & Čech, 2016)."""
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        v1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
        v2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
        h = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        if h < 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def process_eyes(self, left_ear: float, right_ear: float) -> Tuple[Optional[str], Dict[str, Any]]:
        action = None
        rearm_threshold = self.config.EAR_THRESHOLD + self.config.EAR_HYSTERESIS_OFFSET

        left_is_closed = left_ear < self.config.EAR_THRESHOLD
        right_is_closed = right_ear < self.config.EAR_THRESHOLD

        # Anti-Midas 1: Involuntary Bilateral Blink Suppression
        if self.config.SUPPRESS_BILATERAL_BLINKS and left_is_closed and right_is_closed:
            self.left_closed_frames = 0
            self.right_closed_frames = 0
            return None, {
                "left_ear": left_ear, "right_ear": right_ear,
                "left_frames": 0, "right_frames": 0,
                "bilateral_blink": True
            }

        # Anti-Midas 2: Sustained Left Wink (Subject perspective -> Left Click)
        if left_is_closed and not right_is_closed:
            self.left_closed_frames += 1
            if self.left_closed_frames >= self.config.CONSECUTIVE_FRAMES_TRIGGER and not self.left_click_latched:
                action = "left_click"
                self.left_click_latched = True
        elif left_ear >= rearm_threshold:
            self.left_closed_frames = 0
            self.left_click_latched = False

        # Anti-Midas 3: Sustained Right Wink (Subject perspective -> Right Click)
        if right_is_closed and not left_is_closed:
            self.right_closed_frames += 1
            if self.right_closed_frames >= self.config.CONSECUTIVE_FRAMES_TRIGGER and not self.right_click_latched:
                action = "right_click"
                self.right_click_latched = True
        elif right_ear >= rearm_threshold:
            self.right_closed_frames = 0
            self.right_click_latched = False

        telemetry = {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "left_frames": self.left_closed_frames,
            "right_frames": self.right_closed_frames,
            "bilateral_blink": False
        }
        return action, telemetry


# ==============================================================================
# 8. DUAL-MONITOR GAZE & HEAD POSE ESTIMATOR
# ==============================================================================
class DualMonitorGazeEstimator:
    """
    Fuses Iris tracking with Head Pose Yaw to enable natural, strain-free
    cursor navigation between dual side-by-side monitors.
    """

    def __init__(self, config: EyeTrackerConfig, display: DisplayManager):
        self.config = config
        self.display = display
        self.anchor_monitor = display.get_anchor_monitor()

        # Calibration baselines (resting gaze looking at camera/center of anchor monitor)
        self.calib_iris_x: Optional[float] = None
        self.calib_iris_y: Optional[float] = None
        self.calib_yaw: Optional[float] = None
        self.calib_pitch: Optional[float] = None

        self._calib_samples: List[Tuple[float, float, float, float]] = []

        # Rolling temporal median buffers to discard single-frame outlier jitter
        self._buf_iris_x = deque(maxlen=3)
        self._buf_iris_y = deque(maxlen=3)
        self._buf_yaw = deque(maxlen=3)
        self._buf_pitch = deque(maxlen=3)

    def calibrate(self, iris_x: float, iris_y: float, yaw: float, pitch: float):
        """Sets current forward gaze as the resting center of the anchor monitor."""
        self.calib_iris_x = iris_x
        self.calib_iris_y = iris_y
        self.calib_yaw = yaw
        self.calib_pitch = pitch
        print(f"[Calibration] Resting center calibrated on {self.anchor_monitor.name} Monitor: "
              f"Iris=({iris_x:.3f}, {iris_y:.3f}), Yaw={yaw:.3f}")

    def estimate_gaze(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """
        Calculates normalized iris gaze and facial head yaw/pitch with temporal median smoothing.
        Returns: (iris_x, iris_y, head_yaw, head_pitch)
        """
        h_img, w_img = frame_shape

        def get_xy(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * w_img, lm.y * h_img], dtype=np.float32)

        # 1. Subject Right Eye (Iris 468)
        r_iris = get_xy(Landmarks.RIGHT_IRIS_CENTER)
        r_outer = get_xy(Landmarks.RIGHT_EYE_BOUNDS["outer"])
        r_inner = get_xy(Landmarks.RIGHT_EYE_BOUNDS["inner"])
        r_top = get_xy(Landmarks.RIGHT_EYE_BOUNDS["top"])
        r_bot = get_xy(Landmarks.RIGHT_EYE_BOUNDS["bottom"])

        r_norm_x = (r_iris[0] - min(r_outer[0], r_inner[0])) / max(abs(r_outer[0] - r_inner[0]), 1e-4)
        r_norm_y = (r_iris[1] - r_top[1]) / max(abs(r_bot[1] - r_top[1]), 1e-4)

        # 2. Subject Left Eye (Iris 473)
        l_iris = get_xy(Landmarks.LEFT_IRIS_CENTER)
        l_inner = get_xy(Landmarks.LEFT_EYE_BOUNDS["inner"])
        l_outer = get_xy(Landmarks.LEFT_EYE_BOUNDS["outer"])
        l_top = get_xy(Landmarks.LEFT_EYE_BOUNDS["top"])
        l_bot = get_xy(Landmarks.LEFT_EYE_BOUNDS["bottom"])

        l_norm_x = (l_iris[0] - min(l_inner[0], l_outer[0])) / max(abs(l_outer[0] - l_inner[0]), 1e-4)
        l_norm_y = (l_iris[1] - l_top[1]) / max(abs(l_bot[1] - l_top[1]), 1e-4)

        avg_iris_x = float((r_norm_x + l_norm_x) / 2.0)
        avg_iris_y = float((r_norm_y + l_norm_y) / 2.0)

        # 3. Head Pose (Nose vs. Cheek Edges)
        nose = get_xy(Landmarks.NOSE_TIP)
        f_left = get_xy(Landmarks.FACE_RIGHT_CHEEK)
        f_right = get_xy(Landmarks.FACE_LEFT_CHEEK)
        f_top = get_xy(Landmarks.FOREHEAD)
        f_bot = get_xy(Landmarks.CHIN)

        face_w = max(abs(f_right[0] - f_left[0]), 1e-4)
        face_h = max(abs(f_bot[1] - f_top[1]), 1e-4)

        head_yaw = float((nose[0] - min(f_left[0], f_right[0])) / face_w)
        head_pitch = float((nose[1] - f_top[1]) / face_h)

        # Direction alignment:
        if self.config.INVERT_X:
            avg_iris_x = 1.0 - avg_iris_x
            head_yaw = 1.0 - head_yaw

        if self.config.INVERT_Y:
            avg_iris_y = 1.0 - avg_iris_y
            head_pitch = 1.0 - head_pitch

        # Stage 1 Stabilization: Rolling 3-frame median to reject transient detector flicker
        self._buf_iris_x.append(avg_iris_x)
        self._buf_iris_y.append(avg_iris_y)
        self._buf_yaw.append(head_yaw)
        self._buf_pitch.append(head_pitch)

        med_iris_x = float(np.median(self._buf_iris_x))
        med_iris_y = float(np.median(self._buf_iris_y))
        med_yaw = float(np.median(self._buf_yaw))
        med_pitch = float(np.median(self._buf_pitch))

        # Auto-calibrate baseline on startup after collecting 10 steady samples
        if self.calib_iris_x is None:
            self._calib_samples.append((med_iris_x, med_iris_y, med_yaw, med_pitch))
            if len(self._calib_samples) >= 10:
                self.calib_iris_x = float(np.mean([s[0] for s in self._calib_samples]))
                self.calib_iris_y = float(np.mean([s[1] for s in self._calib_samples]))
                self.calib_yaw = float(np.mean([s[2] for s in self._calib_samples]))
                self.calib_pitch = float(np.mean([s[3] for s in self._calib_samples]))
                print(f"[Auto-Calibration] Forward baseline established on {self.anchor_monitor.name} Monitor: "
                      f"Iris=({self.calib_iris_x:.3f}, {self.calib_iris_y:.3f}), Yaw={self.calib_yaw:.3f}")

        return med_iris_x, med_iris_y, med_yaw, med_pitch

    def map_to_screen(
        self,
        iris_x: float,
        iris_y: float,
        head_yaw: float,
        head_pitch: float
    ) -> Tuple[int, int, MonitorInfo]:
        """
        Maps fused gaze coordinates across both monitors.
        Returns: (screen_x, screen_y, active_monitor_info)
        """
        # Baseline reference points
        c_ix = self.calib_iris_x if self.calib_iris_x is not None else 0.5
        c_iy = self.calib_iris_y if self.calib_iris_y is not None else 0.5
        c_yaw = self.calib_yaw if self.calib_yaw is not None else 0.5
        c_pitch = self.calib_pitch if self.calib_pitch is not None else 0.5

        # Eye gaze offset
        iris_dx = (iris_x - c_ix)
        iris_dy = (iris_y - c_iy)

        # Head pose assist
        if self.config.HEAD_POSE_ASSIST:
            yaw_dx = (head_yaw - c_yaw)
            pitch_dy = (head_pitch - c_pitch)
            total_dx = iris_dx + self.config.HEAD_YAW_WEIGHT * yaw_dx
            total_dy = iris_dy + 0.45 * self.config.HEAD_YAW_WEIGHT * pitch_dy
        else:
            total_dx = iris_dx
            total_dy = iris_dy

        scaled_dx = total_dx * self.config.SENSITIVITY_X
        scaled_dy = total_dy * self.config.SENSITIVITY_Y

        # Multi-Monitor Horizontal Mapping:
        # Camera is on the RIGHT monitor. Resting forward gaze is anchor_monitor.center_x.
        # Turning/glancing left produces negative scaled_dx, gliding across seam into LEFT monitor.
        anchor = self.anchor_monitor
        screen_x = int(round(anchor.center_x + scaled_dx * anchor.width))

        # Clamp within full virtual desktop width
        v_min_x = self.display.virtual_left
        v_max_x = self.display.virtual_left + self.display.virtual_width - 1
        screen_x = max(v_min_x, min(v_max_x, screen_x))

        # Identify which physical monitor the cursor is horizontally over:
        active_monitor = self.display.get_monitor_for_x(screen_x)

        # Vertical Mapping: Scaled relative to the active monitor's vertical center and height
        screen_y = int(round(active_monitor.center_y + scaled_dy * active_monitor.height))
        screen_y = max(active_monitor.top, min(active_monitor.bottom - 1, screen_y))

        return screen_x, screen_y, active_monitor


# ==============================================================================
# 9. MASTER EYE-TRACKING MOUSE CONTROLLER
# ==============================================================================
class EyeTrackingMouse:
    """Master controller integrating dual-monitor tracking, vision inference, and actuation."""

    def __init__(self, config: Optional[EyeTrackerConfig] = None):
        self.config = config or EyeTrackerConfig()

        # Multi-monitor display geometry
        self.display = DisplayManager(camera_on_monitor=self.config.CAMERA_ON_MONITOR)
        self.mouse = MouseController()

        # Processing sub-modules
        self.stabilizer = AdaptiveStabilizer(
            min_cutoff=self.config.ONE_EURO_MIN_CUTOFF,
            beta=self.config.ONE_EURO_BETA,
            deadzone=self.config.FIXATION_DEADZONE,
            preset=self.config.STABILIZER_PRESET
        )
        self.filter = self.stabilizer  # Backward compatibility
        self.blink_detector = BlinkDetector(self.config)
        self.gaze_estimator = DualMonitorGazeEstimator(self.config, self.display)
        self.mesh_adapter = FaceMeshAdapter(model_asset_path=self.config.MODEL_TASK_PATH)

        # Telemetry
        self.last_action_text = "Idle"
        self.last_action_timestamp = 0.0
        self.fps = 0.0
        self._prev_frame_time = time.time()
        self.active_mon_name = self.display.get_anchor_monitor().name

    def _extract_ear_landmarks(self, landmarks: Any, indices: List[int], w: int, h: int) -> List[Tuple[float, float]]:
        return [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in indices]

    def _execute_mouse_action(self, action: str):
        if not self.config.ENABLE_MOUSE_CONTROL:
            return
        if action == "left_click":
            self.mouse.click(button="left")
        elif action == "right_click":
            self.mouse.click(button="right")

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Auto-detects functional camera device with DirectShow on Windows."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        candidate_indices = [self.config.CAMERA_INDEX] + [i for i in [0, 1, 2] if i != self.config.CAMERA_INDEX]

        for idx in candidate_indices:
            for api in backends:
                try:
                    cap = cv2.VideoCapture(idx, api)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
                            api_name = "CAP_DSHOW" if api == cv2.CAP_DSHOW else "CAP_ANY"
                            print(f"[Camera] Connected to camera {idx} via {api_name}.")
                            return cap
                        cap.release()
                except Exception:
                    pass
        return None

    def _draw_hud(
        self,
        frame: np.ndarray,
        landmarks: Any,
        telemetry: Dict[str, Any],
        cursor_coords: Tuple[int, int],
        active_mon: MonitorInfo
    ):
        h, w, _ = frame.shape

        def draw_pt(idx: int, color: Tuple[int, int, int], radius: int = 3):
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), radius, color, -1)

        # Iris dots (Cyan & Yellow)
        draw_pt(Landmarks.LEFT_IRIS_CENTER, (255, 255, 0), radius=4)
        draw_pt(Landmarks.RIGHT_IRIS_CENTER, (0, 255, 255), radius=4)
        draw_pt(Landmarks.NOSE_TIP, (0, 165, 255), radius=4)  # Orange nose marker for head yaw

        # Status Overlay Card (Expanded for stability metrics)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (385, 205), (15, 20, 26), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (10, 10), (385, 205), (60, 75, 90), 1)

        mode_str = "ACTIVE" if self.config.ENABLE_MOUSE_CONTROL else "PREVIEW ONLY"
        mode_col = (0, 255, 180) if self.config.ENABLE_MOUSE_CONTROL else (100, 180, 255)
        cv2.putText(frame, f"EyeMouse Dual [{mode_str}] | {self.fps:.1f} FPS", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_col, 1, cv2.LINE_AA)

        # Active Monitor Badge
        mon_color = (0, 255, 255) if active_mon.name == "RIGHT" else (255, 180, 0)
        cv2.putText(frame, f"Active Display: [{active_mon.name} MONITOR] ({active_mon.width}x{active_mon.height})",
                    (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, mon_color, 1, cv2.LINE_AA)

        # Cursor Virtual Position
        cv2.putText(frame, f"Virtual Cursor: ({cursor_coords[0]}, {cursor_coords[1]})", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # EAR Gauges
        l_ear = telemetry.get("left_ear", 0.0)
        l_frames = telemetry.get("left_frames", 0)
        l_col = (0, 0, 255) if l_ear < self.config.EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"L-Eye EAR: {l_ear:.3f} [{l_frames}/{self.config.CONSECUTIVE_FRAMES_TRIGGER}] (Left Click)",
                    (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.38, l_col, 1, cv2.LINE_AA)

        r_ear = telemetry.get("right_ear", 0.0)
        r_frames = telemetry.get("right_frames", 0)
        r_col = (0, 0, 255) if r_ear < self.config.EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"R-Eye EAR: {r_ear:.3f} [{r_frames}/{self.config.CONSECUTIVE_FRAMES_TRIGGER}] (Right Click)",
                    (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.38, r_col, 1, cv2.LINE_AA)

        # Stability & Fixation Lock State
        fix_str = "FIXATION LOCKED" if self.stabilizer.is_fixating else "TRACKING"
        fix_col = (0, 255, 120) if self.stabilizer.is_fixating else (255, 200, 0)
        cv2.putText(frame, f"Stability: [{self.stabilizer.preset}] ({self.stabilizer.deadzone:.0f}px) - {fix_str}",
                    (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.38, fix_col, 1, cv2.LINE_AA)

        # Dual Monitor Map representation & Axis Direction
        axis_str = "INVERTED" if self.config.INVERT_X else "NATURAL"
        cv2.putText(frame, f"Cam: RIGHT | Head Yaw: {'ON' if self.config.HEAD_POSE_ASSIST else 'OFF'} | Axis: {axis_str}",
                    (20, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 200), 1, cv2.LINE_AA)

        # Action or Hotkeys
        if time.time() - self.last_action_timestamp < 0.6:
            cv2.putText(frame, f"** {self.last_action_text} TRIGGERED **", (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Hotkeys: [S] Mode | [+/-] Deadzone | [C] Center | [I] Invert | [Q] Quit",
                        (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 150, 160), 1, cv2.LINE_AA)

    def run(self):
        """Starts real-time dual-monitor eye tracking capture loop."""
        print("=" * 80)
        print("👁️  STARTING REAL-TIME DUAL-MONITOR EYE-TRACKING MOUSE (ADVANCED STABILIZATION)")
        print("=" * 80)
        print(f"Detected Monitors ({len(self.display.monitors)}):")
        for m in self.display.monitors:
            tag = " [WEBCAM MOUNTED HERE]" if m == self.display.get_anchor_monitor() else ""
            print(f"  • {m.name} Monitor: {m.width}x{m.height} at ({m.left}, {m.top}){tag}")
        print(f"Virtual Desktop Span:   {self.display.virtual_width} x {self.display.virtual_height}")
        print(f"Camera Anchor:          {self.display.get_anchor_monitor().name} Monitor (Center: {self.display.get_anchor_monitor().center_x}, {self.display.get_anchor_monitor().center_y})")
        print(f"Stabilizer Engine:      1€ Adaptive Filter + Dynamic Fixation Lock (Preset: {self.stabilizer.preset})")
        print(f"Fixation Deadzone:      {self.stabilizer.deadzone:.1f} px")
        print(f"Head Pose Assist:       {'ENABLED (Yaw weight: ' + str(self.config.HEAD_YAW_WEIGHT) + ')' if self.config.HEAD_POSE_ASSIST else 'DISABLED'}")
        print(f"X-Axis Direction:       {'INVERTED' if self.config.INVERT_X else 'NATURAL (Turning Left -> Moves Left)'}")
        print(f"Horizontal Sensitivity: {self.config.SENSITIVITY_X:.1f} (Dual-screen reach)")
        print(f"EAR Threshold:          {self.config.EAR_THRESHOLD:.2f} (Eye Closed Detection)")
        print("-" * 80)
        print("Controls & Hotkeys:")
        print("  • Look at Right Monitor Center + Press 'c' -> Calibrate Resting Center")
        print("  • Press 's'                                -> Cycle Stability: SMOOTH -> ULTRA-STABLE -> RESPONSIVE")
        print("  • Press '+' or '='                         -> Increase Fixation Deadzone (+1 px)")
        print("  • Press '-' or '_'                         -> Decrease Fixation Deadzone (-1 px)")
        print("  • Press 'i'                                -> Toggle Invert X-Axis Direction")
        print("  • Press '1'                                -> Snap Cursor to Left Monitor Center")
        print("  • Press '2'                                -> Snap Cursor to Right Monitor Center")
        print("  • Wink Left Eye (hold ~0.2s)               -> Left Mouse Click")
        print("  • Wink Right Eye (hold ~0.2s)              -> Right Mouse Click")
        print("  • Natural Blinking (both eyes)             -> Automatically Ignored (Anti-Midas)")
        print("  • Press 'm'                                -> Toggle Mouse Actuation On/Off")
        print("  • Press 'q' or 'ESC'                       -> Quit")
        print("=" * 80)

        cap = self._open_camera()
        if cap is None:
            print("[ERROR] Could not connect to a working webcam device.")
            return

        consecutive_read_failures = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    consecutive_read_failures += 1
                    if consecutive_read_failures > 60:
                        print("\n[ERROR] Camera frame stream lost. Exiting.")
                        break
                    time.sleep(0.01)
                    continue

                consecutive_read_failures = 0

                if self.config.MIRROR_WEBCAM:
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape
                face_landmarks = self.mesh_adapter.process(frame)

                cursor_x = self.display.get_anchor_monitor().center_x
                cursor_y = self.display.get_anchor_monitor().center_y
                active_mon = self.display.get_anchor_monitor()
                telemetry = {"left_ear": 0.0, "right_ear": 0.0, "left_frames": 0, "right_frames": 0}
                ix, iy, yaw, pitch = 0.5, 0.5, 0.5, 0.5

                if face_landmarks:
                    # 1. Blink Detection via 6-Point EAR
                    left_pts = self._extract_ear_landmarks(face_landmarks, Landmarks.LEFT_EYE_EAR, w, h)
                    right_pts = self._extract_ear_landmarks(face_landmarks, Landmarks.RIGHT_EYE_EAR, w, h)

                    left_ear = self.blink_detector.calculate_ear(left_pts)
                    right_ear = self.blink_detector.calculate_ear(right_pts)

                    action, telemetry = self.blink_detector.process_eyes(left_ear, right_ear)
                    if action:
                        self.last_action_text = action.upper().replace("_", " ")
                        self.last_action_timestamp = time.time()
                        self._execute_mouse_action(action)

                    # 2. Dual-Monitor Gaze & Head Pose Estimation
                    ix, iy, yaw, pitch = self.gaze_estimator.estimate_gaze(face_landmarks, (h, w))
                    raw_x, raw_y, active_mon = self.gaze_estimator.map_to_screen(ix, iy, yaw, pitch)

                    # 3. Jitter Reduction: EMA + Deadzone
                    cursor_x, cursor_y = self.filter.update(raw_x, raw_y)

                    # 4. Native Multi-Monitor Cursor Actuation
                    if self.config.ENABLE_MOUSE_CONTROL:
                        self.mouse.move(cursor_x, cursor_y)

                    # 5. Render Heads-Up Display
                    if self.config.SHOW_DEBUG_HUD:
                        self._draw_hud(frame, face_landmarks, telemetry, (cursor_x, cursor_y), active_mon)

                curr_time = time.time()
                self.fps = 1.0 / max(curr_time - self._prev_frame_time, 1e-4)
                self._prev_frame_time = curr_time

                cv2.imshow("Real-Time Dual-Monitor Eye-Tracking Mouse", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    print("\n[INFO] User exit requested. Shutting down.")
                    break
                elif key == ord('m'):
                    self.config.ENABLE_MOUSE_CONTROL = not self.config.ENABLE_MOUSE_CONTROL
                    state = "ACTIVE" if self.config.ENABLE_MOUSE_CONTROL else "DISABLED"
                    print(f"[TOGGLE] Mouse actuation is now: {state}")
                elif key == ord('i'):
                    self.config.INVERT_X = not self.config.INVERT_X
                    state = "INVERTED" if self.config.INVERT_X else "NATURAL (Turn Left -> Moves Left)"
                    print(f"[TOGGLE] Horizontal tracking direction is now: {state}")
                elif key == ord('s'):
                    self.stabilizer.cycle_preset()
                elif key in [ord('+'), ord('=')]:
                    self.stabilizer.adjust_deadzone(+1.0)
                elif key in [ord('-'), ord('_')]:
                    self.stabilizer.adjust_deadzone(-1.0)
                elif key == ord('c') and face_landmarks:
                    self.gaze_estimator.calibrate(ix, iy, yaw, pitch)
                    self.stabilizer.reset()
                elif key == ord('1') and len(self.display.monitors) > 0:
                    left_m = self.display.monitors[0]
                    self.mouse.move(left_m.center_x, left_m.center_y)
                    self.stabilizer.reset()
                    print(f"[SNAP] Cursor snapped to Left Monitor center: ({left_m.center_x}, {left_m.center_y})")
                elif key == ord('2') and len(self.display.monitors) > 1:
                    right_m = self.display.monitors[-1]
                    self.mouse.move(right_m.center_x, right_m.center_y)
                    self.stabilizer.reset()
                    print(f"[SNAP] Cursor snapped to Right Monitor center: ({right_m.center_x}, {right_m.center_y})")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mesh_adapter.close()
            print("[INFO] Cleanup complete. Camera and Face Mesh resources released.")


# ==============================================================================
# SCRIPT ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    config = EyeTrackerConfig(
        ENABLE_DUAL_MONITORS=True,        # Seamless navigation across both monitors
        CAMERA_ON_MONITOR="right",         # Camera is mounted on the RIGHT monitor
        HEAD_POSE_ASSIST=True,            # Fuses head yaw for natural glance across monitors
        HEAD_YAW_WEIGHT=0.70,             # Head turn weight
        SENSITIVITY_X=3.2,                # Gain factor to comfortably reach both monitor edges
        SENSITIVITY_Y=2.8,                # Gain factor for vertical span
        EAR_THRESHOLD=0.18,               # Blink detection threshold (winks drop to ~0.10)
        CONSECUTIVE_FRAMES_TRIGGER=5,     # 5 frames @ 30 FPS ~ 160ms deliberate wink
        STABILIZER_PRESET="SMOOTH",       # "SMOOTH" (balanced), "ULTRA-STABLE" (pinned), or "RESPONSIVE"
        ONE_EURO_MIN_CUTOFF=0.55,         # 1€ Filter low-speed cutoff frequency (Hz)
        ONE_EURO_BETA=0.008,              # 1€ Filter velocity adaptation coefficient
        FIXATION_DEADZONE=7.0,            # Radius in pixels to lock cursor during target fixation
        ENABLE_MOUSE_CONTROL=True,        # True = live cursor control
        SHOW_DEBUG_HUD=True,
        MODEL_TASK_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
    )

    eye_mouse = EyeTrackingMouse(config)
    eye_mouse.run()
