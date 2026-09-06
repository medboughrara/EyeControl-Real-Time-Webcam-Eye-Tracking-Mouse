"""
================================================================================
Gaze Calibration & Posture Invariant Regression Engine
================================================================================
Provides:
  1. PolynomialRidgeModel: Pure-NumPy Degree-2 Polynomial Ridge Regressor.
     Solves non-linear gaze curvature across dual-monitor planes via closed-form
     regularized least squares: W = (Phi^T Phi + alpha * I)^(-1) Phi^T Y
  2. PostureNormalizer: 3D Inter-Ocular Distance (IOD) tracker to compensate for
     depth drift, slouching, and leaning forward/backward.
  3. CalibrationWizard: Interactive fullscreen target calibration UI.
================================================================================
"""

import os
import json
import time
import math
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

# Landmark indices for Inter-Ocular Distance (IOD)
LANDMARK_RIGHT_EYE_OUTER = 33
LANDMARK_LEFT_EYE_OUTER = 263


class PostureNormalizer:
    """
    Tracks 3D Inter-Ocular Distance (IOD) between outer eye corners (landmarks 33 and 263)
    to normalize feature vectors against user distance, posture changes, and camera depth.
    """

    def __init__(self, default_baseline_iod: Optional[float] = None):
        self.baseline_iod: Optional[float] = default_baseline_iod

    def compute_iod(self, landmarks: Any, frame_w: int, frame_h: int) -> float:
        """Computes 3D Euclidean distance between outer eye corners in pixel space."""
        p_r = landmarks[LANDMARK_RIGHT_EYE_OUTER]
        p_l = landmarks[LANDMARK_LEFT_EYE_OUTER]

        dx = (p_r.x - p_l.x) * frame_w
        dy = (p_r.y - p_l.y) * frame_h
        dz = (p_r.z - p_l.z) * frame_w  # MediaPipe Z scale is relative to width

        iod = float(math.sqrt(dx * dx + dy * dy + dz * dz))
        return max(iod, 1e-4)

    def set_baseline(self, iod: float):
        """Sets the resting baseline IOD captured during calibration."""
        self.baseline_iod = float(max(iod, 1e-4))

    def get_depth_scale(self, current_iod: float) -> float:
        """
        Returns depth ratio s_depth = current_iod / baseline_iod.
        > 1.0 = leaning forward / closer to camera
        < 1.0 = leaning backward / farther from camera
        """
        if self.baseline_iod is None or self.baseline_iod < 1e-4:
            return 1.0
        scale = current_iod / self.baseline_iod
        return float(np.clip(scale, 0.4, 2.5))


class PolynomialRidgeModel:
    """
    Degree-2 Polynomial Ridge Regressor implemented in pure NumPy.
    Maps non-linear eye-gaze and head-pose features to screen pixel coordinates (X, Y).
    """

    def __init__(self, degree: int = 2, alpha: float = 5.0):
        self.degree = degree
        self.alpha = float(alpha)  # L2 regularization penalty
        self.weights: Optional[np.ndarray] = None  # Shape (M, 2)
        self.is_calibrated: bool = False
        self.metadata: Dict[str, Any] = {}

    def _expand_features(self, X: np.ndarray) -> np.ndarray:
        """
        Expands input feature matrix X (N, D) into degree-2 polynomial features:
        [1, x_1, ..., x_D, x_1^2, x_1*x_2, ..., x_D^2]
        """
        N, D = X.shape
        terms = [np.ones((N, 1), dtype=np.float64)]  # Bias / intercept term

        # Linear terms
        terms.append(X.astype(np.float64))

        # Degree 2 interaction and quadratic terms
        if self.degree >= 2:
            quad_terms = []
            for i in range(D):
                for j in range(i, D):
                    quad_terms.append((X[:, i] * X[:, j])[:, np.newaxis])
            if quad_terms:
                terms.append(np.hstack(quad_terms))

        return np.hstack(terms)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Trains model weights W via closed-form Ridge Regression:
        W = (Phi^T Phi + alpha * I)^(-1) Phi^T Y
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        Phi = self._expand_features(X)
        N, M = Phi.shape

        A = Phi.T @ Phi
        # Regularization: penalize non-constant weights to prevent extrapolation spikes
        reg_matrix = np.eye(M, dtype=np.float64) * self.alpha
        reg_matrix[0, 0] = 0.0  # Do not penalize the bias/intercept term

        A_reg = A + reg_matrix
        b = Phi.T @ Y

        try:
            self.weights = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            # Pseudo-inverse fallback for ill-conditioned systems
            self.weights = np.linalg.pinv(A_reg) @ b

        self.is_calibrated = True

        # Compute training residual MSE
        residuals = Y - (Phi @ self.weights)
        mse = float(np.mean(np.sum(residuals ** 2, axis=1)))
        self.metadata["train_mse"] = mse
        self.metadata["num_samples"] = int(N)
        self.metadata["num_features"] = int(M)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts (X_screen, Y_screen) given input feature array."""
        if not self.is_calibrated or self.weights is None:
            raise RuntimeError("Model is not calibrated. Call fit() or load_profile() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        Phi = self._expand_features(X)
        preds = Phi @ self.weights
        return preds

    def save_profile(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """Persists trained weights and metadata to JSON."""
        if not self.is_calibrated or self.weights is None:
            raise RuntimeError("Cannot save an uncalibrated model.")

        save_dict = {
            "version": "2.0.0",
            "model_type": "PolynomialRidge",
            "degree": self.degree,
            "alpha": self.alpha,
            "weights": self.weights.tolist(),
            "metadata": {**self.metadata, **(metadata or {})}
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2)

    @classmethod
    def load_profile(cls, filepath: str) -> "PolynomialRidgeModel":
        """Loads a persisted calibration profile from JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(degree=data.get("degree", 2), alpha=data.get("alpha", 5.0))
        model.weights = np.array(data["weights"], dtype=np.float64)
        model.metadata = data.get("metadata", {})
        model.is_calibrated = True
        return model


class CalibrationWizard:
    """
    Interactive fullscreen calibration routine presenting a 9-point grid per monitor.
    Collects steady feature vectors and fits the PolynomialRidgeModel.
    """

    def __init__(
        self,
        display_manager: Any,
        mesh_adapter: Any,
        posture_normalizer: PostureNormalizer,
        config: Any,
        output_profile_path: str = "calibration_profile.json"
    ):
        self.display = display_manager
        self.mesh_adapter = mesh_adapter
        self.normalizer = posture_normalizer
        self.config = config
        self.output_profile_path = output_profile_path

    def _generate_target_points(self) -> List[Tuple[int, int, str]]:
        """
        Generates 9 calibration target points per monitor (3x3 grid).
        Returns list of (screen_x, screen_y, monitor_name).
        """
        targets = []
        # Normalized target coordinates (inset from borders to avoid bezel occlusion)
        grid_coords = [0.15, 0.50, 0.85]

        for m in self.display.monitors:
            for gy in grid_coords:
                for gx in grid_coords:
                    tx = int(m.left + gx * m.width)
                    ty = int(m.top + gy * m.height)
                    targets.append((tx, ty, m.name))

        return targets

    def run_calibration(self, cap: cv2.VideoCapture) -> Optional[PolynomialRidgeModel]:
        """
        Runs the interactive calibration wizard.
        Fixates on targets across monitors, collects stable eye features, and fits the model.
        """
        targets = self._generate_target_points()
        print(f"\n[Calibration Wizard] Starting interactive calibration ({len(targets)} targets across displays)...")
        print("  • Look at each pulsing target until it turns GREEN.")
        print("  • Keep head in natural resting posture. Press ESC to cancel.\n")

        v_left = self.display.virtual_left
        v_top = self.display.virtual_top
        v_w = self.display.virtual_width
        v_h = self.display.virtual_height

        window_name = "EyeControl Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, v_w, v_h)
        cv2.moveWindow(window_name, v_left, v_top)

        all_features: List[List[float]] = []
        all_targets: List[Tuple[int, int]] = []
        iod_samples: List[float] = []

        target_duration = 1.3  # seconds per target
        collect_after = 0.5    # ignore first 500ms to allow saccadic eye transition

        for idx, (tx, ty, mon_name) in enumerate(targets):
            start_time = time.time()
            target_features: List[List[float]] = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                if self.config.MIRROR_WEBCAM:
                    frame = cv2.flip(frame, 1)

                h_f, w_f, _ = frame.shape
                res = self.mesh_adapter.process(frame)
                landmarks = res[0] if isinstance(res, tuple) else res

                elapsed = time.time() - start_time
                progress = min(elapsed / target_duration, 1.0)

                # Feature extraction if landmarks detected
                if landmarks:
                    iod = self.normalizer.compute_iod(landmarks, w_f, h_f)
                    iod_samples.append(iod)

                    # Extract relative eye and head features
                    feats = self._extract_features(landmarks, (h_f, w_f), iod)
                    if elapsed >= collect_after:
                        target_features.append(feats)

                # Render fullscreen calibration canvas
                canvas = np.zeros((v_h, v_w, 3), dtype=np.uint8)

                # Convert virtual desktop target (tx, ty) to canvas local (cx, cy)
                cx = tx - v_left
                cy = ty - v_top

                # Draw subtle grid guide
                cv2.circle(canvas, (cx, cy), 50, (35, 45, 55), 1)

                # Outer shrinking progress ring
                ring_radius = int(50 * (1.0 - progress)) + 8
                ring_color = (0, 255, 120) if elapsed >= collect_after else (0, 200, 255)
                cv2.circle(canvas, (cx, cy), ring_radius, ring_color, 2, cv2.LINE_AA)

                # Inner bullseye dot
                dot_color = (0, 255, 0) if elapsed >= collect_after else (0, 255, 255)
                cv2.circle(canvas, (cx, cy), 7, dot_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

                # Header instruction text
                status_text = f"Target {idx + 1} of {len(targets)} [{mon_name} Monitor] - Fixate on the target"
                cv2.putText(canvas, status_text, (v_w // 2 - 250, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q')]:
                    print("[Calibration Wizard] Calibration cancelled by user.")
                    cv2.destroyWindow(window_name)
                    return None

                if elapsed >= target_duration:
                    break

            if target_features:
                # Average samples collected during fixation window
                mean_feats = np.mean(target_features, axis=0).tolist()
                all_features.append(mean_feats)
                all_targets.append((tx, ty))

        cv2.destroyWindow(window_name)

        if len(all_features) < 4:
            print("[Calibration Wizard] Insufficient samples collected. Calibration failed.")
            return None

        # Compute baseline IOD across calibration
        baseline_iod = float(np.median(iod_samples)) if iod_samples else 1.0
        self.normalizer.set_baseline(baseline_iod)

        # Fit Degree-2 Polynomial Ridge Model
        X_train = np.array(all_features, dtype=np.float64)
        Y_train = np.array(all_targets, dtype=np.float64)

        model = PolynomialRidgeModel(degree=2, alpha=5.0)
        model.fit(X_train, Y_train)

        # Save profile
        metadata = {
            "baseline_iod": baseline_iod,
            "calibrated_targets": len(all_targets),
            "timestamp": time.time(),
            "virtual_geometry": {
                "left": v_left, "top": v_top, "width": v_w, "height": v_h
            }
        }
        model.save_profile(self.output_profile_path, metadata=metadata)
        print(f"\n✓ Calibration Successful! Profile saved to '{self.output_profile_path}'.")
        print(f"  • Trained on {len(all_targets)} targets across dual screens.")
        print(f"  • Baseline IOD: {baseline_iod:.1f}px (Depth Normalization Active).")
        print(f"  • Training RMSE: {math.sqrt(model.metadata.get('train_mse', 0.0)):.1f} pixels.")

        return model

    def _extract_features(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int],
        iod: float
    ) -> List[float]:
        """Extracts normalized 6D feature vector for polynomial regression."""
        h_img, w_img = frame_shape

        def get_xy(idx: int) -> Tuple[float, float]:
            lm = landmarks[idx]
            return lm.x * w_img, lm.y * h_img

        # Right eye (subject perspective: camera left)
        r_iris = get_xy(468)
        r_outer = get_xy(33)
        r_inner = get_xy(133)
        r_top = get_xy(159)
        r_bot = get_xy(145)

        r_w = max(abs(r_outer[0] - r_inner[0]), 1e-4)
        r_h = max(abs(r_bot[1] - r_top[1]), 1e-4)
        r_nx = (r_iris[0] - min(r_outer[0], r_inner[0])) / r_w
        r_ny = (r_iris[1] - r_top[1]) / r_h

        # Left eye (subject perspective: camera right)
        l_iris = get_xy(473)
        l_inner = get_xy(362)
        l_outer = get_xy(263)
        l_top = get_xy(386)
        l_bot = get_xy(374)

        l_w = max(abs(l_outer[0] - l_inner[0]), 1e-4)
        l_h = max(abs(l_bot[1] - l_top[1]), 1e-4)
        l_nx = (l_iris[0] - min(l_inner[0], l_outer[0])) / l_w
        l_ny = (l_iris[1] - l_top[1]) / l_h

        avg_iris_x = float((r_nx + l_nx) / 2.0)
        avg_iris_y = float((r_ny + l_ny) / 2.0)

        # Head pose via nose vs facial bounds
        nose = get_xy(1)
        f_left = get_xy(234)
        f_right = get_xy(454)
        f_top = get_xy(10)
        f_bot = get_xy(152)

        face_w = max(abs(f_right[0] - f_left[0]), 1e-4)
        face_h = max(abs(f_bot[1] - f_top[1]), 1e-4)

        head_yaw = float((nose[0] - min(f_left[0], f_right[0])) / face_w)
        head_pitch = float((nose[1] - f_top[1]) / face_h)

        # Depth scale ratio relative to baseline
        depth_scale = self.normalizer.get_depth_scale(iod)

        # Pupil aspect ratio
        pupil_aspect = float(r_w / max(l_w, 1e-4))

        return [avg_iris_x, avg_iris_y, head_yaw, head_pitch, depth_scale, pupil_aspect]
