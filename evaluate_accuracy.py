"""
================================================================================
EyeControl Ground-Truth Accuracy Benchmark Harness
================================================================================
Empirical evaluation tool measuring cursor accuracy (Mean Radial Error in pixels)
against a ground-truth grid of visual fixation targets across dual displays.

Usage:
  python evaluate_accuracy.py                  # Evaluates current calibrated model (or heuristic fallback)
  python evaluate_accuracy.py --baseline       # Forces evaluation using the baseline linear heuristic
================================================================================
"""

import os
import sys
import time
import math
import argparse
import json
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

from eye_tracking_mouse import EyeTrackerConfig, DisplayManager, FaceMeshAdapter, DualMonitorGazeEstimator
from gaze_calibration import PolynomialRidgeModel, PostureNormalizer


class AccuracyEvaluator:
    """
    Presents a test grid of visual fixation targets across displays,
    records cursor predictions, and computes rigorous empirical error statistics.
    """

    def __init__(
        self,
        config: Optional[EyeTrackerConfig] = None,
        force_baseline: bool = False,
        profile_path: str = "calibration_profile.json"
    ):
        self.config = config or EyeTrackerConfig()
        self.force_baseline = force_baseline
        self.profile_path = profile_path

        self.display = DisplayManager(camera_on_monitor=self.config.CAMERA_ON_MONITOR)
        self.mesh_adapter = FaceMeshAdapter(model_asset_path=self.config.MODEL_TASK_PATH)
        self.normalizer = PostureNormalizer()

        # Load calibration if available and not forced baseline
        self.calib_model: Optional[PolynomialRidgeModel] = None
        if not self.force_baseline and os.path.exists(self.profile_path):
            try:
                self.calib_model = PolynomialRidgeModel.load_profile(self.profile_path)
                base_iod = self.calib_model.metadata.get("baseline_iod")
                if base_iod:
                    self.normalizer.set_baseline(base_iod)
                print(f"[AccuracyEvaluator] Loaded calibrated PolynomialRidgeModel from '{self.profile_path}'.")
            except Exception as e:
                print(f"[AccuracyEvaluator] Warning: Could not load calibration profile: {e}")

        # Baseline heuristic estimator as fallback
        self.heuristic_estimator = DualMonitorGazeEstimator(self.config, self.display)

    def _generate_eval_targets(self) -> List[Tuple[int, int, str]]:
        """
        Generates evaluation targets distributed across each monitor.
        Uses intermediate test fractions (0.25, 0.50, 0.75) to test interpolation accuracy.
        """
        targets = []
        fractions = [0.25, 0.50, 0.75]

        for m in self.display.monitors:
            for fy in fractions:
                for fx in fractions:
                    tx = int(m.left + fx * m.width)
                    ty = int(m.top + fy * m.height)
                    targets.append((tx, ty, m.name))

        return targets

    def _predict_screen_pos(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int],
        iod: float
    ) -> Tuple[int, int]:
        """Predicts screen (X, Y) via calibrated model or baseline heuristic."""
        h_f, w_f = frame_shape

        if self.calib_model and self.calib_model.is_calibrated and not self.force_baseline:
            # Extract 6D polynomial features
            feats = self._extract_features(landmarks, frame_shape, iod)
            pred = self.calib_model.predict(np.array(feats))
            return int(round(pred[0, 0])), int(round(pred[0, 1]))
        else:
            # Baseline linear heuristic
            ix, iy, yaw, pitch = self.heuristic_estimator.estimate_gaze(landmarks, frame_shape)
            sx, sy, _ = self.heuristic_estimator.map_to_screen(ix, iy, yaw, pitch)
            return sx, sy

    def _extract_features(self, landmarks: Any, frame_shape: Tuple[int, int], iod: float) -> List[float]:
        h_img, w_img = frame_shape

        def get_xy(idx: int) -> Tuple[float, float]:
            lm = landmarks[idx]
            return lm.x * w_img, lm.y * h_img

        r_iris = get_xy(468)
        r_outer = get_xy(33)
        r_inner = get_xy(133)
        r_top = get_xy(159)
        r_bot = get_xy(145)
        r_w = max(abs(r_outer[0] - r_inner[0]), 1e-4)
        r_h = max(abs(r_bot[1] - r_top[1]), 1e-4)
        r_nx = (r_iris[0] - min(r_outer[0], r_inner[0])) / r_w
        r_ny = (r_iris[1] - r_top[1]) / r_h

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

        nose = get_xy(1)
        f_left = get_xy(234)
        f_right = get_xy(454)
        f_top = get_xy(10)
        f_bot = get_xy(152)

        face_w = max(abs(f_right[0] - f_left[0]), 1e-4)
        face_h = max(abs(f_bot[1] - f_top[1]), 1e-4)

        head_yaw = float((nose[0] - min(f_left[0], f_right[0])) / face_w)
        head_pitch = float((nose[1] - f_top[1]) / face_h)

        depth_scale = self.normalizer.get_depth_scale(iod)
        pupil_aspect = float(r_w / max(l_w, 1e-4))

        return [avg_iris_x, avg_iris_y, head_yaw, head_pitch, depth_scale, pupil_aspect]

    def run_benchmark(self, cap: Optional[cv2.VideoCapture] = None) -> Dict[str, Any]:
        """Runs the accuracy benchmark sequence and returns aggregated evaluation metrics."""
        targets = self._generate_eval_targets()
        mode_label = "BASELINE HEURISTIC" if (self.force_baseline or not self.calib_model) else "POLYNOMIAL RIDGE (Degree-2)"

        print("=" * 80)
        print(f"🎯 STARTING ACCURACY BENCHMARK HARNESS [{mode_label}]")
        print(f"   Targets: {len(targets)} across {len(self.display.monitors)} monitors")
        print("   Fixate steadily on each red target until the ring closes.")
        print("=" * 80)

        own_cap = False
        if cap is None:
            own_cap = True
            cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)

        v_left = self.display.virtual_left
        v_top = self.display.virtual_top
        v_w = self.display.virtual_width
        v_h = self.display.virtual_height

        window_name = "EyeControl Accuracy Benchmark"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, v_w, v_h)
        cv2.moveWindow(window_name, v_left, v_top)

        target_duration = 1.3
        collect_after = 0.5

        results_per_target: List[Dict[str, Any]] = []

        try:
            for idx, (tx, ty, mon_name) in enumerate(targets):
                start_time = time.time()
                predicted_points: List[Tuple[int, int]] = []

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

                    if landmarks:
                        iod = self.normalizer.compute_iod(landmarks, w_f, h_f)
                        pred_x, pred_y = self._predict_screen_pos(landmarks, (h_f, w_f), iod)
                        if elapsed >= collect_after:
                            predicted_points.append((pred_x, pred_y))

                    # Fullscreen canvas
                    canvas = np.zeros((v_h, v_w, 3), dtype=np.uint8)
                    cx = tx - v_left
                    cy = ty - v_top

                    # Outer countdown ring
                    ring_r = int(45 * (1.0 - progress)) + 6
                    ring_color = (0, 165, 255) if elapsed < collect_after else (0, 255, 100)
                    cv2.circle(canvas, (cx, cy), ring_r, ring_color, 2, cv2.LINE_AA)

                    # Center target
                    cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

                    # Header text
                    title_str = f"Accuracy Benchmark [{mode_label}] - Target {idx + 1}/{len(targets)} ({mon_name})"
                    cv2.putText(canvas, title_str, (v_w // 2 - 320, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)

                    cv2.imshow(window_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [27, ord('q')]:
                        print("[Benchmark] Evaluation aborted by user.")
                        return {}

                    if elapsed >= target_duration:
                        break

                if predicted_points:
                    mean_pred_x = float(np.mean([p[0] for p in predicted_points]))
                    mean_pred_y = float(np.mean([p[1] for p in predicted_points]))
                    radial_err = float(math.hypot(mean_pred_x - tx, mean_pred_y - ty))
                    dx = float(mean_pred_x - tx)
                    dy = float(mean_pred_y - ty)

                    results_per_target.append({
                        "target_idx": idx,
                        "monitor": mon_name,
                        "target_pos": [tx, ty],
                        "predicted_pos": [round(mean_pred_x, 1), round(mean_pred_y, 1)],
                        "radial_error_px": round(radial_err, 1),
                        "dx_px": round(dx, 1),
                        "dy_px": round(dy, 1)
                    })

        finally:
            cv2.destroyWindow(window_name)
            if own_cap:
                cap.release()

        if not results_per_target:
            print("[Benchmark] No valid measurements recorded.")
            return {}

        # Aggregate Statistics
        all_errors = [r["radial_error_px"] for r in results_per_target]
        mean_radial_error = float(np.mean(all_errors))
        median_radial_error = float(np.median(all_errors))
        std_radial_error = float(np.std(all_errors))
        p90_error = float(np.percentile(all_errors, 90))

        # Per-monitor breakdown
        monitors_summary: Dict[str, Any] = {}
        for m in self.display.monitors:
            m_errs = [r["radial_error_px"] for r in results_per_target if r["monitor"] == m.name]
            if m_errs:
                monitors_summary[m.name] = {
                    "count": len(m_errs),
                    "mean_error_px": round(float(np.mean(m_errs)), 1),
                    "median_error_px": round(float(np.median(m_errs)), 1)
                }

        summary_report = {
            "mode": mode_label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_targets": len(results_per_target),
            "mean_radial_error_px": round(mean_radial_error, 1),
            "median_radial_error_px": round(median_radial_error, 1),
            "std_radial_error_px": round(std_radial_error, 1),
            "p90_error_px": round(p90_error, 1),
            "monitors": monitors_summary,
            "targets": results_per_target
        }

        # Print formatted ASCII table
        print("\n" + "=" * 65)
        print(f"📊 EMPIRICAL ACCURACY BENCHMARK RESULTS [{mode_label}]")
        print("=" * 65)
        print(f"  • Total Targets Tested:        {len(results_per_target)}")
        print(f"  • Mean Radial Error (MRE):     {mean_radial_error:.1f} px")
        print(f"  • Median Radial Error:         {median_radial_error:.1f} px")
        print(f"  • Standard Deviation (σ):      {std_radial_error:.1f} px")
        print(f"  • 90th Percentile Error:       {p90_error:.1f} px")
        print("-" * 65)
        print("Per-Monitor Breakdown:")
        for mon_name, stats in monitors_summary.items():
            print(f"  [{mon_name} Monitor]: Mean={stats['mean_error_px']} px | Median={stats['median_error_px']} px")
        print("=" * 65 + "\n")

        # Save report to JSON
        report_filename = f"accuracy_report_{'baseline' if self.force_baseline else 'calibrated'}.json"
        with open(report_filename, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)
        print(f"[Benchmark] Full metrics report written to '{report_filename}'.")

        return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EyeControl Ground-Truth Accuracy Benchmark")
    parser.add_argument("--baseline", action="store_true", help="Force benchmark using baseline linear heuristic")
    args = parser.parse_args()

    evaluator = AccuracyEvaluator(force_baseline=args.baseline)
    evaluator.run_benchmark()
