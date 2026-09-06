import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np

from eye_tracking_mouse import EyeTrackerConfig, BlinkDetector, DisplayManager, DualMonitorGazeEstimator
from gaze_calibration import PolynomialRidgeModel

class TestTrackerIntegration(unittest.TestCase):
    def test_blink_detector_neural_blendshapes(self):
        config = EyeTrackerConfig(
            USE_NEURAL_BLENDSHAPES=True,
            BLENDSHAPE_BLINK_THRESHOLD=0.45,
            CONSECUTIVE_FRAMES_TRIGGER=3
        )
        detector = BlinkDetector(config)

        # 1. Bilateral blink suppression
        action, telemetry = detector.process_eyes(
            left_ear=0.25, right_ear=0.25,
            blendshapes={"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.88}
        )
        self.assertIsNone(action)
        self.assertTrue(telemetry["bilateral_blink"])
        self.assertEqual(telemetry["signal_source"], "neural_blendshape")

        # 2. Deliberate left wink held for 3 frames -> Left Click
        for i in range(2):
            action, _ = detector.process_eyes(
                left_ear=0.25, right_ear=0.25,
                blendshapes={"eyeBlinkLeft": 0.75, "eyeBlinkRight": 0.05}
            )
            self.assertIsNone(action)

        action, _ = detector.process_eyes(
            left_ear=0.25, right_ear=0.25,
            blendshapes={"eyeBlinkLeft": 0.75, "eyeBlinkRight": 0.05}
        )
        self.assertEqual(action, "left_click")

        # 3. Re-arming: eye opens back up below rearm threshold
        action, telemetry = detector.process_eyes(
            left_ear=0.25, right_ear=0.25,
            blendshapes={"eyeBlinkLeft": 0.15, "eyeBlinkRight": 0.05}
        )
        self.assertIsNone(action)
        self.assertFalse(detector.left_click_latched)

    def test_gaze_estimator_calibrated_vs_fallback(self):
        config = EyeTrackerConfig(USE_POLYNOMIAL_CALIBRATION=True)
        display = DisplayManager(camera_on_monitor="right")
        estimator = DualMonitorGazeEstimator(config, display)

        # When uncalibrated, is_calibrated should be False
        estimator.calib_model = None
        self.assertFalse(estimator.is_calibrated)

        # Fallback mapping produces valid screen coords within virtual desktop
        sx, sy, active_mon = estimator.map_to_screen(0.5, 0.5, 0.5, 0.5)
        self.assertGreaterEqual(sx, display.virtual_left)
        self.assertLessEqual(sx, display.virtual_left + display.virtual_width)

        # When calibrated model is attached:
        # Create a dummy model predicting virtual coordinate (800, 400)
        calib_model = PolynomialRidgeModel(degree=2, alpha=1.0)
        X_dummy = np.random.randn(10, 6)
        Y_dummy = np.zeros((10, 2))
        Y_dummy[:, 0] = 800.0
        Y_dummy[:, 1] = 400.0
        calib_model.fit(X_dummy, Y_dummy)

        estimator.calib_model = calib_model
        self.assertTrue(estimator.is_calibrated)

        estimator.current_features = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        sx, sy, active_mon = estimator.map_to_screen(0.5, 0.5, 0.5, 0.5)
        self.assertAlmostEqual(sx, 800, delta=20)
        self.assertAlmostEqual(sy, 400, delta=20)

if __name__ == "__main__":
    unittest.main()
