import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import time
import math

from eye_tracking_mouse import DwellClickEngine, IVTClassifier, MouseController

class TestErgonomics(unittest.TestCase):
    def test_ivt_classifier_states(self):
        classifier = IVTClassifier(saccade_threshold_px_s=1000.0, fixation_threshold_px_s=200.0)

        t0 = 1000.0
        # Initialize
        state = classifier.update(500.0, 500.0, t=t0)
        self.assertEqual(state, "FIXATION")

        # 1. Slow movement: 50px in 0.5s -> 100 px/s -> FIXATION
        state = classifier.update(550.0, 500.0, t=t0 + 0.5)
        self.assertEqual(state, "FIXATION")
        self.assertAlmostEqual(classifier.current_velocity, 100.0, delta=1.0)

        # 2. Moderate movement: 250px in 0.5s -> 500 px/s -> PURSUIT
        state = classifier.update(800.0, 500.0, t=t0 + 1.0)
        self.assertEqual(state, "PURSUIT")
        self.assertAlmostEqual(classifier.current_velocity, 500.0, delta=1.0)

        # 3. High-speed saccade: 800px in 0.1s -> 8000 px/s -> SACCADE
        state = classifier.update(1600.0, 500.0, t=t0 + 1.1)
        self.assertEqual(state, "SACCADE")
        self.assertAlmostEqual(classifier.current_velocity, 8000.0, delta=10.0)

    def test_dwell_clicking_countdown_and_fire(self):
        dwell = DwellClickEngine(dwell_time=0.5, dwell_radius=20.0, cooldown_time=0.3)

        t0 = 1000.0
        # Initial point
        clicked, progress = dwell.update(500.0, 400.0, is_fixating=True, t=t0)
        self.assertFalse(clicked)
        self.assertEqual(progress, 0.0)

        # Halfway through dwell time: 0.25s / 0.5s -> ~50%
        clicked, progress = dwell.update(505.0, 402.0, is_fixating=True, t=t0 + 0.25)
        self.assertFalse(clicked)
        self.assertAlmostEqual(progress, 0.50, delta=0.05)

        # Complete dwell time: 0.50s -> Click fired!
        clicked, progress = dwell.update(504.0, 401.0, is_fixating=True, t=t0 + 0.50)
        self.assertTrue(clicked)
        self.assertEqual(progress, 1.0)

        # Immediate next frame should be in cooldown
        clicked, progress = dwell.update(504.0, 401.0, is_fixating=True, t=t0 + 0.55)
        self.assertFalse(clicked)
        self.assertEqual(progress, 0.0)

    def test_dwell_clicking_cancels_on_movement(self):
        dwell = DwellClickEngine(dwell_time=0.5, dwell_radius=20.0, cooldown_time=0.3)

        t0 = 1000.0
        dwell.update(500.0, 400.0, is_fixating=True, t=t0)
        dwell.update(505.0, 402.0, is_fixating=True, t=t0 + 0.25)

        # Gaze jumps away by 80px (> radius 20px) -> Progress resets to 0.0
        clicked, progress = dwell.update(580.0, 400.0, is_fixating=True, t=t0 + 0.30)
        self.assertFalse(clicked)
        self.assertEqual(progress, 0.0)

    def test_mouse_controller_scroll(self):
        mouse = MouseController()
        # Should execute without throwing any exception on Windows or non-Windows
        try:
            mouse.scroll(120)
            mouse.scroll(-120)
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
