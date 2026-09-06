import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
import tempfile
from gaze_calibration import PolynomialRidgeModel, PostureNormalizer

class TestGazeCalibration(unittest.TestCase):
    def test_polynomial_expansion(self):
        model = PolynomialRidgeModel(degree=2, alpha=1.0)
        # 1 sample, 3 features: [x1, x2, x3]
        X = np.array([[2.0, 3.0, 4.0]])
        phi = model._expand_features(X)
        # Expected:
        # 1 (const)
        # x1, x2, x3 (3 linear)
        # x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2 (6 quadratic)
        # Total = 1 + 3 + 6 = 10 features
        self.assertEqual(phi.shape, (1, 10))
        expected = np.array([[1.0, 2.0, 3.0, 4.0, 4.0, 6.0, 8.0, 9.0, 12.0, 16.0]])
        np.testing.assert_allclose(phi, expected, rtol=1e-5)

    def test_synthetic_fit_predict(self):
        # Generate synthetic 2D surface:
        # screen_x = 500 + 300*x1 - 100*x2 + 50*x1^2 + 20*x1*x2
        # screen_y = 400 - 150*x1 + 250*x2 - 30*x2^2
        np.random.seed(42)
        N = 50
        x1 = np.random.uniform(-0.5, 0.5, N)
        x2 = np.random.uniform(-0.5, 0.5, N)
        X = np.column_stack([x1, x2])
        
        target_x = 500.0 + 300.0*x1 - 100.0*x2 + 50.0*(x1**2) + 20.0*(x1*x2)
        target_y = 400.0 - 150.0*x1 + 250.0*x2 - 30.0*(x2**2)
        Y = np.column_stack([target_x, target_y])

        model = PolynomialRidgeModel(degree=2, alpha=0.001)
        model.fit(X, Y)

        # Predict on new points
        X_test = np.array([[0.1, -0.2], [-0.3, 0.4]])
        exp_x = 500.0 + 300.0*X_test[:,0] - 100.0*X_test[:,1] + 50.0*(X_test[:,0]**2) + 20.0*(X_test[:,0]*X_test[:,1])
        exp_y = 400.0 - 150.0*X_test[:,0] + 250.0*X_test[:,1] - 30.0*(X_test[:,1]**2)
        exp_Y = np.column_stack([exp_x, exp_y])

        pred_Y = model.predict(X_test)
        np.testing.assert_allclose(pred_Y, exp_Y, atol=0.5)

    def test_save_load_profile(self):
        model = PolynomialRidgeModel(degree=2, alpha=1.0)
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2) * 1000.0
        model.fit(X, Y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            model.save_profile(temp_path, metadata={"user": "test", "d_iod_calib": 65.0})
            
            loaded_model = PolynomialRidgeModel.load_profile(temp_path)
            self.assertTrue(loaded_model.is_calibrated)
            self.assertEqual(loaded_model.metadata.get("d_iod_calib"), 65.0)

            # Test predictions match
            p1 = model.predict(X[:5])
            p2 = loaded_model.predict(X[:5])
            np.testing.assert_allclose(p1, p2, rtol=1e-5)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_posture_normalizer(self):
        class DummyLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        landmarks = [None] * 500
        # Landmark 33 (right eye outer) and 263 (left eye outer)
        landmarks[33] = DummyLandmark(0.3, 0.5, 0.0)
        landmarks[263] = DummyLandmark(0.7, 0.5, 0.0)

        normalizer = PostureNormalizer()
        iod = normalizer.compute_iod(landmarks, frame_w=1000, frame_h=1000)
        self.assertAlmostEqual(iod, 400.0, places=2)

        normalizer.set_baseline(iod)
        # If user moves closer, iod increases to 480
        ratio = normalizer.get_depth_scale(480.0)
        self.assertAlmostEqual(ratio, 1.2, places=2)

if __name__ == "__main__":
    unittest.main()
