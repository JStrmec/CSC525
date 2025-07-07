import unittest
import numpy as np
from polynomial_salary_regressor import PolynomialSalaryRegressor


class TestPolynomialSalaryRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PolynomialSalaryRegressor(degree=2)
        cls.model.load_data("resources/Salary_data.csv")
        cls.model.train()

    def test_data_loaded(self):
        self.assertEqual(len(self.model.X), 30)
        self.assertEqual(len(self.model.y), 30)
        self.assertIsInstance(self.model.X, np.ndarray)
        self.assertIsInstance(self.model.y, np.ndarray)

    def test_model_training(self):
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.poly_features)

    def test_prediction_output(self):
        prediction = self.model.predict([3])
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape[0], 1)
        self.assertTrue(prediction[0] > 0)

    def test_prediction_multiple_inputs(self):
        predictions = self.model.predict([1, 2, 3])
        self.assertEqual(predictions.shape, (3,))
        self.assertTrue(all(p > 0 for p in predictions))

    def test_evaluate_mse(self):
        mse = self.model.evaluate()
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0)


if __name__ == "__main__":
    unittest.main()
