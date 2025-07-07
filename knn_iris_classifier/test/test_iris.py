import unittest
from iris_knn import knn_predict

class TestKNNClassifier(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            (5.1, 3.5, 1.4, 0.2),
            (6.7, 3.1, 4.4, 1.4),
            (7.2, 3.6, 6.1, 2.5),
            (4.9, 2.5, 4.5, 1.7),
            (5.9, 3.0, 5.1, 1.8) 
        ]
        sample_classes = [
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica",
            "Iris-versicolor",
            "Iris-virginica"
        ]
        self.expected_classes = {
            sample: expected_class for sample, expected_class in zip(self.sample_data, sample_classes)
        }

    def test_classifier(self):
        k_values = [1, 3, 5]
        for k in k_values:
            print(f"Testing with k={k}")
            for i, sample in enumerate(self.sample_data, start=1):
                predicted_class = knn_predict(sample, 3 )
                print(f"Test Case {i}: Input = {sample} => Predicted Iris Type: {predicted_class}")
                expected_class = self.expected_classes[sample]
                self.assertEqual(predicted_class, expected_class, f"Failed for sample {i} with k={k}")