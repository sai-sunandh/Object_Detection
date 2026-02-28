import unittest
from src.inference import ObjectDetector

class TestInference(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetector(model_path='src/models/weights/model_weights.h5')

    def test_load_model(self):
        self.assertIsNotNone(self.detector.model)

    def test_inference(self):
        test_image = 'data/images/test_image.jpg'
        predictions = self.detector.run_inference(test_image)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

    def test_invalid_image(self):
        invalid_image = 'data/images/invalid_image.jpg'
        with self.assertRaises(Exception):
            self.detector.run_inference(invalid_image)

if __name__ == '__main__':
    unittest.main()