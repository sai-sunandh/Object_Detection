from flask import jsonify
import cv2
import numpy as np
from models.detector import ObjectDetector
from config import Config

class Inference:
    def __init__(self):
        self.detector = ObjectDetector(model_path=Config.MODEL_PATH)

    def process_image(self, image):
        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (Config.INPUT_WIDTH, Config.INPUT_HEIGHT))
        image = image / 255.0
        return image

    def predict(self, image):
        processed_image = self.process_image(image)
        predictions = self.detector.run_inference(processed_image)
        return predictions

    def handle_inference(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Image not found"}), 404
        
        predictions = self.predict(image)
        return jsonify(predictions)