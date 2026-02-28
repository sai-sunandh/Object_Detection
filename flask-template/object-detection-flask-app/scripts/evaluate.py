import os
import sys
import json
import cv2
import numpy as np

# ensure project 'src' is importable when running this script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.models.detector import ObjectDetector
from src.utils.preprocessing import preprocess_image
from src.utils.postprocessing import postprocess_detections

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_model(detector, test_images_path):
    results = []
    for image_file in os.listdir(test_images_path):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(test_images_path, image_file)
            image = cv2.imread(image_path)
            processed_image = preprocess_image(image)
            detections = detector.predict(processed_image)
            postprocessed_results = postprocess_detections(detections)
            results.append({
                'image': image_file,
                'detections': postprocessed_results
            })
    return results

if __name__ == "__main__":
    config_path = 'src/config.json'  # Update with the actual path to your config file
    test_images_path = 'data/images'  # Update with the actual path to your test images

    config = load_config(config_path)
    detector = ObjectDetector(config['model_path'])
    detector.load_model()

    evaluation_results = evaluate_model(detector, test_images_path)

    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print("Evaluation completed. Results saved to evaluation_results.json")