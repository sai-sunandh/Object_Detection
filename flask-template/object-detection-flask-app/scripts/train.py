import os
import json
import numpy as np
import tensorflow as tf
from src.models.detector import ObjectDetector
from src.utils.preprocessing import preprocess_image
from src.utils.postprocessing import postprocess_predictions

# Configuration settings
with open('src/config.py') as config_file:
    config = json.load(config_file)

# Load dataset
def load_data(data_dir):
    images = []
    annotations = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(data_dir, filename)
            images.append(img_path)
            # Load corresponding annotation if exists
            annotation_path = os.path.join(data_dir, 'annotations', filename.replace('.jpg', '.json').replace('.png', '.json'))
            if os.path.exists(annotation_path):
                with open(annotation_path) as annotation_file:
                    annotations.append(json.load(annotation_file))
    return images, annotations

# Training loop
def train_model():
    # Load images and annotations
    images, annotations = load_data('data/images')
    
    # Initialize the object detector
    detector = ObjectDetector(model_path=config['model_path'])

    for epoch in range(config['num_epochs']):
        for img_path, annotation in zip(images, annotations):
            # Preprocess image
            image = preprocess_image(img_path)
            # Train the model with the image and its annotation
            detector.train(image, annotation)

        # Save model after each epoch
        detector.save_model(f"models/weights/epoch_{epoch+1}.h5")

if __name__ == "__main__":
    train_model()