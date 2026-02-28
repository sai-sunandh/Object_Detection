class Config:
    """Configuration settings for the object detection application."""
    
    # Model settings
    MODEL_PATH = 'src/models/weights/model_weights.h5'  # Path to the pre-trained model weights
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold for predictions
    NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold

    # Input settings
    INPUT_IMAGE_SIZE = (224, 224)  # Size to which input images will be resized

    # Other settings
    MAX_DETECTIONS = 100  # Maximum number of detections to return per image
    CLASS_NAMES = ['class1', 'class2', 'class3']  # List of class names for the model

    @staticmethod
    def get_model_path():
        return Config.MODEL_PATH

    @staticmethod
    def get_class_names():
        return Config.CLASS_NAMES