class ObjectDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        # Load the pre-trained model from the specified path
        pass

    def run_inference(self, image):
        # Run inference on the input image and return the results
        pass

    def process_results(self, results):
        # Process the inference results and return formatted output
        pass