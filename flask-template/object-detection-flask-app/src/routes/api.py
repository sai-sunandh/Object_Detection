from flask import Blueprint, request, jsonify
from src.inference import run_inference

api = Blueprint('api', __name__)

@api.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    
    try:
        results = run_inference(image_file)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500