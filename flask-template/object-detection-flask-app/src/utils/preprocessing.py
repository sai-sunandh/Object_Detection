def resize_image(image, target_size):
    from PIL import Image
    return image.resize(target_size, Image.ANTIALIAS)

def normalize_image(image):
    import numpy as np
    return np.array(image) / 255.0

def preprocess_image(image, target_size):
    resized_image = resize_image(image, target_size)
    normalized_image = normalize_image(resized_image)
    return normalized_image