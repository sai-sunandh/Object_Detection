def filter_predictions(predictions, confidence_threshold=0.5):
    filtered = []
    for pred in predictions:
        if pred['confidence'] >= confidence_threshold:
            filtered.append(pred)
    return filtered

def non_max_suppression(predictions, iou_threshold=0.5):
    if len(predictions) == 0:
        return []

    boxes = []
    scores = []
    for pred in predictions:
        boxes.append(pred['bbox'])
        scores.append(pred['confidence'])

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_threshold)
    return [predictions[i] for i in indices.flatten()] if len(indices) > 0 else []