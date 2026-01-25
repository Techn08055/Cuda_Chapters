def postprocess_yolo_results(results):
    """
    Postprocesses YOLOv11 results from ultralytics into a serializable format.
    """
    detections = []
    # results is a list of ultralytics.engine.results.Results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates, confidence, and class id
            b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]
            
            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [round(x, 2) for x in b]
            })
    return detections
