import numpy as np
import os
import torch  # Required for the fix
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel # Required for allowlisting
from utils.preprocess import CLASS_NAMES, preprocess_for_yolo

_model = None

def load_model():
    global _model
    if _model is None:
        
        
        torch.serialization.add_safe_globals([ClassificationModel])
        
        
        _model = YOLO("weights/yolov8/best.pt")
    return _model


def predict(image_bytes: bytes) -> dict:
    model    = load_model()
    tmp_path = preprocess_for_yolo(image_bytes)
    try:
        results = model.predict(tmp_path, imgsz=224, verbose=False)
        probs   = results[0].probs.data.cpu().numpy()
        idx     = int(np.argmax(probs))
        return {
            "disease":         CLASS_NAMES[idx],
            "confidence":      round(float(probs[idx]), 4),
            "all_predictions": {CLASS_NAMES[i]: round(float(p), 4)
                                for i, p in enumerate(probs)}
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)