import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from utils.preprocess import CLASS_NAMES, preprocess_for_keras

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(CLASS_NAMES), activation='softmax')(x)
    _model  = Model(inputs=base.input, outputs=outputs)
    _model.load_weights("weights/best_resnet50.weights.h5")
    return _model


def predict(image_bytes: bytes) -> dict:
    model = load_model()
    arr   = preprocess_for_keras(image_bytes)
    arr   = tf.keras.applications.resnet50.preprocess_input(arr)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {
        "disease":         CLASS_NAMES[idx],
        "confidence":      round(float(probs[idx]), 4),
        "all_predictions": {CLASS_NAMES[i]: round(float(p), 4)
                            for i, p in enumerate(probs)}
    }