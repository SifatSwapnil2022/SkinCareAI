import numpy as np
from PIL import Image
import io

CLASS_NAMES = [
    "1. Eczema 1677",
    "10. Warts Molluscum and other Viral Infections - 2103",
    "2. Melanoma 15.75k",
    "3. Atopic Dermatitis - 1.25k",
    "4. Basal Cell Carcinoma (BCC) 3323",
    "5. Melanocytic Nevi (NV) - 7970",
    "6. Benign Keratosis-like Lesions (BKL) 2624",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k",
]

# Clean display names (for UI)
DISPLAY_NAMES = {
    "1. Eczema 1677":                                               "Eczema",
    "10. Warts Molluscum and other Viral Infections - 2103":        "Warts & Molluscum",
    "2. Melanoma 15.75k":                                           "Melanoma",
    "3. Atopic Dermatitis - 1.25k":                                 "Atopic Dermatitis",
    "4. Basal Cell Carcinoma (BCC) 3323":                           "Basal Cell Carcinoma",
    "5. Melanocytic Nevi (NV) - 7970":                              "Melanocytic Nevi",
    "6. Benign Keratosis-like Lesions (BKL) 2624":                  "Benign Keratosis",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "Psoriasis & Lichen Planus",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k":       "Seborrheic Keratoses",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "Tinea & Ringworm",
}


def preprocess_for_keras(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def preprocess_for_yolo(image_bytes: bytes) -> str:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(image_bytes)
    tmp.close()
    return tmp.name


def image_to_base64(image_bytes: bytes) -> str:
    import base64
    return base64.b64encode(image_bytes).decode("utf-8")