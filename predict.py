from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# --------------------------------------------------
# Load models (safe for Keras 3.x)
# --------------------------------------------------
disease_model = load_model("model1.keras", compile=False)
breed_model   = load_model("breed_model_clean.keras", compile=False)

print("✅ Disease model loaded")
print("✅ Breed model loaded")

# --------------------------------------------------
# Class labels
# --------------------------------------------------
disease_classes = ["healthy", "lumpy", "mastitis"]

breed_classes = [
    "Banni buffalo",
    "Bhadawari buffalo",
    "Gir",
    "Godavari buffalo",
    "Jafarabadi buffalo",
    "Murrah buffalo",
    "Ongole",
    "Poda Thurpu",
    "Red Sindhi",
    "Sahiwal",
    "jersy",
    "punganur"
]

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --------------------------------------------------
# Disease prediction with healthy fallback
# --------------------------------------------------
def predict_disease(image_path):
    preds = disease_model.predict(preprocess(image_path))[0]

    DISEASE_THRESHOLD = 0.60  # 60%

    # index of healthy class
    healthy_index = disease_classes.index("healthy")

    # find best disease excluding healthy
    best_disease_idx = np.argmax([
        p if i != healthy_index else 0
        for i, p in enumerate(preds)
    ])

    best_disease_conf = preds[best_disease_idx]

    if best_disease_conf >= DISEASE_THRESHOLD:
        disease = disease_classes[best_disease_idx]
        confidence = best_disease_conf
    else:
        disease = "Healthy (No disease detected)"
        confidence = 1 - best_disease_conf

    return disease, float(confidence)

# --------------------------------------------------
# Breed prediction (unchanged)
# --------------------------------------------------
def predict_breed(image_path):
    preds = breed_model.predict(preprocess(image_path))[0]
    idx = np.argmax(preds)
    return breed_classes[idx], float(preds[idx])
