from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_breed, predict_disease
import os

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    image_path = "temp.jpg"
    file.save(image_path)

    # 🔮 Model predictions
    breed, breed_conf = predict_breed(image_path)
    disease, disease_conf = predict_disease(image_path)

    os.remove(image_path)

    # ✅ THIS IS STEP 3 (VERY IMPORTANT)
    return jsonify({
        "breed": breed,
        "breed_confidence": round(breed_conf * 100, 2),
        "disease": disease,
        "disease_confidence": round(disease_conf * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
