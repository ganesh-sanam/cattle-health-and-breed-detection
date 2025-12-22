import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Cattle AI Predictor", layout="centered")

st.title("🐄 Cattle Breed & Disease Prediction")
st.write("Upload an image of cattle and get predictions.")

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

uploaded_file = st.file_uploader(
    "Upload cattle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            # Convert image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()

            # Send request to Flask
            files = {"file": img_bytes}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                st.success("✅ Prediction Complete")

                st.subheader("Results")
                st.write(f"**Breed:** {result['breed']} ({result['breed_confidence']}%)")
                st.write(f"**Disease:** {result['disease']} ({result['disease_confidence']}%)")
            else:
                st.error("❌ Prediction failed. Check Flask server.")
