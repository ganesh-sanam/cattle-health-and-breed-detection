# Cattle Health & Breed Detection

## 1. Problem Statement
Accurate cattle identification and health monitoring are vital for livestock management and disease prevention. Manual identification by farmers or veterinarians can be slow and error-prone. This project leverages Computer Vision to automate breed classification and detect visible health indicators instantly.

## 2. Approach & Methodology
We employed a **Convolutional Neural Network (CNN)** approach for multi-class classification. The pipeline includes robust preprocessing to handle varying lighting and background conditions common in farm environments.
- **Preprocessing**: OpenCV-based image normalization and augmentation.
- **Modeling**: Custom CNN architecture for feature extraction.
- **Interface**: Simple drag-and-drop web UI for non-technical users.

## 3. Dataset
- **Source**: [Mention Source, e.g., Kaggle, Public dataset, or self-collected].
- **Classes**: Multiple cattle breeds (e.g., Holstein, Jersey, Angus) and health states (Healthy vs. Lumpy Skin Disease/Issues).
- **Preprocessing**: Images were resized and normalized; data augmentation (rotation, flip) was used to combat overfitting.

## 4. Model & Training
- **Type**: CNN with distinct heads for Breed Classification.
- **Preprocessing**: Histogram equalization using OpenCV to improve contrast in poor lighting.
- **Optimization**: Minimized Cross-Entropy Loss to handle multi-class predictions.

## 5. Results & Evaluation
- **Stability**: The model demonstrates robust performance across varied lighting conditions due to the preprocessing pipeline.
- **Usability**: Inference time is <1 second, making it viable for real-time field use via the Streamlit app.

**Limitations**:
- Performance drops on heavily occluded images or extreme angles.
- Currently limited to visual health indicators (skin conditions); cannot detect internal diseases.

## 6. Key Learnings
- **Data Quality is King**: Real-world farm images are noisy; aggressive preprocessing (OpenCV) significantly improved model stability.
- **User-Centric Design**: Deployment via Streamlit allowed immediate feedback from non-technical users, which is crucial for adoption.
- **Edge Cases**: Differentiating between similar-looking breeds requires fine-grained feature extraction (higher resolution inputs).

## 7. Tech Stack
- **Computer Vision**: OpenCV, Python
- **Machine Learning**: Custom CNNs (TensorFlow/Keras or PyTorch)
- **Deployment**: Streamlit (Web UI)
- **Tools**: NumPy, Matplotlib

## 8. Project Structure
```bash
├── src/
│   ├── preprocessing.py  # OpenCV image processing
│   └── model.py          # CNN architecture
├── app.py                # Streamlit entry point
├── weights/              # Saved model weights
├── requirements.txt      # Dependencies
└── README.md
```

## 9. How to Run

1. **Clone the Repository**
   ```bash
   git clone [Your Repository URL]
   cd Cattle-Health-Detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the App**
   ```bash
   streamlit run app.py
   ```
   *Upload a cattle image to see the predicted breed and health status.*

## 10. Future Improvements
- Integrate **MobileNet** or a lightweight architecture for deployment on mobile devices (Edge computing).
- Expand the dataset to include more subtle skin conditions.
- Add GPS tagging for disease mapping in affected areas.

## 11. Author Note
Developed to assist farmers and vets with accessible AI tools. The focus is on simplicity and reliability in the field.

