import tensorflow as tf

print("TF:", tf.__version__)

model = tf.keras.models.load_model(
    "breed_model.h5",
    compile=False
)

model.save("breed_model_converted.keras")

print("✅ Conversion successful")
