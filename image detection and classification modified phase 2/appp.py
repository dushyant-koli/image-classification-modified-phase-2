import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image  # Import for handling uploaded images

# Load the pre-trained model
model = load_model('cifar10_model.h5')

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((32, 32))  # Resize to 32x32 (CIFAR-10 input size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Streamlit app
st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    img = Image.open(uploaded_file)  # Open the uploaded image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        predicted_class, confidence = predict_image(img)
        st.success(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
