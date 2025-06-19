import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load U-Net model
model = load_model("unet_cell_membrane.h5")

st.set_page_config(page_title="Cell Membrane Segmentation", layout="centered")
st.title("🧬 Cell Membrane Segmentation")

uploaded_file = st.file_uploader("Upload a grayscale microscopy image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((512, 512))
    image_array = np.array(image) / 255.0
    input_tensor = image_array[np.newaxis, ..., np.newaxis]

    pred = model.predict(input_tensor)[0]
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    st.subheader("Predicted Mask")
    st.image(pred_mask.squeeze(), use_column_width=True, clamp=True)
