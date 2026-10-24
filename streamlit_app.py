import os
import urllib.request
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Traffic Monitoring System", layout="wide")
st.title("🚗 Traffic Monitoring & Detection System")

# Cache the model so it downloads and loads only once into RAM
@st.cache_resource
def load_model():
    model_path = "best.pt"
    # If the weight file isn't in the root folder, download it directly from Hugging Face
    if not os.path.exists(model_path):
        with st.spinner("Downloading YOLO model weights..."):
            url = "https://huggingface.co/spaces/Siva11234567/traffic/resolve/main/best.pt"
            urllib.request.urlretrieve(url, model_path)
    return YOLO(model_path)

model = load_model()

uploaded_file = st.file_uploader("Upload Traffic Image or Video Frame", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
    if st.button("Run Traffic Analysis"):
        with st.spinner("Processing image..."):
            results = model(image)
            res_plotted = results[0].plot()
            
        with col2:
            st.subheader("Detection Output")
            st.image(res_plotted, use_column_width=True)