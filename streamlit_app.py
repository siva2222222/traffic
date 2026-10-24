import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Traffic Monitoring System", layout="wide")
st.title("🚗 Traffic Monitoring & Detection System")

@st.cache_resource
def load_model():
    # Automatically downloads official YOLOv8 nano model
    return YOLO("yolov8n.pt") 

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