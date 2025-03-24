import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import json
from model4 import EfficientCNN  # Import model structure

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = EfficientCNN(K=39)
model.load_state_dict(torch.load("plant_disease_model_1_state_dict.pt", map_location=device))
model.to(device)
model.eval()

# Load disease information
disease_info_path = "disease_info.csv"
try:
    disease_info = pd.read_csv(disease_info_path, encoding="cp1252")
except FileNotFoundError:
    st.error(f"⚠️ Disease information file not found at: {disease_info_path}")
    disease_info = None

# Load evaluation metrics from JSON
try:
    with open("evaluation_results.json", "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {"accuracy": "N/A", "average_loss": "N/A"}

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Function to predict disease
def predict_disease(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    index = probabilities.argmax().item()
    
    disease_name = disease_info["disease_name"][index] if disease_info is not None else f"Class {index}"
    confidence = probabilities[index].item()
    
    return disease_name, confidence

# Function to capture image using OpenCV
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return None
    
    st.info("📷 Capturing image...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Failed to capture image.")
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(frame_rgb)
    
    return captured_image

# Streamlit UI Styling
st.title("🌿 Plant Disease Detection")
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        text-align: center;
    }
    .scroll-container {
        max-height: 150px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Image selection section
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None

# File uploader and live capture buttons
st.markdown("### Choose an Option:")
options = st.selectbox("Select", ["Upload Image", "Capture Image", "Show Model Accuracy"], index=0)

if options == "Upload Image":
    uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state["selected_image"] = image

elif options == "Capture Image":
    if st.button("📸 Capture Image"):
        captured_image = capture_image()
        if captured_image:
            st.session_state["selected_image"] = captured_image

elif options == "Show Model Accuracy":
    st.write(f"📊 Model Test Accuracy: {metrics['accuracy']:.2f}%")
    st.write(f"📉 Average Loss: {metrics['average_loss']:.4f}")

st.markdown("---")

# Clear image button
if st.button("🔄 Clear All"):
    st.session_state["selected_image"] = None

# Display the selected image and prediction result
if st.session_state["selected_image"]:
    st.image(st.session_state["selected_image"], caption="📸 Selected Image", use_column_width=True)
    
    with st.spinner("🔍 Analyzing Image..."):
        disease, confidence = predict_disease(st.session_state["selected_image"])
        
    st.success(f"🌱 Predicted Disease: {disease}")
    st.info(f"🔬 Confidence: {confidence * 100:.2f}%")
