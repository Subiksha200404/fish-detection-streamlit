import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import gdown

# -------------------------
# Download Model (Google Drive)
# -------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        url = "https://drive.google.com/uc?id=1jC8l4yqmqXCwSEfPDgrsTe2lrXn7gHMS"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Underwater Fish and Coral Detection Dashboard",
    page_icon="🌊",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #00BFFF;
    text-align: center;
}
h2, h3 {
    color: #1E90FF;
}
.stButton>button {
    background-color: #1E90FF;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model (Cached)
# -------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------
# Sidebar Navigation
# -------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📷 Image Detection", "🎥 Video Detection"]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":
    st.title("Underwater Fish and Coral Detection")
    st.markdown("""
    ### 🔍 Project Overview

    This web application demonstrates deployment of a trained YOLO model
    for detecting underwater fish and coral species.

    ---
    ### 🚀 Features
    - Upload single or multiple images
    - Upload video files
    - Automatic object detection
    - Bounding box visualization
    - Confidence score display

    ---
    Developed as part of Final Year Project.
    """)

# =====================================================
# IMAGE DETECTION
# =====================================================
elif page == "📷 Image Detection":
    st.title("Image Detection")

    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")

            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            with st.spinner("Running detection..."):
                results = model(image, imgsz=640)

            result_img = results[0].plot()

            st.image(result_img, caption="Detection Result", use_column_width=True)
            st.success("Detection Complete ✅")

# =====================================================
# VIDEO DETECTION
# =====================================================
elif page == "🎥 Video Detection":
    st.title("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        # Create temporary folder
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_video.name)

        # Save video with original extension
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info("Video uploaded successfully.")

        if st.button("Run Detection"):

            with st.spinner("Processing video... This may take some time ⏳"):
                model(video_path, save=True)

            st.success("Video processed successfully! ✅")
            st.write("Processed video saved inside 'runs/detect/' folder.")
