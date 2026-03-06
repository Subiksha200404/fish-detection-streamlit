import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import cv2
import pandas as pd
import gdown

# -------------------------
# Page Config (MUST BE FIRST)
# -------------------------
st.set_page_config(
    page_title="Underwater Fish and Coral Detection",
    page_icon="🌊",
    layout="wide"
)

# -------------------------
# Download Model
# -------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = "https://drive.google.com/uc?id=1jC8l4yqmqXCwSEfPDgrsTe2lrXn7gHMS"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# Styling
# -------------------------
st.markdown("""
<style>
.main {
background-color: #0E1117;
}
h1 {color:#00BFFF;text-align:center;}
h2,h3 {color:#1E90FF;}
.stButton>button {
background-color:#1E90FF;
color:white;
border-radius:8px;
height:3em;
width:100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------
# Sidebar
# -------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home","📷 Image Detection","🎥 Video Detection"]
)

# =====================================================
# HOME
# =====================================================
if page == "🏠 Home":

    st.title("Underwater Fish and Coral Detection")

    st.markdown("""
### 🔍 Project Overview

This web application demonstrates deployment of a trained **YOLO model**
for detecting underwater fish and coral species.

---

### 🚀 Features

- Upload images
- Upload videos
- Automatic object detection
- Bounding boxes
- Confidence scores
- Download processed results

---

Developed as part of **Final Year Project**.
""")

# =====================================================
# IMAGE DETECTION
# =====================================================
elif page == "📷 Image Detection":

    st.title("Image Detection")

    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["jpg","png","jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            st.subheader(uploaded_file.name)

            image = Image.open(uploaded_file)

            st.image(image, caption="Original Image", width=500)

            with st.spinner("Running detection..."):

                results = model(image)

            result_img = results[0].plot()

            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            st.image(result_img, caption="Detection Result", width=500)

            boxes = results[0].boxes
            names = results[0].names

            if boxes is not None:

                classes = boxes.cls.tolist()

                detected = [names[int(i)] for i in classes]

                if len(detected) > 0:

                    df = pd.DataFrame(detected, columns=["Object"])

                    summary = df["Object"].value_counts()

                    st.subheader("Detection Summary")

                    st.bar_chart(summary)

# =====================================================
# VIDEO DETECTION
# =====================================================
elif page == "🎥 Video Detection":

    st.title("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4","avi","mov"]
    )

    if uploaded_video:

        st.video(uploaded_video)

        # Save uploaded video
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video uploaded successfully")

        if st.button("Run Detection"):

            with st.spinner("Processing video..."):

                results = model.predict(
                    source=video_path,
                    save=True,
                    project=temp_dir,
                    name="result",
                    exist_ok=True
                )

            st.success("Processing complete")

            # Find output video automatically
            output_folder = os.path.join(temp_dir, "result")

            output_video_path = None

            for file in os.listdir(output_folder):
                if file.endswith((".mp4",".avi",".mov")):
                    output_video_path = os.path.join(output_folder, file)
                    break

            if output_video_path and os.path.exists(output_video_path):

                st.subheader("Processed Video")

                st.video(output_video_path)

                with open(output_video_path, "rb") as file:
                    video_bytes = file.read()

                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

            else:

                st.error("Output video not found.")
