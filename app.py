import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
from PIL import Image

# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="Underwater Fish and Coral Detection",
    page_icon="🐟",
    layout="wide"
)

# -------------------------------
# FORCE LIGHT MODE
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: white;
    color: black;
}
h1,h2,h3 {
    color:#0A84FF;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("best.pt")

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📷 Image Detection", "🎥 Video Detection"]
)

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "🏠 Home":

    st.title("Underwater Fish and Coral Detection")

    st.header("🔎 Project Overview")

    st.write(
        "This web application demonstrates deployment of a trained YOLO model "
        "for detecting underwater fish and coral species."
    )

    st.divider()

    st.header("🚀 Features")

    st.write("""
    - Upload images  
    - Upload videos  
    - Automatic object detection  
    - Bounding boxes around detected fish and coral  
    - Download processed video
    """)

# -------------------------------
# IMAGE DETECTION
# -------------------------------
elif page == "📷 Image Detection":

    st.title("Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):

            results = model(image)

            result_image = results[0].plot()

            st.image(result_image, caption="Detection Result", use_column_width=True)

# -------------------------------
# VIDEO DETECTION
# -------------------------------
elif page == "🎥 Video Detection":

    st.title("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        st.video(uploaded_video)

        if st.button("Run Video Detection"):

            st.write("Processing video...")

            results = model.predict(
                source=temp_video.name,
                save=True
            )

            output_folder = results[0].save_dir

            video_files = [f for f in os.listdir(output_folder) if f.endswith(".mp4")]

            if len(video_files) > 0:

                output_video_path = os.path.join(output_folder, video_files[0])

                st.success("Processing complete")

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
