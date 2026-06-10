import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# HEADER
# =========================
st.title("🪖 Helmet Detection System")
st.markdown(
    """
    Detect motorcycle riders wearing helmets and riders without helmets using YOLOv8.
    """
)

st.divider()

# =========================
# SIDEBAR
# =========================
source = st.sidebar.radio(
    "Choose Detection Mode",
    ["Image", "Video", "Webcam"]
)

# =========================
# IMAGE DETECTION
# =========================
if source == "Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            image.save(tmp.name)

            results = model(tmp.name)

        result_img = results[0].plot()

        with col2:
            st.subheader("Detection Result")
            st.image(result_img, use_container_width=True)

# =========================
# VIDEO DETECTION
# =========================
elif source == "Video":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.video(uploaded_video)

        st.info(
            "Video uploaded successfully. "
            "For Streamlit Cloud deployment, "
            "video processing may take longer."
        )

# =========================
# WEBCAM DETECTION
# =========================
elif source == "Webcam":

    st.warning(
        "Webcam mode works best when running locally."
    )

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Cannot access webcam")
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        FRAME_WINDOW.image(
            annotated_frame,
            channels="BGR"
        )

    camera.release()
