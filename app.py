import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

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

```
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    image.save(tmp.name)

    results = model(tmp.name)

    result_img = results[0].plot()

    with col2:
        st.subheader("Detection Result")
        st.image(result_img, use_container_width=True)

    boxes = results[0].boxes

    st.divider()

    if len(boxes) > 0:
        st.success(f"Detected {len(boxes)} object(s)")
    else:
        st.warning("No object detected")
```

# =========================

# VIDEO DETECTION

# =========================

elif source == "Video":

```
uploaded_video = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:

    st.video(uploaded_video)

    st.info(
        "Video uploaded successfully. "
        "Video inference can be added later if needed."
    )
```

# =========================

# WEBCAM DETECTION

# =========================

elif source == "Webcam":

```
st.warning(
    "Webcam mode is not supported on Streamlit Cloud. Please use Image or Video mode."
)
```
