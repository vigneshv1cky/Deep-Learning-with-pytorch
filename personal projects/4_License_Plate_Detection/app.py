import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import uuid

st.set_page_config(page_title="License Plate Detection - YOLOv8", layout="centered")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

# Header
st.title("🔍 License Plate Detection using YOLOv8")
st.markdown("""
This web application allows you to detect **license plates** in uploaded images using the YOLOv8 object detection model.

- Trained on a Roboflow license plate dataset
- Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Deployed in **Google Colab + Streamlit + ngrok**

👉 Upload one or more images and see the detection results instantly!
""")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# File uploader
uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📷 Image: `{uploaded_file.name}`")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("Detecting license plates..."):
            # Run detection
            results = model.predict(image, conf=conf_threshold)

            # Save annotated image
            output_filename = f"output_{uuid.uuid4().hex}.jpg"
            results[0].save(filename=output_filename)

            # Show annotated result
            st.image(output_filename, caption="Detected Output", use_container_width=True)

            # Show detection info
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                st.success(f"✅ {len(detections)} license plate(s) detected")
                for i, box in enumerate(detections.data):
                  cls = int(box[5].item())  # YOLO format: [x1, y1, x2, y2, conf, class]
                  conf = box[4].item()
                  class_name = model.names.get(cls, "Unknown")
                  st.write(f"🔹 Detection {i+1}: Class = `{class_name}`, Confidence = `{conf:.2f}`")
            else:
                st.warning("⚠️ No license plates detected.")

            # Download button
            with open(output_filename, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Output Image",
                    data=file,
                    file_name=output_filename,
                    mime="image/jpeg"
                )
