import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Load the trained YOLOv8 model
trained_model_path = 'best.pt'

# Ensure the model file exists, or provide a mechanism to download it
if not os.path.exists(trained_model_path):
    st.error("Model file not found. Ensure that 'best.pt' is uploaded.")
    st.stop()

model = YOLO(trained_model_path)

# Title of the Streamlit app
st.title("Person Detection with YOLOv8")

# Upload a file (image or video)
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

def process_image(image):
    image_np = np.array(image)
    results = model(image_np)
    annotated_frame = results[0].plot()
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return annotated_frame_rgb

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        image = Image.open(uploaded_file)
        with st.spinner('Processing image...'):
            annotated_frame_rgb = process_image(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(annotated_frame_rgb, caption='Annotated Image', use_column_width=True)

    elif file_type == 'video':
        with st.spinner('Processing video...'):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            output_path = process_video(video_path)
            if output_path:
                st.video(output_path)
            os.remove(video_path)

    st.success("Processing completed.")
