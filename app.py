import cv2
import streamlit as st
from yolov8 import YOLOv8
import numpy as np
from PIL import Image
import wget

CURRENT_THEME ="dark"

# Load the YOLOv8 model
model_path = "best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.3)

def detect_objects(image):
    # Detect objects using YOLOv8
    boxes, scores, class_ids = yolov8_detector(image)
    detected_image = yolov8_detector.draw_detections(image)
    return detected_image

def count_objects(image):
    boxes, scores, class_ids = yolov8_detector(image)
    detected_image = yolov8_detector.draw_det(image)
    return detected_image

# Streamlit app
def main():
    # Set app title and description
    st.title("Retail Object Detection")
    st.markdown("""
    Welcome to the Retail Object Detection app!
    This app allows you to upload retail images and performs object detection using the YOLOv8 model.
    It detects various objects in the image and displays the result with bounding boxes.
    """)

    # Select the mode (image or video)
    mode = st.sidebar.radio("Select Mode", ("Image", "Webcam" ,"Further information"))

    if mode == "Image":
        st.header("Image Object Detection")

        # File uploader for image input
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the uploaded image file
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

            # Display the uploaded image and the detected image side by side
            col1, col2,col3 = st.columns(3)
            with col1:
                st.image(image, channels="BGR", caption="Uploaded Image")
            with col2:
                # Perform object detection on the uploaded image
                detected_image = detect_objects(image)
                # Convert the detected image to RGB format
                detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                # Create a PIL image
                pil_image = Image.fromarray(detected_image_rgb)
                st.image(pil_image, caption="Detected Objects")
            with col3:
                detected_image = count_objects(image)
                # Convert the detected image to RGB format
                detected_image_rgb = cv2.cvtColor(detected_image[1], cv2.COLOR_BGR2RGB)
                count = detected_image[0]
                # Create a PIL image
                pil_image = Image.fromarray(detected_image_rgb)
                st.image(pil_image, caption="Counted Objects")
            st.write("Total number of objects detected: ", count)

    elif mode == "Webcam":
        st.header("Video Object Detection")
        st.write("Coming soon...")
    
    elif mode =="Further information":
        st.header("Further information")
        st.write("Improve the model by adding more data and training for longer. The model is currently trained on a small dataset of retail images. The model can be improved by adding more images and training for longer. The model can also be improved by adding more data augmentation techniques such as random cropping, random rotation, and random color jittering. The model can also be improved by using a larger model such as YOLOv8 or EfficientDet.")
        st.write("Train it to identify objects by brands as well.")
        



    # Additional pages
    st.sidebar.title("Experiment Information")
    st.sidebar.write("This experiment utilizes the YOLOv8 model to perform object detection on retail images and videos. The model has been trained on a large dataset of retail images and can detect various objects commonly found in retail settings.")
    st.sidebar.write("Object detection is an important task in retail as it enables automated analysis of visual data, leading to applications such as inventory management, customer behavior analysis, and loss prevention.")

if __name__ == "__main__":
    main()
