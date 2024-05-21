import streamlit as st
import cv2
import os
from PIL import Image
import tempfile
import numpy as np

st.title("Crowd Counting from Image")

# File upload widget for image
uploaded_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to count people (Placeholder for the actual model)
def count_people(image_path):
    # Here you should implement or call your actual crowd counting model
    # For demonstration, we'll assume a dummy model that returns a fixed number
    # This is where you would load your model, preprocess the image, and get the count
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image.")
    processed_image = image.copy()  # Placeholder for actual image processing
    estimated_count = 100  # Placeholder for actual counting logic
    output_image_path = image_path.replace(".jpg", "_output.png")
    cv2.imwrite(output_image_path, processed_image)
    return estimated_count, output_image_path

# Process the image
if uploaded_image_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
        input_image_path = temp_image_file.name

        # Save uploaded image to input path
        with open(input_image_path, "wb") as f:
            f.write(uploaded_image_file.getvalue())

        # Display the uploaded image
        uploaded_image = cv2.imread(input_image_path)
        uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Button to run the crowd counting logic
        if st.button("Run Crowd Counting on Image"):
            try:
                estimated_count, output_image_path = count_people(input_image_path)

                # Display the output image
                processed_image = cv2.imread(output_image_path)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                st.image(processed_image, caption="Processed Image", use_column_width=True)

                # Display the estimated count
                st.write(f"Number of people counted: {estimated_count}")

            except Exception as e:
                st.error(f"Error occurred during processing: {e}")
