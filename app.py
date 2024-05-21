import streamlit as st
import subprocess
import cv2
import os
from PIL import Image
import tempfile

st.title("Crowd Counting from Image or Video")

# File upload widget for image
uploaded_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the image
if uploaded_image_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
        input_image_path = temp_image_file.name
        output_image_path = input_image_path.replace(".jpg", "_output.png")

        # Save uploaded image to input path
        with open(input_image_path, "wb") as f:
            f.write(uploaded_image_file.getvalue())

        # Display the uploaded image
        uploaded_image = cv2.imread(input_image_path)
        uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Button to run the script
        if st.button("Run Crowd Counting on Image"):
            # Run the script with specified input and output paths
            cmd = f"python3 crowdcount-cascaded-mtl.py --input {input_image_path} --savepath {output_image_path}"
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()

            # Check if the process ran successfully
            if process.returncode == 0:
                # Display the output image
                processed_image = Image.open(output_image_path)
                st.image(processed_image, caption="Processed Image", use_column_width=True)

                # Extract count from the output
                output_text = output.decode("utf-8")
                count_line = next((line for line in output_text.splitlines() if "Est Count:" in line), None)
                if count_line:
                    count = int(count_line.split(":")[1].strip())
                    st.write(f"Number of people counted: {count}")
            else:
                st.error("Error occurred during processing.")
                st.error(error.decode("utf-8"))