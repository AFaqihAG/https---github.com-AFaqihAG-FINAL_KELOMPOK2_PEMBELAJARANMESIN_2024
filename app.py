import streamlit as st
import subprocess
import cv2
import os
from PIL import Image
import tempfile
import numpy as np
import sys
import time

import ailia

sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))

from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "crowdcount.opt.onnx"
MODEL_PATH = "crowdcount.opt.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/crowd_count/"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Initialize Streamlit app
st.title("Crowd Counting from Image")

# File upload widget for image
uploaded_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def estimate_from_image(image_path, save_path):
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=0)  # env_id=0 for CPU

    # prepare input data
    org_img = load_image(
        image_path,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
    )
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    input_data = load_image(
        image_path,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        normalize_type='None',
        gen_input_ailia=True,
    )

    # inference
    preds_ailia = net.predict(input_data)

    # estimated crowd count
    et_count = int(np.sum(preds_ailia))

    # density map
    density_map = (255 * preds_ailia / np.max(preds_ailia))[0][0]
    density_map = cv2.resize(density_map, (IMAGE_WIDTH, IMAGE_HEIGHT))
    heatmap = cv2.applyColorMap(
        density_map.astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.putText(
        heatmap,
        f'Est Count: {et_count}',
        (40, 440),  # position
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        0.8,  # fontscale
        (255, 255, 255),  # color
        2,  # thickness
    )

    res_img = np.hstack((org_img, heatmap))
    cv2.imwrite(save_path, res_img)
    return et_count, save_path

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

        # Button to run the crowd counting logic
        if st.button("Run Crowd Counting on Image"):
            try:
                estimated_count, processed_image_path = estimate_from_image(input_image_path, output_image_path)

                # Display the processed image
                processed_image = Image.open(processed_image_path)
                st.image(processed_image, caption="Processed Image", use_column_width=True)

                # Display the estimated count
                st.write(f"Number of people counted: {estimated_count}")

            except Exception as e:
                st.error(f"Error occurred during processing: {e}")

# Download the model files if not already present
check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
