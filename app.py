

import streamlit as st
import subprocess
import cv2
import os
from PIL import Image
import tempfile

import sys
import time
import os

import cv2
import numpy as np

import ailia

sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))

from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402ss
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

IMAGE_PATH = 'test.jpeg'
SAVE_IMAGE_PATH = 'result.png'
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Single image crowd counting.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
args = update_parser(parser)

def estimate_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
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
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing time {end - start} ms")
        else:
            preds_ailia = net.predict(input_data)

        # estimated crowd count
        et_count = int(np.sum(preds_ailia))
        
        # Print the count
        print(et_count)

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
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)
    logger.info('Script finished successfully.')


# def estimate_from_video():
#     # net initialize
#     net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

#     capture = webcamera_utils.get_capture(args.video)

#     # create video writer if savepath is specified as video format
#     if args.savepath != SAVE_IMAGE_PATH:
#         f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         save_h, save_w = webcamera_utils.calc_adjust_fsize(
#             f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
#         )
#         # save_w * 2: we stack source frame and estimated heatmap
#         writer = webcamera_utils.get_writer(args.savepath, save_h, save_w * 2)
#     else:
#         writer = None

#     frame_shown = False
#     while(True):
#         ret, frame = capture.read()
#         if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
#             break
#         if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
#             break

#         input_image, input_data = webcamera_utils.preprocess_frame(
#             frame,
#             IMAGE_HEIGHT,
#             IMAGE_WIDTH,
#             data_rgb=False,
#             normalize_type='None',
#         )

#         # inference
#         preds_ailia = net.predict(input_data)

#         # estimated crowd count
#         et_count = int(np.sum(preds_ailia))

#         # density map
#         density_map = (255 * preds_ailia / np.max(preds_ailia))[0][0]
#         density_map = cv2.resize(
#             density_map,
#             (input_image.shape[1], input_image.shape[0]),
#         )
#         heatmap = cv2.applyColorMap(
#             density_map.astype(np.uint8), cv2.COLORMAP_JET
#         )
#         cv2.putText(
#             heatmap,
#             f'Est Count: {et_count}',
#             (40, 440),  # position
#             cv2.FONT_HERSHEY_SIMPLEX,  # font
#             0.8,  # fontscale
#             (255, 255, 255),  # color
#             2,  # thickness
#         )
#         res_img = np.hstack((input_image, heatmap))
#         cv2.imshow('frame', res_img)
#         frame_shown = True

#         # save results
#         if writer is not None:
#             writer.write(res_img)

#     capture.release()
#     cv2.destroyAllWindows()
#     if writer is not None:
#         writer.release()
#     logger.info('Script finished successfully.')

def estimate_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        # save_w * 2: we stack source frame and estimated heatmap
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            data_rgb=False,
            normalize_type='None',
        )

        # inference
        preds_ailia = net.predict(input_data)

        # estimated crowd count
        et_count = int(np.sum(preds_ailia))

        # density map
        density_map = (255 * preds_ailia / np.max(preds_ailia))[0][0]
        density_map = cv2.resize(
            density_map,
            (input_image.shape[1], input_image.shape[0]),
        )
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
        res_img = np.hstack((input_image, heatmap))

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

# model files check and download
check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

if args.video is not None:
    # video mode
    estimate_from_video()
else:
    # image mode
    estimate_from_image()














st.title("Crowd Counting from Image")

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
