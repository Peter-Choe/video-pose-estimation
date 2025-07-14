from app.core.config import settings
import cv2
import numpy as np
import tritonclient.http as httpclient
import random


triton_url = settings.TRITON_URL.replace("http://", "").replace("https://", "")
triton_client = httpclient.InferenceServerClient(url=triton_url)

# Constants
MODEL_NAME = "mobilenetv2_pose"
INPUT_W, INPUT_H = 192, 256
HEATMAP_W, HEATMAP_H = 48, 64
STRIDE = INPUT_W // HEATMAP_W


def infer_pose(frame: np.ndarray, orig_w: int, orig_h: int):
    # Resize frame to model input size (e.g., 192x256)
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))

    # Normalize pixel values to [0, 1] and convert to CHW format with batch dimension
    img = resized.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]  # Shape: [1, 3, H, W]

    # Prepare input for Triton inference
    inputs = [httpclient.InferInput("input", img.shape, "FP32")]
    inputs[0].set_data_from_numpy(img)

    # Request the output named "heatmap"
    outputs = [httpclient.InferRequestedOutput("heatmap")]

    # Perform inference using Triton
    response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

    # Get the heatmap output and remove batch dimension → shape: [17, 64, 48]
    heatmap = response.as_numpy("heatmap")[0]

    # Optional debug logging (random sampling to avoid too much output)
    if settings.DEBUG and random.random() < 0.03:
        output_info = response.get_output("heatmap")
        print("[Triton Request] Model:", MODEL_NAME)
        print("[Triton Request] Input shape:", img.shape)
        print("[Triton Request] Input dtype:", inputs[0]._datatype)
        print("[Triton Response] Output shape:", heatmap.shape)
        print("[Triton Response] Output dtype:", output_info['datatype'])
        print("[Triton Response] Output name:", output_info['name'])

    # Post-processing: convert heatmaps to keypoint coordinates
    keypoints = []
    for i in range(heatmap.shape[0]):  # Iterate over 17 keypoints
        # Find (y, x) coordinate of the peak in the heatmap (most confident location)
        y, x = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)

        # Scale heatmap coordinates back to model input resolution using stride
        kp_x, kp_y = int(x * STRIDE), int(y * STRIDE)

        # Scale from model input size to original image size
        scale_x, scale_y = orig_w / INPUT_W, orig_h / INPUT_H
        real_x, real_y = int(kp_x * scale_x), int(kp_y * scale_y)

        # Append final keypoint in original image coordinates
        keypoints.append((real_x, real_y))

    return keypoints
