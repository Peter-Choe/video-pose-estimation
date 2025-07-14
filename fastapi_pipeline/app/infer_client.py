from app.core.config import settings
import cv2
import numpy as np
import tritonclient.http as httpclient
import random
import os
import signal
import sys

# Setup signal handler for graceful window closing
def handle_exit(*args):
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

triton_url = settings.TRITON_URL.replace("http://", "").replace("https://", "")
triton_client = httpclient.InferenceServerClient(url=triton_url)

# Constants
MODEL_NAME = "mobilenetv2_pose"
INPUT_W, INPUT_H = 192, 256
HEATMAP_W, HEATMAP_H = 48, 64
STRIDE = INPUT_W // HEATMAP_W


def infer_pose(frame: np.ndarray, orig_w: int, orig_h: int):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]

    inputs = [httpclient.InferInput("input", img.shape, "FP32")]
    inputs[0].set_data_from_numpy(img)
    outputs = [httpclient.InferRequestedOutput("heatmap")]

    response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    heatmap = response.as_numpy("heatmap")[0]

    # Conditional debug logging
    if settings.DEBUG and random.random() < 0.03:
        output_info = response.get_output("heatmap")
        print("[Triton Request] Model:", MODEL_NAME)
        print("[Triton Request] Input shape:", img.shape)
        print("[Triton Request] Input dtype:", inputs[0]._datatype)
        print("[Triton Response] Output shape:", heatmap.shape)
        print("[Triton Response] Output dtype:", output_info['datatype'])
        print("[Triton Response] Output name:", output_info['name'])

    # Keypoint extraction
    keypoints = []
    for i in range(heatmap.shape[0]):
        y, x = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)
        kp_x, kp_y = int(x * STRIDE), int(y * STRIDE)
        scale_x, scale_y = orig_w / INPUT_W, orig_h / INPUT_H
        real_x, real_y = int(kp_x * scale_x), int(kp_y * scale_y)
        keypoints.append((real_x, real_y))

    return keypoints