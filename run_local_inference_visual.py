import cv2
import numpy as np
import os
import random
from fastapi_pipeline.app.core.config import settings
import tritonclient.http as httpclient


#프로젝트 루트 가상환경에서 ENV=local python run_local_inference_visual.py 실행

# Constants
MODEL_NAME = "mobilenetv2_pose"
INPUT_W, INPUT_H = 192, 256
HEATMAP_W, HEATMAP_H = 48, 64
STRIDE = INPUT_W // HEATMAP_W

# Triton Client
triton_url = settings.TRITON_URL.replace("http://", "").replace("https://", "")
triton_client = httpclient.InferenceServerClient(url=triton_url)

def should_visualize():
    return os.environ.get("DISPLAY") is not None

def infer_pose(frame: np.ndarray, orig_w: int, orig_h: int, visualize: bool = True):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = resized.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]

    inputs = [httpclient.InferInput("input", img.shape, "FP32")]
    inputs[0].set_data_from_numpy(img)
    outputs = [httpclient.InferRequestedOutput("heatmap")]

    response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    heatmap = response.as_numpy("heatmap")[0]

    keypoints = []
    for i in range(heatmap.shape[0]):
        y, x = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)
        kp_x, kp_y = int(x * STRIDE), int(y * STRIDE)
        scale_x, scale_y = orig_w / INPUT_W, orig_h / INPUT_H
        real_x, real_y = int(kp_x * scale_x), int(kp_y * scale_y)
        keypoints.append((real_x, real_y))

        if visualize and should_visualize():
            cv2.circle(frame, (real_x, real_y), 3, (0, 255, 0), -1)

    if visualize and should_visualize():
        vis_display = cv2.resize(frame, (0, 0), fx=0.7, fy=0.5)
        cv2.imshow("Pose Inference", vis_display)
        cv2.waitKey(1)

    return keypoints

def main(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video file:", video_path)
        return

    print("Video opened, starting inference loop...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        orig_h, orig_w = frame.shape[:2]
        keypoints = infer_pose(frame, orig_w, orig_h, visualize=True)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Frame {frame_count}, Keypoints: {keypoints}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done and windows closed.")


if __name__ == "__main__":
    main("resources/video/Djokovic_forehand_slow_motion.mp4")  