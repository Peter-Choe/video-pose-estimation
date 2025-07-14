from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.apis import init_model
import torch
import os

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.apis import init_model
import torch
import os

# === File paths for config, checkpoint, and ONNX output ===
config_file = 'mmpose_models/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py'
checkpoint_file = 'mmpose_models/td-hm_mobilenetv2_8xb64-210e_coco-256x192-55a04c35_20221016.pth'
onnx_output = 'onnx_output/mobilenetv2_pose/1.01/model.onnx'

# === Create output directory if it doesn't exist ===
os.makedirs(os.path.dirname(onnx_output), exist_ok=True)

# === Load model config from .py file ===
cfg = Config.fromfile(config_file)

# === Initialize pose estimation model with checkpoint ===
model = init_model(cfg, checkpoint_file, device='cpu')

# === Load weights (optional: already done by init_model) ===
_ = load_checkpoint(model, checkpoint_file, map_location='cpu')

# === Set model to evaluation mode ===
model.eval()

# === Dummy input for ONNX tracing (batch size = 1, 3x256x192 image) ===
dummy_input = torch.randn(1, 3, 256, 192)

# === Export model to ONNX format ===
#     - Input/output named for Triton compatibility
#     - Batch size is dynamic (0th dim)
torch.onnx.export(
    model,
    dummy_input,
    onnx_output,
    input_names=['input'],
    output_names=['heatmap'],
    dynamic_axes={'input': {0: 'batch'}, 'heatmap': {0: 'batch'}},
    opset_version=11
)

#
#docker run --rm -v $PWD/../triton_models:/output
