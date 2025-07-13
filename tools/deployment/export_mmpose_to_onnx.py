# python tools/deployment/pytorch2onnx.py \
#   td-hm_mobilenetv2_8xb64-210e_coco-256x192.py \
#   td-hm_mobilenetv2_8xb64-210e_coco-256x192-55a04c35_20221016.pth \
#   --output-file mobilenetv2_pose.onnx \
#   --verify \
#   --opset-version 11

# save as: export_mmpose_to_onnx.py

from mmcv import Config
from mmpose.models import build_posenet
from mmcv.runner import load_checkpoint
import torch

config_file = 'td-hm_mobilenetv2_8xb64-210e_coco-256x192.py'
checkpoint_file = 'td-hm_mobilenetv2_8xb64-210e_coco-256x192-55a04c35_20221016.pth'
onnx_output = 'mobilenetv2_pose.onnx'

cfg = Config.fromfile(config_file)
model = build_posenet(cfg.model)
_ = load_checkpoint(model, checkpoint_file, map_location='cpu')

model.eval()

dummy_input = torch.randn(1, 3, 256, 192)  # B x C x H x W

torch.onnx.export(
    model,
    dummy_input,
    onnx_output,
    input_names=['input'],
    output_names=['heatmap'],
    dynamic_axes={'input': {0: 'batch'}, 'heatmap': {0: 'batch'}},
    opset_version=11
)
