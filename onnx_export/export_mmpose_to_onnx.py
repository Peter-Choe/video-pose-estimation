from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.apis import init_model
import torch
import os

# 파일 경로
config_file = 'mmpose_models/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py'
checkpoint_file = 'mmpose_models/td-hm_mobilenetv2_8xb64-210e_coco-256x192-55a04c35_20221016.pth'
onnx_output = 'onnx_output/mobilenetv2_pose/1.01/model.onnx'

# 모델 저장 디렉토리 생성
os.makedirs(os.path.dirname(onnx_output), exist_ok=True)

# 모델 로딩 및 ONNX 변환
cfg = Config.fromfile(config_file)

# model = build_posenet(cfg.model)
model = init_model(cfg, checkpoint_file, device='cpu')

_ = load_checkpoint(model, checkpoint_file, map_location='cpu')

model.eval()

dummy_input = torch.randn(1, 3, 256, 192)

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
