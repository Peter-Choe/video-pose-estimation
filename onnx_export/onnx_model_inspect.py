import onnx

model = onnx.load("models/mobilenetv2_pose/1/model.onnx")
input_tensor = model.graph.input[0]

# Get input tensor shape
shape = []
for dim in input_tensor.type.tensor_type.shape.dim:
    dim_val = dim.dim_value if dim.HasField("dim_value") else "?"
    shape.append(dim_val)

print("Input shape:", shape)
