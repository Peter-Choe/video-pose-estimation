import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto

input_tensor = helper.make_tensor_value_info("INPUT0", TensorProto.FLOAT, [None, 4])
output_tensor = helper.make_tensor_value_info("OUTPUT0", TensorProto.FLOAT, [None, 4])

node_def = helper.make_node(
    "Identity",
    inputs=["INPUT0"],
    outputs=["OUTPUT0"]
)

graph_def = helper.make_graph(
    [node_def],
    "identity_model",
    [input_tensor],
    [output_tensor]
)

model_def = helper.make_model(graph_def, producer_name="onnx-example")
onnx.save(model_def, "models/identity_model/1/model.onnx")
