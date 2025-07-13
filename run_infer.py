from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

client = InferenceServerClient(url="localhost:8000")

input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
inputs = [InferInput("INPUT0", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

outputs = [InferRequestedOutput("OUTPUT0")]

response = client.infer("identity_model", inputs=inputs, outputs=outputs)
print("Output:", response.as_numpy("OUTPUT0"))
