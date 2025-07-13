from tritonclient.http import InferenceServerClient
client = InferenceServerClient(url="localhost:8000")
print(client.is_server_live())