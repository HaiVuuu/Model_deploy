import numpy as np
import tritonclient
from tritonclient.http import InferenceServerClient
from tritonclient.http import InferInput, InferRequestedOutput
from PIL import Image

# Load and preprocess your image (resize and normalize as required)
img = Image.open('F:/AI/Kì 6/OJT/6_3_2/6_3_2/478 Clark Creek Lane - Cary, NC/Exteriors 17/handsome_cat.jpg').convert('L')  # Convert to grayscale
img = img.resize((224, 224))  # Resize to model's expected input size
img_array = np.array(img).astype(np.float32)  # Convert to numpy array

# Ensure the array has a channel dimension (1 channel for grayscale)
img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 224, 224)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension, shape becomes (1, 1, 224, 224)

# Normalize if required (e.g., if model expects values between 0 and 1)
img_array /= 255.0  # Example normalization

client = InferenceServerClient(url="localhost:8000")

# Prepare the input
inputs = InferInput("input", img_array.shape, datatype="FP32")
inputs.set_data_from_numpy(img_array)

# Prepare the output
outputs = InferRequestedOutput("output", binary_data=True)

# Query the server
results = client.infer(model_name="super_resolution_onnx", inputs=[inputs], outputs=[outputs])

# Process the output
inference_output = results.as_numpy('output')
print(inference_output.shape)

