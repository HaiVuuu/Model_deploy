import numpy as np
from trism import TritonModel
from PIL import Image


# Load and preprocess the image (convert to grayscale, resize, and normalize)
img = Image.open('_NFP4129.png').convert('L')  # Convert to grayscale
img = img.resize((224, 224))  # Resize to match model input
img_array = np.array(img).astype(np.float32)  # Convert to numpy array

# Ensure the array has a channel dimension (1 channel for grayscale)
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension, shape: (1, 1, 224, 224)

# Normalize if required (e.g., scale pixel values to [0, 1])
img_array /= 255.0  # Example normalization

# Connect to Triton Model using TRISM
model = TritonModel(
    model="super_resolution_onnx",  # Model name in Triton Server
    url="localhost:8001",    # Triton gRPC URL
    version=1,  
    grpc=True  # Use gRPC (set False for HTTP)
)

# View metadata.
print("Metadata information:")
for inp in model.inputs:
  print(f"Name: {inp.name}, Shape: {inp.shape}, Datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"Name: {out.name}, Shape: {out.shape}, Datatype: {out.dtype}\n")
# Run inference
outputs = model.run(data=[img_array])
print("----------------------------------------------------------\n")
# Process the output
inference_output = outputs["output"]  # Access by output tensor name
  # Extract the output tensor
print("Inference output Shape:\n", inference_output.shape)

#Output image on local

# Convert output tensor to an image
output_array = inference_output[0, 0]  # Remove batch & channel dimensions (shape: 672x672)
output_array = (output_array * 255.0).clip(0, 255).astype(np.uint8)  # Convert to uint8 (0-255)

# Convert NumPy array to PIL Image
output_image = Image.fromarray(output_array)

# Show the output image
output_image.show()