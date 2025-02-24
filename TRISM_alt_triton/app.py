from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
from trism import TritonModel
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Triton Inference Server URL
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "super_resolution_onnx"

# Connect to Triton
model = TritonModel(
    model="super_resolution_onnx",  # Model name in Triton Server
    url="localhost:8001",    # Triton gRPC URL
    version=1,  
    grpc=True  # Use gRPC (set False for HTTP)
)



@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img).astype(np.float32)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=(0, 1))  # Add batch and channel dims
    img_array /= 255.0  # Normalize

    # Run inference
    outputs = model.run(data=[img_array])
    inference_output = outputs["output"] 

    # Convert output tensor to image
    output_array = inference_output[0, 0]  # Remove batch & channel dims (shape: 672x672)
    output_array = (output_array * 255.0).clip(0, 255).astype(np.uint8)  # Convert to uint8

    # Convert NumPy array back to PIL image
    output_image = Image.fromarray(output_array)
    # Convert to BytesIO for streaming response
    img_io = io.BytesIO()
    output_image.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")
