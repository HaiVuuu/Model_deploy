from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Triton Inference Server URL
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "super_resolution_onnx"

# Connect to Triton
client = InferenceServerClient(url=TRITON_SERVER_URL)



@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img).astype(np.float32)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=(0, 1))  # Add batch and channel dims
    img_array /= 255.0  # Normalize

    # Prepare Triton input
    inputs = InferInput("input", img_array.shape, datatype="FP32")
    inputs.set_data_from_numpy(img_array)
    outputs = InferRequestedOutput("output", binary_data=True)

    # Send request to Triton
    results = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])
    
    # Get output and convert to image
    inference_output = results.as_numpy("output")[0, 0]  # Remove batch & channel dims
    output_img = Image.fromarray((inference_output * 255).astype(np.uint8))  # Convert to grayscale image

    # Convert to BytesIO for streaming response
    img_io = io.BytesIO()
    output_img.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")
