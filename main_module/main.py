import io
import os
import math
import json
import base64
import uuid
import threading
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
import numpy as np
import torch
import requests
import websocket
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

app = FastAPI()

API_TOKENS_STR = os.environ.get("POWERIMAGINATOR_API_TOKENS")
if not API_TOKENS_STR:
    raise ValueError("POWERIMAGINATOR_API_TOKENS environment variable not defined")
COMFYUI_URL = os.environ.get("POWERIMAGINATOR_COMFYUI_URL")
if not COMFYUI_URL:
    raise ValueError("POWERIMAGINATOR_COMFYUI_URL environment variable not defined")
WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "..", "workflows", "PowerImaginatorCUDA.json")

DEVICE = "cuda" if torch.cuda.is_available() else "mps"

def load_workflow():
    """Load the ComfyUI workflow JSON"""
    with open(WORKFLOW_PATH, 'r') as f:
        return json.load(f)

def submit_comfyui_prompt(workflow_data):
    """Submit a prompt to ComfyUI API and return the prompt ID"""
    url = f"{COMFYUI_URL}/prompt"
    response = requests.post(url, json={"prompt": workflow_data})
    response.raise_for_status()
    return response.json()["prompt_id"]

def listen_for_comfyui_result(prompt_id, timeout=300):
    """Listen for ComfyUI WebSocket results and return the image data"""
    executed_nodes = {}
    image_data = None
    result_received = threading.Event()

    def on_message(ws, message):
        nonlocal executed_nodes, image_data

        # Handle binary messages (image data from ETN_SendImageWebSocket)
        if isinstance(message, bytes):
            print(f"Received binary message, length: {len(message)}")
            # ETN_SendImageWebSocket sends: 4 bytes (big endian int 1) + 4 bytes (big endian int 2) + PNG data
            if len(message) >= 8:
                # Check the header (should be 1, 2 in big endian)
                header1 = int.from_bytes(message[0:4], byteorder='big')
                header2 = int.from_bytes(message[4:8], byteorder='big')
                if header1 == 1 and header2 == 2:
                    # Extract PNG data
                    image_data = message[8:]
                    print(f"Extracted image data, size: {len(image_data)} bytes")
                    # Binary message comes first, then JSON executed message
                    # We have the image data, signal completion
                    result_received.set()
            return

        # Handle JSON messages
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            print(f"Received WebSocket message: {msg_type}")  # Debug logging

            if msg_type == "execution_start" and data.get("data", {}).get("prompt_id") == prompt_id:
                print(f"Execution started for prompt {prompt_id}")
            elif msg_type == "executing":
                node = data.get("data", {}).get("node")
                if node is None:
                    # Execution finished
                    print("Execution finished")
                    result_received.set()
                else:
                    print(f"Executing node: {node}")
            elif msg_type == "executed":
                node_id = data.get("data", {}).get("node")
                if node_id and data.get("data", {}).get("prompt_id") == prompt_id:
                    print(f"Node {node_id} executed with output: {data['data'].get('output', {})}")
                    executed_nodes[node_id] = data["data"]
        except json.JSONDecodeError:
            print(f"Received non-JSON message: {message[:100]}...")

    def on_error(ws, error):
        print(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code}, {close_msg}")

    def on_open(ws):
        print("WebSocket connected")

    # Connect to WebSocket
    ws_url = COMFYUI_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws?clientId=" + str(uuid.uuid4())
    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)

    # Run WebSocket in a separate thread
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    # Wait for result or timeout
    if result_received.wait(timeout):
        ws.close()
        print(f"Execution completed. Executed nodes: {list(executed_nodes.keys())}, Image data: {image_data is not None}")
        return image_data
    else:
        ws.close()
        raise HTTPException(status_code=504, detail="ComfyUI processing timeout")

def process_with_comfyui(image_data, prompt, negative_prompt=""):
    """Process image through ComfyUI workflow"""
    # Load and modify workflow
    workflow = load_workflow()

    # Convert image to base64
    image_b64 = base64.b64encode(image_data).decode('utf-8')

    # Replace placeholders in workflow
    workflow_str = json.dumps(workflow)
    workflow_str = workflow_str.replace("__POWERIMAGINATOR_IMAGE__", image_b64)
    workflow_str = workflow_str.replace("__POWERIMAGINATOR_PROMPT__", prompt)
    workflow_str = workflow_str.replace("__POWERIMAGINATOR_NEGATIVE_PROMPT__", negative_prompt)
    workflow_data = json.loads(workflow_str)

    # Submit prompt
    prompt_id = submit_comfyui_prompt(workflow_data)
    print(f"Submitted prompt with ID: {prompt_id}")

    # Listen for result - ETN_SendImageWebSocket sends image data directly via WebSocket
    result_image_data = listen_for_comfyui_result(prompt_id)
    if result_image_data is not None:
        print(f"Successfully received image data: {len(result_image_data)} bytes")

        # Debug: Save image to images/test.png
        try:
            with open("images/test.png", "wb") as f:
                f.write(result_image_data)
            print("Debug: Saved image to images/test.png")
        except Exception as e:
            print(f"Debug: Failed to save images/test.png: {e}")

        return result_image_data

    raise HTTPException(status_code=500, detail="No image output received from ETN_SendImageWebSocket")

def validate_api_token(api_token: str):
    tokens_list = API_TOKENS_STR.split(",")
    for token in tokens_list:
        token = token.strip()
        if not token:
            continue
        if token == api_token:
            return True
    raise HTTPException(status_code=401, detail="Invalid API token")

depth_image_processor = None
depth_model = None

@app.get("/config")
def get_config(api_token: str):
    validate_api_token(api_token)

    return {
        "width": 1024,
        "height": 1024,
        "fov_y": 75.0 * (math.pi / 180.0),
        "allow_brush_strength": False,
    }

@app.post("/inpaint")
def inpaint(
    api_token: str,
    init_image_file: UploadFile,
    mask_image_file: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    seed: int = Form(0),
    num_inference_steps: int = Form(30),
    strength: float = Form(1.0),
):
    validate_api_token(api_token)

    # Read the image and mask files
    init_image_data = init_image_file.file.read()
    mask_image_data = mask_image_file.file.read()

    # Process init image: replace non-zero mask pixels with pure red
    init_image = Image.open(io.BytesIO(init_image_data)).convert("RGB")
    mask_image = Image.open(io.BytesIO(mask_image_data)).convert("L")  # Convert to grayscale

    init_array = np.array(init_image)
    mask_array = np.array(mask_image)

    # Create red color array (255, 0, 0) - pure red
    red_color = np.array([255, 0, 0], dtype=np.uint8)

    # Replace pixels where mask is non-zero with red
    mask_nonzero = mask_array > 0
    init_array[mask_nonzero] = red_color

    # Convert back to PIL Image and then to bytes
    processed_image = Image.fromarray(init_array)
    buf = io.BytesIO()
    processed_image.save(buf, format="PNG")
    image_data = buf.getvalue()

    # Process with ComfyUI (ignoring seed, steps, strength as requested)
    result_image_data = process_with_comfyui(image_data, prompt, negative_prompt)

    # Ensure the result is in the same format as original PIL output
    result_image = Image.open(io.BytesIO(result_image_data))

    # Return the processed image (exact same format as original)
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")

@app.post("/depth")
def depth(api_token: str, image_file: UploadFile):
    validate_api_token(api_token)

    global depth_image_processor
    global depth_model

    DEPTH_CHECKPOINT = "apple/DepthPro-hf"

    if depth_image_processor is None:
        depth_image_processor = AutoImageProcessor.from_pretrained(DEPTH_CHECKPOINT)

    if depth_model is None:
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_CHECKPOINT).to(DEVICE)

    image = Image.open(image_file.file).convert("RGB")

    pixel_values = depth_image_processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(pixel_values)
    post_processed_output = depth_image_processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    predicted_depth = post_processed_output[0]["predicted_depth"]

    depth_array = predicted_depth.cpu().numpy().flatten().astype(np.float32)
    binary_data = depth_array.tobytes()
    return Response(content=binary_data, media_type="application/octet-stream")
