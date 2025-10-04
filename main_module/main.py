import io
import os
import math
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
import numpy as np
import torch
import diffusers
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

app = FastAPI()

DEVICE = "cuda"

def validate_api_token(api_token: str):
    tokens_str = os.environ.get("POWERIMAGINATOR_API_TOKENS")
    if not tokens_str:
        raise HTTPException(status_code=500, detail="API tokens not set")
    tokens_list = tokens_str.split(",")
    for token in tokens_list:
        token = token.strip()
        if not token:
            continue
        if token == api_token:
            return True
    raise HTTPException(status_code=401, detail="Invalid API token")

inpaint_pipe = None
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
async def inpaint(
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

    global inpaint_pipe

    if inpaint_pipe is None:
        inpaint_pipe = diffusers.QwenImageInpaintPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch.bfloat16,
            # safety_checker=diffusers.pipelines.stable_diffusion.safety_checker.StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            # requires_safety_checker=True,
        )
        inpaint_pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-4steps-V2.0.safetensors"
        )
        inpaint_pipe.enable_sequential_cpu_offload()
        # inpaint_pipe.enable_xformers_memory_efficient_attention()

    init_image = Image.open(init_image_file.file).convert("RGB")
    mask_image = Image.open(mask_image_file.file).convert("RGB")

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    inpainted_image = inpaint_pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
        num_inference_steps=num_inference_steps,
        strength=strength,
    ).images[0]

    buf = io.BytesIO()
    inpainted_image.save(buf, format="PNG")
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
