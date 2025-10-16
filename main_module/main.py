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

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

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
):
    validate_api_token(api_token)

    global inpaint_pipe

    if inpaint_pipe is None:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        inpaint_pipe = diffusers.QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16,
            scheduler=scheduler,
            # safety_checker=diffusers.pipelines.stable_diffusion.safety_checker.StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            # requires_safety_checker=True,
        )
        print("Model loaded successfully")

        inpaint_pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
        )
        print("LoRA weights loaded successfully")

        inpaint_pipe.fuse_lora()
        print("LoRA fusion applied successfully")

        inpaint_pipe = inpaint_pipe.to(DEVICE)
        print("Model moved to GPU")

    init_image = Image.open(init_image_file.file).convert("RGB")
    mask_image = Image.open(mask_image_file.file).convert("RGB")

    # Convert images to numpy arrays for pixel manipulation
    init_array = np.array(init_image)
    mask_array = np.array(mask_image)
    # Find mask pixels where any R, G, or B component is greater than 0
    white_mask = (mask_array > 0).any(axis=2)
    # Set corresponding pixels in init_image to green (RGB: 0, 255, 0)
    init_array[white_mask] = [0, 255, 0]
    # Convert back to PIL Image
    init_image = Image.fromarray(init_array)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Run inference (same as HF Space - 8 steps!)
    inpainted_image = inpaint_pipe(
        init_image,
        prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=8,  # Same as HF Space
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
