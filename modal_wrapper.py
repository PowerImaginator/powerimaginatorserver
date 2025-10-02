import modal

image = modal.Image.debian_slim().pip_install(
    "diffusers>=0.35.1",
    "fastapi[standard]>=0.117.1",
    "hf-transfer>=0.1.9",
    "numpy>=2.3.3",
    "pillow>=11.3.0",
    "python-multipart>=0.0.20",
    "torch>=2.8.0",
    "torchvision>=0.23.0",
    "transformers>=4.56.2",
).env({
    "HF_HOME": "/hf-home",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}).add_local_python_source("main_module")

app = modal.App(name="powerimaginatorserver", image=image)

@app.function(
    volumes={
        "/hf-home": modal.Volume.from_name("powerimaginatorserver-hf-home", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("powerimaginatorserver-secrets")],
    gpu=["A10", "L40S"],
    timeout=5 * 60,
    max_containers=1,
    scaledown_window=3 * 60,
)
@modal.asgi_app()
def entry():
    from main_module import app
    return app
