import modal

# ============================================================
# ResearchPilot — Modal Deployment
# ============================================================

app = modal.App("researchpilot")

# Persistent storage:
# 1. Chroma vector database
# 2. Hugging Face model cache
data_volume = modal.Volume.from_name(
    "researchpilot-data",
    create_if_missing=True,
)

model_cache = modal.Volume.from_name(
    "researchpilot-hf-cache",
    create_if_missing=True,
)

# Build the runtime from the same dependencies as the repo.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("fastapi[standard]")
    .add_local_file("app.py", "/root/app.py")
    .add_local_file("retrieval.py", "/root/retrieval.py")
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("tools.py", "/root/tools.py")
)


# ============================================================
# ONE-TIME MODEL CACHE WARMUP
#
# Run once with:
#
# modal run modal_app.py::cache_models
#
# This downloads the Qwen models into a persistent Modal Volume
# so later cold starts do not have to download all model files
# from Hugging Face again.
# ============================================================

@app.function(
    image=image,
    volumes={
        "/model-cache": model_cache,
    },
    secrets=[
        modal.Secret.from_name("researchpilot-secrets"),
    ],
    env={
        "HF_HOME": "/model-cache",
    },
    timeout=3600,
    memory=32768,
)
def cache_models():

    from huggingface_hub import snapshot_download

    models = [
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Reranker-4B",
    ]

    for model_name in models:

        print(
            f"Downloading {model_name}..."
        )

        snapshot_download(
            repo_id=model_name,
        )

        print(
            f"Cached {model_name}"
        )

    model_cache.commit()

    print(
        "✅ Model cache committed."
    )


# ============================================================
# PUBLIC GRADIO APP
#
# Uses the same two-GPU layout that was validated during
# ResearchPilot development:
#
# cuda:0 -> Qwen3-Embedding-8B
# cuda:1 -> Qwen3-Reranker-4B
# ============================================================

@app.function(
    image=image,

    # Preserve the tested dual-T4 architecture.
    gpu="T4:2",

    cpu=4,
    memory=32768,

    volumes={
        "/data": data_volume,
        "/model-cache": model_cache,
    },

    secrets=[
        modal.Secret.from_name(
            "researchpilot-secrets"
        ),
    ],

    env={
        # config.py will read these values on Modal.
        "VECTOR_DB_PATH":
            "/data/vector_db_qwen",

        "EMBEDDING_DEVICE":
            "cuda:0",

        "RERANKER_DEVICE":
            "cuda:1",

        # Reuse the persistent Hugging Face cache.
        "HF_HOME":
            "/model-cache",
    },

    # Keep one deployment container maximum so a sudden burst
    # of visitors cannot start several expensive copies.
    max_containers=1,

    # Keep the loaded models alive for five minutes after use,
    # then allow Modal to scale the app to zero.
    scaledown_window=300,

    # Model loading can take longer than a normal web function.
    timeout=1800,
    startup_timeout=1800,
)
@modal.concurrent(
    max_inputs=100
)
@modal.asgi_app()
def ui():

    import gradio as gr

    from fastapi import FastAPI

    # Importing app.py creates the Blocks UI and loads the
    # retrieval pipeline. app.py must NOT call demo.launch()
    # at import time.
    from app import demo

    fastapi_app = FastAPI(
        title="ResearchPilot"
    )

    return gr.mount_gradio_app(
        fastapi_app,
        demo,
        path="/",
    )
