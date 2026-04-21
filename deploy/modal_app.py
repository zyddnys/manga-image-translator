"""
Modal deployment for Manga Image Translator.

This deployment correctly handles the Master/Worker subprocess architecture:
- Master: FastAPI server (server/main.py) that handles HTTP requests
- Worker: MangaTranslator subprocess (mode/share.py) that does actual ML processing
- Starts worker subprocess within the container
- Manages lifecycle of both master and worker processes
- Properly forwards environment variables and GPU configuration
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

import modal

# Get the parent directory (project root) relative to this file
DEPLOY_DIR = Path(__file__).parent
PROJECT_ROOT = DEPLOY_DIR.parent

# Import configuration
from modal_config import (
    APP_NAME,
    IMAGE_NAME,
    MODEL_VOLUME_NAME,
    RESULT_VOLUME_NAME,
    MODEL_MOUNT_PATH,
    RESULT_MOUNT_PATH,
    APP_ROOT_PATH,
    ENV_SECRET_NAME,
    GPU_CONFIG,
    BASE_IMAGE,
    APT_PACKAGES,
    ENV_VARS,
)

# Create Modal app
app = modal.App(APP_NAME)

# Create or reference persistent volumes
model_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    create_if_missing=True
)
result_volume = modal.Volume.from_name(
    RESULT_VOLUME_NAME,
    create_if_missing=True
)

# Build container image
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.10")
    # Install system packages including build tools (needed for pyhyphen, pydensecrf)
    .apt_install([
        "build-essential",  # gcc, g++, make
        "gcc",
        "g++",
    ] + APT_PACKAGES)
    .env(ENV_VARS)
    # Copy requirements.txt and install Python dependencies
    .add_local_file(str(PROJECT_ROOT / "requirements.txt"), "/app/requirements.txt", copy=True)
    .add_local_file(str(DEPLOY_DIR / "modal_config.py"), "/root/modal_config.py", copy=True)
    .run_commands(
        "cd /app && pip install --no-cache-dir -r requirements.txt",
        gpu="t4",  # Use GPU during build for PyTorch installation
    )
    # Create necessary directories
    .run_commands(
        "mkdir -p /app/models",
        "mkdir -p /app/result",
        "mkdir -p /app/upload-cache",
    )
    # Copy application code (will be mounted on container startup for fast iteration)
    .add_local_dir(str(PROJECT_ROOT / "manga_translator"), "/app/manga_translator")
    .add_local_dir(str(PROJECT_ROOT / "server"), "/app/server")
    .add_local_file(str(PROJECT_ROOT / "docker_prepare.py"), "/app/docker_prepare.py")
)


@app.function(
    image=image,
    gpu=GPU_CONFIG["gpu"],
    cpu=GPU_CONFIG["cpu"],
    memory=GPU_CONFIG["memory"],
    timeout=GPU_CONFIG["timeout"],
    min_containers=GPU_CONFIG["min_containers"],
    scaledown_window=GPU_CONFIG.get("scaledown_window", 300),
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        RESULT_MOUNT_PATH: result_volume,
    },
    secrets=[
        modal.Secret.from_name(ENV_SECRET_NAME),
    ],
)
@modal.concurrent(max_inputs=GPU_CONFIG["max_inputs"])
@modal.asgi_app()
def web():
    """
    Main ASGI web application endpoint with worker subprocess.

    This function:
    1. Starts a worker subprocess (mode=shared) that does actual ML processing
    2. Imports and returns the FastAPI app (master) that handles HTTP requests
    3. Manages the lifecycle of both processes

    Architecture:
    - Master (FastAPI): Handles HTTP, queues tasks, returns results
    - Worker (subprocess): Loads models, runs detection/OCR/inpainting
    - Communication: HTTP requests with pickle serialization + nonce auth
    """
    import sys
    import subprocess
    import os
    import time
    import atexit
    sys.path.insert(0, "/app")

    # Get nonce from environment (set by Modal secret)
    nonce = os.environ.get('MT_WEB_NONCE')
    if not nonce:
        print("WARNING: MT_WEB_NONCE not set, generating temporary nonce")
        import secrets
        nonce = secrets.token_hex(16)
        os.environ['MT_WEB_NONCE'] = nonce

    # Worker configuration
    worker_host = "127.0.0.1"
    worker_port = 5004  # Different from master port

    # Check if GPU is available
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        print(f"GPU available: {use_gpu}")
    except Exception as e:
        print(f"Could not detect GPU: {e}")

    # Start worker subprocess
    worker_cmd = [
        sys.executable,
        '-m', 'manga_translator',
        'shared',
        '--host', worker_host,
        '--port', str(worker_port),
        '--nonce', nonce,
    ]

    if use_gpu:
        worker_cmd.append('--use-gpu')

    print(f"Starting worker subprocess: {' '.join(worker_cmd)}")

    worker_process = subprocess.Popen(
        worker_cmd,
        cwd="/app",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give worker time to start
    print("Waiting for worker to start...")
    time.sleep(5)

    # Check if worker is still running
    if worker_process.poll() is not None:
        stdout, stderr = worker_process.communicate()
        print(f"Worker failed to start!")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        raise RuntimeError("Worker subprocess failed to start")

    print(f"Worker subprocess started with PID: {worker_process.pid}")

    # Register worker instance with master
    # This is done automatically by server.main when --start-instance is used,
    # but since we're starting manually, we need to do it ourselves
    from server.instance import ExecutorInstance, executor_instances
    executor_instances.register(ExecutorInstance(ip=worker_host, port=worker_port))
    print(f"Registered worker at {worker_host}:{worker_port}")

    # Setup cleanup on exit
    def cleanup_worker():
        print("Cleaning up worker subprocess...")
        if worker_process.poll() is None:
            worker_process.terminate()
            try:
                worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker_process.kill()
        print("Worker subprocess terminated")

    atexit.register(cleanup_worker)

    # Import the FastAPI app (master)
    from server.main import app as fastapi_app
    from server.main import prepare
    from argparse import Namespace

    # Prepare the server (initialize upload-cache, etc.)
    args = Namespace(
        host=worker_host,
        port=worker_port,
        nonce=nonce,
        start_instance=False,  # We already started it manually
        use_gpu=use_gpu,
        use_gpu_limited=False,
        ignore_errors=False,
        verbose=True,
        models_ttl=None,
        pre_dict=None,
        post_dict=None,
    )
    prepare(args)

    # Add health check endpoint
    from fastapi import Response

    @fastapi_app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        worker_alive = worker_process.poll() is None

        # Check if worker is responding
        worker_healthy = False
        if worker_alive:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://{worker_host}:{worker_port}/is_locked",
                        timeout=5.0
                    )
                    worker_healthy = response.status_code == 200
            except Exception as e:
                print(f"Worker health check failed: {e}")

        return {
            "status": "healthy" if (worker_alive and worker_healthy) else "degraded",
            "service": "manga-translator",
            "gpu_available": use_gpu,
            "worker": {
                "pid": worker_process.pid,
                "alive": worker_alive,
                "healthy": worker_healthy,
                "host": worker_host,
                "port": worker_port,
            }
        }

    return fastapi_app


@app.function(
    image=image,
    gpu=GPU_CONFIG["gpu"],
    cpu=2.0,
    memory=8192,
    timeout=3600,  # 1 hour for model downloads
    volumes={
        MODEL_MOUNT_PATH: model_volume,
    },
)
def download_models():
    """
    Download all required models to the persistent volume.

    This function should be run once during initial setup to pre-populate
    the model cache. It can also be run periodically to update models.

    Usage:
        modal run deploy.modal_app::download_models
    """
    import subprocess
    import sys

    print("Starting model download...")
    print(f"Model volume mounted at: {MODEL_MOUNT_PATH}")

    # Set environment variables for model paths BEFORE running script
    env = os.environ.copy()
    env["TORCH_HOME"] = MODEL_MOUNT_PATH
    env["HF_HOME"] = f"{MODEL_MOUNT_PATH}/huggingface"
    env["TRANSFORMERS_CACHE"] = f"{MODEL_MOUNT_PATH}/transformers"
    env["XDG_CACHE_HOME"] = f"{MODEL_MOUNT_PATH}/cache"

    print(f"Environment variables set:")
    print(f"  TORCH_HOME={env['TORCH_HOME']}")
    print(f"  HF_HOME={env['HF_HOME']}")
    print(f"  TRANSFORMERS_CACHE={env['TRANSFORMERS_CACHE']}")

    # Run the docker_prepare.py script with environment
    try:
        result = subprocess.run(
            [sys.executable, "/app/docker_prepare.py", "--continue-on-error"],
            cwd="/app",
            env=env,  # Pass environment variables
            check=True,
            capture_output=True,
            text=True,
        )
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # List what was downloaded
        print("\nChecking downloaded files...")
        list_result = subprocess.run(
            ["find", MODEL_MOUNT_PATH, "-type", "f", "-name", "*.onnx", "-o", "-name", "*.pt", "-o", "-name", "*.pth"],
            capture_output=True,
            text=True,
        )
        if list_result.stdout:
            print("Found model files:")
            print(list_result.stdout)
        else:
            print("⚠️ Warning: No model files found!")

        # Commit the volume changes
        print("\nCommitting volume changes...")
        model_volume.commit()

        print("✅ Model download completed successfully!")
        return {"status": "success", "message": "All models downloaded"}

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during model download: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)

        # Still commit partial downloads
        model_volume.commit()

        return {
            "status": "partial_failure",
            "message": "Some models may have failed to download",
            "error": str(e)
        }


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    volumes={
        RESULT_MOUNT_PATH: result_volume,
    },
)
def cleanup_old_results(max_age_days: int = 7, max_count: int = 100):
    """
    Clean up old result files from the result volume.

    Args:
        max_age_days: Delete results older than this many days
        max_count: Keep at most this many most recent results

    Usage:
        modal run deploy.modal_app::cleanup_old_results --max-age-days 7
    """
    import shutil
    from datetime import datetime, timedelta

    result_path = Path(RESULT_MOUNT_PATH)

    if not result_path.exists():
        print("Result directory does not exist")
        return {"status": "skipped", "message": "No results to clean"}

    # Get all result directories
    result_dirs = [d for d in result_path.iterdir() if d.is_dir()]
    result_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    deleted_count = 0
    kept_count = 0
    cutoff_time = datetime.now().timestamp() - (max_age_days * 86400)

    for i, result_dir in enumerate(result_dirs):
        # Keep the most recent max_count results
        if i < max_count:
            # But still delete if too old
            if result_dir.stat().st_mtime < cutoff_time:
                print(f"Deleting old result: {result_dir.name}")
                shutil.rmtree(result_dir)
                deleted_count += 1
            else:
                kept_count += 1
        else:
            # Delete anything beyond max_count
            print(f"Deleting excess result: {result_dir.name}")
            shutil.rmtree(result_dir)
            deleted_count += 1

    # Commit volume changes
    result_volume.commit()

    print(f"✅ Cleanup completed: {deleted_count} deleted, {kept_count} kept")
    return {
        "status": "success",
        "deleted": deleted_count,
        "kept": kept_count,
    }


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        RESULT_MOUNT_PATH: result_volume,
    },
)
def list_volumes():
    """
    List contents of both volumes for debugging.

    Usage:
        modal run deploy.modal_app::list_volumes
    """
    import subprocess

    print("=" * 60)
    print("MODEL VOLUME CONTENTS:")
    print("=" * 60)
    result = subprocess.run(
        ["du", "-h", "-d", "2", MODEL_MOUNT_PATH],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    print("\n" + "=" * 60)
    print("RESULT VOLUME CONTENTS:")
    print("=" * 60)
    result = subprocess.run(
        ["ls", "-lh", RESULT_MOUNT_PATH],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    return {"status": "success"}


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """
    Local entrypoint for testing Modal functions.

    Usage:
        modal run deploy.modal_app
    """
    print("Manga Image Translator - Modal Deployment")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  modal deploy deploy.modal_app         # Deploy the web service")
    print("  modal run deploy.modal_app::download_models  # Download models")
    print("  modal run deploy.modal_app::cleanup_old_results  # Clean results")
    print("  modal run deploy.modal_app::list_volumes  # List volume contents")
    print("\nFor more information, see deploy/README.md")
