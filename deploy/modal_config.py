"""
Modal deployment configuration for Manga Image Translator.

This file contains all constants and configuration for Modal deployment.
"""

# Modal App Configuration
APP_NAME = "manga-translator"
IMAGE_NAME = "manga-translator-image"

# Volume Names
MODEL_VOLUME_NAME = "manga-models"
RESULT_VOLUME_NAME = "manga-results"

# Mount Paths
MODEL_MOUNT_PATH = "/app/models"
RESULT_MOUNT_PATH = "/app/result"
APP_ROOT_PATH = "/app"

# Secret Names
ENV_SECRET_NAME = "manga-translator-env"  # Contains all env vars including MT_WEB_NONCE

# GPU Configuration
GPU_CONFIG = {
    "gpu": "A10G",  # Options: T4 (cost-effective), A10G, A100
    "cpu": 4.0,
    "memory": 16384,  # 16GB RAM
    "timeout": 600,  # 10 minutes
    "min_containers": 0,  # Set to 1 or more for production
    "max_inputs": 2,  # Limit concurrency to prevent OOM
    "scaledown_window": 300,  # 5 minutes idle timeout
}

# Alternative GPU configs for different use cases
GPU_CONFIGS = {
    "cost_optimized": {
        "gpu": "T4",
        "cpu": 4.0,
        "memory": 16384,
        "min_containers": 0,
    },
    "balanced": {
        "gpu": "A10G",
        "cpu": 8.0,
        "memory": 32768,
        "min_containers": 1,
    },
    "performance": {
        "gpu": "A100",
        "cpu": 16.0,
        "memory": 65536,
        "min_containers": 2,
    },
}

# Base Image Configuration
BASE_IMAGE = "pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime"

# System Dependencies
APT_PACKAGES = [
    "ffmpeg",
    "libsm6",
    "libxext6",
    "libxrender-dev",
    "libgomp1",
    "libglib2.0-0",
    "curl",
    "wget",
    "git",
]

# Model Download Configuration
MODEL_CATEGORIES = {
    "detection": ["default", "ctd", "craft", "db"],
    "ocr": ["48px", "48px_ctc", "32px", "manga-ocr"],
    "inpainting": ["aot", "lama", "lama_mpe"],
    "translator": ["nllb", "nllb_big", "sugoi", "m2m100"],
}

# Default models to pre-download (for cost optimization)
DEFAULT_MODELS_TO_DOWNLOAD = [
    "default",  # Default detector
    "48px",  # Default OCR
    "aot",  # Default inpainter
]

# Estimated model sizes (in MB)
MODEL_SIZES = {
    "detection": 461,
    "ocr": 195,
    "inpainting": 320,
    "translator": 4300,
}

# Health Check Configuration
HEALTH_CHECK_ENDPOINT = "/health"
HEALTH_CHECK_TIMEOUT = 30

# Server Configuration
SERVER_PORT = 8000
SERVER_HOST = "0.0.0.0"

# Result Storage Configuration
MAX_RESULT_AGE_DAYS = 7  # Auto-cleanup results older than 7 days
MAX_RESULT_COUNT = 100  # Keep at most 100 results

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance Tuning
TORCH_OPTIMIZATIONS = {
    "allow_tf32": True,
    "cudnn_benchmark": True,
    "enable_cuda_graph": False,  # Set to True for production if stable
}

# Environment Variables for Model Loading
ENV_VARS = {
    "PYTHONPATH": "/app",
    "TORCH_HOME": "/app/models",
    "HF_HOME": "/app/models/huggingface",
    "TRANSFORMERS_CACHE": "/app/models/transformers",
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONUNBUFFERED": "1",
}

# Required secrets (these should be set in Modal)
REQUIRED_ENV_VARS = [
    "MT_WEB_NONCE",  # Required for master/worker authentication
]

# Optional API keys (for translation services)
OPTIONAL_ENV_VARS = [
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "DEEPL_AUTH_KEY",
    "BAIDU_APP_ID",
    "BAIDU_SECRET_KEY",
    "YOUDAO_APP_KEY",
    "YOUDAO_SECRET_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "CAIYUN_TOKEN",
    "SAKURA_API_BASE",
]

# Deployment Modes
DEPLOYMENT_MODE = "standalone"  # Options: "standalone", "distributed"

# Rate Limiting (if needed)
RATE_LIMIT_CONFIG = {
    "enabled": False,
    "requests_per_minute": 60,
    "burst_size": 10,
}

# Monitoring and Alerting
MONITORING_CONFIG = {
    "enable_metrics": True,
    "enable_traces": False,
    "log_requests": True,
    "log_response_time": True,
}
