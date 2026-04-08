import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# LLM API Keys - YOUR 3 MODELS
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

HF_API_KEY = os.getenv("HF_API_KEY")
GPT_OSS_BASE_URL = os.getenv("GPT_OSS_BASE_URL", "https://api-inference.huggingface.co")

GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

# Model paths
ML_MODEL_PATH = MODEL_DIR / "ml_baselines"
FEATURE_BUILDER_PATH = MODEL_DIR / "feature_builder.pkl"

# LLM configuration - YOUR EXACT 3 MODELS
LLM_CONFIG = {
    "emergency": {
        "name": "DeepSeek-R1",
        "model": "deepseek-reasoner",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "temperature": 0.2,
        "max_tokens": 300,
        "timeout": 30
    },
    "primary": {
        "name": "GPT-OSS-120B",
        "model": "openai/gpt-oss-120b",
        "api_key": HF_API_KEY,
        "base_url": GPT_OSS_BASE_URL,
        "temperature": 0.4,
        "max_tokens": 300,
        "timeout": 30
    },
    "guideline": {
        "name": "GLM-4.5V",
        "model": "glm-4-plus",
        "api_key": GLM_API_KEY,
        "base_url": GLM_BASE_URL,
        "temperature": 0.3,
        "max_tokens": 300,
        "timeout": 30
    }
}

# Consensus configuration
CONSENSUS_CONFIG = {
    "weights": {
        "emergency": float(os.getenv("EMERGENCY_WEIGHT", 0.5)),
        "guideline": float(os.getenv("GUIDELINE_WEIGHT", 0.3)),
        "primary": float(os.getenv("PRIMARY_WEIGHT", 0.2))
    },
    "safety_override_threshold": float(os.getenv("SAFETY_OVERRIDE_THRESHOLD", 0.75)),
    "low_confidence_threshold": float(os.getenv("LOW_CONFIDENCE_THRESHOLD", 0.6))
}

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / os.getenv("LOG_FILE", "medicouncil.log")

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)
