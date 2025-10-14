import os

# --- Service Configuration ---
# This dictionary holds settings for different services.
# The active service can be chosen in main.py using the --service flag.
SERVICES = {
    "local": {
        # --- Endpoints for local microservices ---
        # For a single, multi-GPU server, use 127.0.0.1 for both and assign different ports.
        "api_base_url_image": "http://127.0.0.1:8000/v1",
        "api_base_url_llm": "http://127.0.0.1:8001/v1",
        "api_key": "local-key", # Not used by local server, but required for client
        
        # --- API model names (should match what the local server expects) ---
        "image_gen_model": "qwen-image-edit",
        "llm_model": "qwen2.5llm",
        
        # --- Local model paths (used by local_server.py on startup) ---
        # Replace these with the actual paths to your downloaded model weights
        "image_model_path": "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_image", 
        "llm_model_path": "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen2_5Instruct"
    },
    "remote": {
        # --- Single endpoint for remote API ---
        "api_base_url": "http://35.220.164.252:3888/v1",
        "api_key": os.environ.get("API_KEY", "your_api_key_here"),
        
        # --- Remote model names ---
        "image_gen_model": "doubao-seedream-4-0-250828",
        "llm_model": "gpt-4.1",
    }
}

# --- Shared Configuration ---
IMAGE_SIZE = "1024x1024"
OUTPUT_DIR = "generated_dataset_local"

# --- Other Steps Configuration ---
OCR_LANGUAGES = ['ch_sim', 'en'] # For step 3

