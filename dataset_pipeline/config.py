import os

# --- API Configuration ---
# Point this to your local server's address
API_BASE_URL = "http://127.0.0.1:8000/v1" 
# This can be any string, as the local server doesn't validate it
API_KEY = "local-api-key" 

# --- Model Configuration ---
# This should match the model name expected by your local server
IMAGE_GEN_MODEL = "qwen-image-edit"
IMAGE_SIZE = "1024x1024"

# --- LLaMA V3 Configuration for Step 2 ---
# (Keep this as is or update if you have a local LLaMA service)
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Directory Configuration for the pipeline ---
OUTPUT_DIR = "sample_dataset"
