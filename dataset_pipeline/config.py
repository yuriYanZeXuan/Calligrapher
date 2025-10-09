import os

# API configuration
API_BASE_URL = "http://35.220.164.252:3888/v1"
API_KEY = os.environ.get("API_KEY", "your_api_key_here") # It's better to use environment variables

# Model configuration
IMAGE_GEN_MODEL = "doubao-seedream-4-0-250828"
TEXT_GEN_MODEL = "gpt-4.1"

# Image configuration
IMAGE_SIZE = "1024x1024"

# OCR configuration
OCR_LANGUAGES = ['ch_sim', 'en']

# Output directory
OUTPUT_DIR = "generated_dataset"
