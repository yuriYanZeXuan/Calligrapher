
import argparse
import base64
import io
from fastapi import FastAPI, Request
from PIL import Image
import uvicorn
from pydantic import BaseModel
import logging
import os
import fcntl # For process-safe file locking

# Import both scorers
from qwenvl import QwenVLScorer
from ocr import OCRScorer

# Configure logging - THIS IS NO LONGER NEEDED as Uvicorn will handle it.
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI()

# Global variables to hold the scorers
qwen_scorer = None
ocr_scorer = None

# --- New: Globals for debug image saving ---
DEBUG_SAVE_DIR = "reward_debug_output"
COUNTER_FILE = os.path.join(DEBUG_SAVE_DIR, "counter.txt")
# Ensure the directory and counter file exist on startup
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
if not os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE, "w") as f:
        f.write("0")
# --- End of new globals ---

def save_debug_sample(image_pil, prompt, vlm_score, ocr_text, ocr_confidence):
    """
    Process-safe function to increment a counter and save a debug sample every 100 images.
    """
    with open(COUNTER_FILE, "r+") as f:
        # Use a file lock to prevent race conditions between multiple server processes
        fcntl.flock(f, fcntl.LOCK_EX)
        
        content = f.read()
        current_count = int(content) if content else 0
        new_count = current_count + 1
        
        f.seek(0)
        f.truncate()
        f.write(str(new_count))
        
        fcntl.flock(f, fcntl.LOCK_UN)
    
    # Save a sample every 100 images
    if new_count % 20 == 0:
        try:
            filename_base = f"{new_count:06d}_vlm_{vlm_score:.2f}_ocr_{ocr_confidence:.2f}"
            image_path = os.path.join(DEBUG_SAVE_DIR, f"{filename_base}.png")
            info_path = os.path.join(DEBUG_SAVE_DIR, f"{filename_base}.txt")

            image_pil.save(image_path)

            with open(info_path, "w", encoding="utf-8") as info_f:
                info_f.write(f"Prompt: {prompt}\n")
                info_f.write(f"VLM Score: {vlm_score}\n")
                info_f.write(f"OCR Text: {ocr_text}\n")
                info_f.write(f"OCR Confidence: {ocr_confidence}\n")

            logger.info(f"Saved debug image and info for count {new_count} to {DEBUG_SAVE_DIR}")
        except Exception as e:
            logger.error(f"Failed to save debug image for count {new_count}: {e}")

class ScoreRequest(BaseModel):
    image: str  # Base64 encoded image string
    prompt: str

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    # This is where we would initialize the model, but we do it before `uvicorn.run`
    # to pass arguments to the scorer.
    pass

@app.post("/score")
async def get_score(request: ScoreRequest):
    try:
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')
        logger.info(f"Received score request on GPU: {gpu_id}")

        image_data = base64.b64decode(request.image)
        image_pil = Image.open(io.BytesIO(image_data))
        
        # Get scores from both models
        vlm_score = qwen_scorer.score(image_pil, request.prompt)
        ocr_text, ocr_confidence = ocr_scorer.score(image_pil)
        
        logger.info(f"Scored prompt: '{request.prompt[:30]}...' -> VLM: {vlm_score:.4f}, OCR: '{ocr_text}', OCR Conf: {ocr_confidence:.4f}")
        
        # --- Call the new debug saving function ---
        save_debug_sample(image_pil, request.prompt, vlm_score, ocr_text, ocr_confidence)

        return {
            "vlm_score": vlm_score,
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_confidence
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {"error": str(e)}, 500

def main():
    parser = argparse.ArgumentParser(description="Run the VLM and OCR Reward Model API Server.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen-VL model.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda:0', 'cpu').")
    args = parser.parse_args()

    global qwen_scorer, ocr_scorer
    logger.info(f"Loading QwenVLScorer model from {args.model_path} on device {args.device}...")
    qwen_scorer = QwenVLScorer(model_path=args.model_path, device=args.device)
    logger.info("QwenVLScorer model loaded successfully.")
    
    logger.info("Loading OCRScorer model...")
    ocr_scorer = OCRScorer()
    logger.info("OCRScorer model loaded successfully.")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
