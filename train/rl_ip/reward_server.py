
import argparse
import base64
import io
from fastapi import FastAPI, Request
from PIL import Image
import uvicorn
from pydantic import BaseModel
import logging
import os

# Import both scorers
from qwenvl import QwenVLScorer
from ocr import OCRScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variables to hold the scorers
qwen_scorer = None
ocr_scorer = None

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
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
