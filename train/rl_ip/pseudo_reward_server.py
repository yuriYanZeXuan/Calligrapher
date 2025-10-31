import argparse
import base64
import io
import logging
import random
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import uvicorn
import os
import fcntl
import numpy as np


DEBUG_SAVE_DIR = "reward_debug_output"
COUNTER_FILE = os.path.join(DEBUG_SAVE_DIR, "counter.txt")
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
if not os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE, "w") as f:
        f.write("0")


def save_debug_sample(image_pil: Image.Image, masked_image_pil: Image.Image, prompt: str, vlm_score: float, ocr_confidence: float):
    with open(COUNTER_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        content = f.read()
        current_count = int(content) if content else 0
        new_count = current_count + 1
        f.seek(0)
        f.truncate()
        f.write(str(new_count))
        fcntl.flock(f, fcntl.LOCK_UN)

    if new_count % 20 == 0:
        try:
            filename_base = f"{new_count:06d}_vlm_{vlm_score:.2f}_ocr_{ocr_confidence:.2f}_pseudo"
            image_path = os.path.join(DEBUG_SAVE_DIR, f"{filename_base}.png")
            masked_path = os.path.join(DEBUG_SAVE_DIR, f"{filename_base}_masked.png")
            info_path = os.path.join(DEBUG_SAVE_DIR, f"{filename_base}.txt")

            image_pil.save(image_path)
            masked_image_pil.save(masked_path)

            with open(info_path, "w", encoding="utf-8") as info_f:
                info_f.write(f"Prompt: {prompt}\n")
                info_f.write(f"VLM Score (pseudo): {vlm_score}\n")
                info_f.write(f"OCR Confidence (pseudo): {ocr_confidence}\n")
        except Exception as exc:
            logger.warning("Failed to save pseudo debug sample: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pseudo reward server starting up...")
    yield
    logger.info("Pseudo reward server shutting down...")


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("uvicorn")


class ScoreRequest(BaseModel):
    image: str
    prompt: str
    mask: Optional[str] = None


class PseudoRewardGenerator:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def sample_scores(self):
        vlm_score = self._rng.random()
        ocr_confidence = self._rng.random()
        combined = 0.5 * vlm_score + 0.5 * ocr_confidence
        return vlm_score, ocr_confidence, combined


generator: Optional[PseudoRewardGenerator] = None


@app.post("/score")
async def score(request: ScoreRequest):
    global generator

    # Decode inputs for API compatibility (errors are logged but do not break the response)
    try:
        image_bytes = base64.b64decode(request.image)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to decode image payload: %s", exc)
        image_pil = Image.new("RGB", (512, 512), color="gray")

    mask_pil = None
    if request.mask:
        try:
            mask_bytes = base64.b64decode(request.mask)
            mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
            if mask_pil.size != image_pil.size:
                mask_pil = mask_pil.resize(image_pil.size, Image.NEAREST)
        except Exception as exc:
            logger.warning("Failed to decode mask payload: %s", exc)
            mask_pil = None

    masked_image_pil = image_pil
    if mask_pil is not None:
        mask_array = (np.array(mask_pil) > 127).astype(np.uint8)
        image_array = np.array(image_pil)
        masked_array = image_array.copy()
        masked_array[mask_array == 0] = 255
        masked_image_pil = Image.fromarray(masked_array)

    vlm_score, ocr_confidence, combined_score = generator.sample_scores()

    save_debug_sample(image_pil, masked_image_pil, request.prompt, vlm_score, ocr_confidence)

    return {
        "vlm_score": vlm_score,
        "ocr_text": "",
        "ocr_confidence": ocr_confidence,
        "combined_score": combined_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Launch a pseudo reward server that returns random scores.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    args = parser.parse_args()

    global generator
    generator = PseudoRewardGenerator(seed=args.seed)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

