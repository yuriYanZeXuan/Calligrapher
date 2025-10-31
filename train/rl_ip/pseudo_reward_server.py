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
        base64.b64decode(request.image)
    except Exception as exc:
        logger.warning("Failed to decode image payload: %s", exc)

    if request.mask:
        try:
            base64.b64decode(request.mask)
        except Exception as exc:
            logger.warning("Failed to decode mask payload: %s", exc)

    vlm_score, ocr_confidence, combined_score = generator.sample_scores()

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

