import os
import uuid
from datetime import datetime
from typing import Optional

from PIL import Image


def _resolve_debug_dir() -> str:
    return os.environ.get("REWARD_DEBUG_DIR", "reward_debug_output")


def save_debug_sample(
    image: Image.Image,
    masked_image: Image.Image,
    prompt: str,
    vlm_score: float,
    ocr_confidence: float,
    ocr_text: str = "",
    prefix: str = "reward",
    timestep: Optional[str] = None,
) -> None:
    """Save debug artifacts for reward evaluation.

    Files are named with a timestamp + random suffix to avoid collisions, so no
    inter-process locking is required.
    """

    debug_dir = _resolve_debug_dir()
    os.makedirs(debug_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique = uuid.uuid4().hex[:8]
    base_parts = [prefix, timestamp, unique]
    if timestep is not None:
        safe_timestep = str(timestep).replace(":", "-").replace(" ", "_")
        base_parts.append(f"t_{safe_timestep}")
    base_parts.extend([f"vlm_{vlm_score:.2f}", f"ocr_{ocr_confidence:.2f}"])
    base_name = "_".join(base_parts)

    image_path = os.path.join(debug_dir, f"{base_name}.png")
    masked_path = os.path.join(debug_dir, f"{base_name}_masked.png")
    info_path = os.path.join(debug_dir, f"{base_name}.txt")

    try:
        image.save(image_path)
        masked_image.save(masked_path)
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"VLM Score: {vlm_score}\n")
            f.write(f"OCR Confidence: {ocr_confidence}\n")
            if timestep is not None:
                f.write(f"Denoise Timestep: {timestep}\n")
            if ocr_text:
                f.write(f"OCR Text: {ocr_text}\n")
    except Exception as exc:
        # Debug saving should never interrupt the reward flow.
        print(f"[debug_utils] Failed to save debug sample: {exc}")


