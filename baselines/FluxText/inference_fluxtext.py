#!/usr/bin/env python3
"""Standalone inference script for FluxText model.

Usage (CLI):
    python inference_fluxtext.py \
        --model_path /path/to/fluxtext_lora.safetensors \
        --config_path train/config/word_512_size.yaml \
        --prompt 'a logo that reads "HELLO"' \
        --output_path output/fluxtext_result.png \
        --seed 42

Usage (Python API):
    from inference_fluxtext import FluxTextGenerator
    gen = FluxTextGenerator(
        model_path="/path/to/fluxtext_lora.safetensors",
        config_path="train/config/word_512_size.yaml",
    )
    gen.generate(prompt='a logo that reads "HELLO"', output_path="out.png")
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

# Ensure FluxText source is importable
FLUXTEXT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FLUXTEXT_DIR)

from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll

# ---------------------------------------------------------------------------
# Aspect‑ratio helpers (ported from app.py)
# ---------------------------------------------------------------------------

ASPECT_RATIO_LD_LIST = [
    "2.39:1", "2:1", "16:9", "1.85:1", "9:16",
    "5:8", "3:2", "4:3", "1:1",
]
PIXELS = [512 * 512, 768 * 768, 1024 * 1024]


def _get_ratio(name: str) -> float:
    w, h = map(float, name.split(":"))
    return h / w


def _get_closest_ratio(height: float, width: float) -> str:
    return min(ASPECT_RATIO_LD_LIST, key=lambda r: abs(height / width - _get_ratio(r)))


def _get_aspect_ratios_dict(total_pixels: int = 256 * 256) -> dict:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    aspect_ratios = {}
    aspect_ratios_v = {}
    for ratio in ASPECT_RATIO_LD_LIST:
        wr, hr = map(float, ratio.split(":"))
        w = int(math.sqrt(total_pixels * (wr / hr)) // D) * D
        h = int((total_pixels / w) // D) * D
        # fine‑tune to match total pixels
        diff = abs(h * w - total_pixels)
        for ch, cw in [(h - D, w), (h + D, w), (h, w - D), (h, w + D)]:
            if abs(ch * cw - total_pixels) < diff:
                h, w = ch, cw
                diff = abs(ch * cw - total_pixels)
        if (h, w) not in aspect_ratios.values():
            aspect_ratios[ratio] = (h, w)
            vr = ":".join(ratio.split(":")[::-1])
            aspect_ratios_v[vr] = (w, h)
    aspect_ratios.update(aspect_ratios_v)
    return aspect_ratios


# ---------------------------------------------------------------------------
# Glyph rendering helpers
# ---------------------------------------------------------------------------

def _insert_spaces(string, n_space):
    if n_space == 0:
        return string
    return (" " * n_space).join(string)[: -(n_space) if n_space else None]


def _draw_glyph(font, text, polygon, scale=1, width=512, height=512,
                vert_ang=10, add_space=True):
    """Render *text* inside *polygon* on a black canvas (white glyphs)."""
    enlarge_polygon = polygon * scale
    rect = cv2.minAreaRect(enlarge_polygon)
    box = cv2.boxPoints(rect)
    box = np.int8(box)
    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if abs(angle) % 90 < vert_ang or abs(90 - abs(angle) % 90) % 90 < vert_ang:
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = Image.fromarray(np.zeros((height * scale, width * scale, 3), np.uint8))

    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = _insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = _insert_spaces(text, i - 1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.75 if vert else 0.85
        font_size = min(w, h) / (text_w / max(w, h)) * shrink
    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text(
            (rect[0][0] - text_width // 2, rect[0][1] - text_height // 2 - top),
            text, font=new_font, fill=(255, 255, 255, 255),
        )
    else:
        x_s = min(box[:, 0]) + (max(box[:, 0]) - min(box[:, 0])) // 2 - text_height // 2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))
    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    img = np.expand_dims(np.array(img.convert("1")), axis=2).astype(np.float64)
    return img


def _make_full_mask_and_glyph(text, font, width, height):
    """Create a full‑image mask and glyph for *text* that covers the whole canvas.

    For the benchmark scenario we have no pre‑existing image / mask, so we
    create a full‑canvas bounding box as the text region.
    """
    # Full‑image mask (all ones → paint everywhere)
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Build a rectangle contour covering ~90 % of the canvas (centred)
    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)
    contour = np.array([
        [margin_x, margin_y],
        [width - margin_x, margin_y],
        [width - margin_x, height - margin_y],
        [margin_x, height - margin_y],
    ], dtype=np.int32).reshape(-1, 1, 2)

    glyphs = _draw_glyph(font, text, contour, scale=1, width=width, height=height)
    return mask, glyphs


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class FluxTextGenerator:
    """High‑level wrapper around FluxText for single‑image generation."""

    DEFAULT_FONT_PATH = os.path.join(FLUXTEXT_DIR, "font", "Arial_Unicode.ttf")

    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = "cuda",
        font_path: str = None,
    ):
        self.device = device
        self.font_path = font_path or self.DEFAULT_FONT_PATH
        self.font = ImageFont.truetype(self.font_path, size=60)

        # Load config
        if config_path is None:
            config_path = os.path.join(FLUXTEXT_DIR, "train", "config", "word_512_size.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        print(f"Initializing FluxText on {device} ...")
        training_cfg = self.config["train"]
        trainable_model = OminiModelFIll(
            flux_pipe_id=self.config["flux_path"],
            lora_config=training_cfg["lora_config"],
            device=device,
            dtype=getattr(torch, self.config["dtype"]),
            optimizer_config=training_cfg["optimizer"],
            model_config=self.config.get("model", {}),
            gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
            byt5_encoder_config=training_cfg.get("byt5_encoder", None),
        )

        # Load LoRA weights
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        state_dict = {
            k.replace("lora_A", "lora_A.default")
             .replace("lora_B", "lora_B.default")
             .replace("transformer.", ""): v
            for k, v in state_dict.items()
        }
        trainable_model.transformer.load_state_dict(state_dict, strict=False)

        self.pipe = trainable_model.flux_pipe
        self.model_config = self.config.get("model", {})
        self._generator = torch.Generator(device=device)
        print("FluxText initialized.")

    # ---- public API --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        output_path: str,
        text: str = None,
        image: Image.Image = None,
        mask: np.ndarray = None,
        glyph: np.ndarray = None,
        seed: int = 42,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """Generate an image with embedded text.

        Parameters
        ----------
        prompt : str
            Text prompt describing the desired image.
        output_path : str
            Where to save the result.
        text : str, optional
            The literal text to render. If *None*, attempt to parse from
            ``prompt`` (looks for quoted substrings). If still empty, a
            full white mask with no glyph condition is used.
        image : PIL.Image, optional
            Background image. Defaults to a white image of *width* x *height*.
        mask / glyph : np.ndarray, optional
            Pre‑computed mask and glyph arrays. If not supplied they are
            generated automatically from *text*.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Resolve target size ------------------------------------------------
        if image is not None:
            ori_w, ori_h = image.size
        else:
            ori_w, ori_h = width, height
            image = Image.new("RGB", (ori_w, ori_h), "white")

        num_pixel = min(PIXELS, key=lambda x: abs(x - ori_w * ori_h))
        ar_dict = _get_aspect_ratios_dict(num_pixel)
        close_ratio = _get_closest_ratio(ori_h, ori_w)
        tgt_h, tgt_w = ar_dict[close_ratio]

        # Resolve text -------------------------------------------------------
        if text is None:
            # try to extract from prompt (quoted strings)
            import re
            matches = re.findall(r'["\'](.+?)["\']', prompt)
            text = " ".join(matches) if matches else ""

        # Build condition images ---------------------------------------------
        if mask is not None and glyph is not None:
            hint_img = Image.fromarray(mask).resize((tgt_w, tgt_h)).convert("RGB")
            glyph_img = Image.fromarray(
                ((1 - glyph) * 255).astype(np.uint8).squeeze()
            ).resize((tgt_w, tgt_h)).convert("RGB")
        elif text.strip():
            raw_mask, raw_glyph = _make_full_mask_and_glyph(
                text, self.font, tgt_w, tgt_h
            )
            hint_img = Image.fromarray(raw_mask).resize((tgt_w, tgt_h)).convert("RGB")
            glyph_img_arr = ((1 - raw_glyph) * 255).astype(np.uint8).squeeze()
            glyph_img = Image.fromarray(glyph_img_arr).resize((tgt_w, tgt_h)).convert("RGB")
        else:
            # No text → full white mask, white glyph (no text condition)
            hint_img = Image.new("RGB", (tgt_w, tgt_h), "white")
            glyph_img = Image.new("RGB", (tgt_w, tgt_h), "white")

        img_resized = image.resize((tgt_w, tgt_h))

        # Convert to condition arrays ----------------------------------------
        hint_arr = np.array(hint_img) / 255.0
        cond_arr = np.array(glyph_img)
        cond_arr = (255 - cond_arr) / 255.0

        condition_img = [cond_arr, hint_arr, img_resized]
        condition = Condition(
            condition_type="word_fill",
            condition=condition_img,
            position_delta=[0, 0],
        )

        # Run generation -----------------------------------------------------
        self._generator.manual_seed(seed)
        result = generate_fill(
            self.pipe,
            prompt=prompt,
            conditions=[condition],
            height=tgt_h,
            width=tgt_w,
            generator=self._generator,
            model_config=self.model_config,
            default_lora=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        out_img = result.images[0]
        out_img.save(output_path)
        print(f"Image saved to {output_path}")
        return out_img


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FluxText Inference Script")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the FluxText LoRA safetensors checkpoint.")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to the YAML config (default: train/config/word_512_size.yaml).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the desired image.")
    parser.add_argument("--text", type=str, default=None,
                        help="Literal text to render (if not embedded in prompt).")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Optional background image path.")
    parser.add_argument("--output_path", type=str, default="output/fluxtext_result.png",
                        help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28,
                        help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--font_path", type=str, default=None,
                        help="Path to a .ttf font file for glyph rendering.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    generator = FluxTextGenerator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        font_path=args.font_path,
    )

    image = None
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")

    generator.generate(
        prompt=args.prompt,
        output_path=args.output_path,
        text=args.text,
        image=image,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
