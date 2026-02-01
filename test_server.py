"""Test script for reward server."""

import argparse
import base64
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont


def encode(img: Image.Image) -> str:
    """Encode image to base64."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_image(text: str = "Hello", size: tuple = (256, 256)) -> Image.Image:
    """Create test image with centered text."""
    img = Image.new("RGB", size, color=(73, 109, 137))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", size[1] // 5)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - (bbox[2] - bbox[0])) // 2
    y = (size[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill=(255, 255, 0), font=font)

    return img


def test(url: str, prompt: str, img: Image.Image):
    """Send request to reward server and print response."""
    payload = {"image": encode(img), "prompt": prompt}

    print(f"Testing {url}...")
    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()

    data = resp.json()
    print(f"  VLM: {data.get('vlm_score', 'N/A')}")
    print(f"  OCR: '{data.get('ocr_text', 'N/A')}'")
    print(f"  Conf: {data.get('ocr_score', 'N/A')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/score")
    parser.add_argument("--prompt", default="A sign that says 'Hello'")
    parser.add_argument("--image", help="Path to image file (optional)")
    args = parser.parse_args()

    img = Image.open(args.image) if args.image else create_image()

    test(args.url, args.prompt, img)


if __name__ == "__main__":
    main()
