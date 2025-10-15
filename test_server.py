import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import argparse
import os

def create_test_image_with_text(text="Hello", size=(256, 256)):
    """Creates a simple PIL image with text for testing."""
    img = Image.new('RGB', size, color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    try:
        # Try to load a common font, adjust size based on image size
        font_size = int(size[1] / 5)
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text position to center it
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2
    
    d.text((x, y), text, fill=(255, 255, 0), font=font)
    return img

def test_reward_server(server_url: str, prompt: str, image: Image.Image):
    """
    Sends a request to the reward server and prints the response.
    """
    print(f"--- Testing Reward Server at: {server_url} ---")
    
    # 1. Convert image to Base64
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print("Image successfully encoded to Base64.")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return

    # 2. Prepare the request payload
    payload = {
        "image": img_base64,
        "prompt": prompt
    }

    # 3. Send the POST request
    print("Sending request to the server...")
    try:
        response = requests.post(server_url, json=payload, timeout=90) # Increased timeout for model loading
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # 4. Print the result
        print("Request successful!")
        print("\n--- Server Response ---")
        try:
            result = response.json()
            print(f"  VLM Score: {result.get('vlm_score', 'N/A')}")
            print(f"  OCR Text: '{result.get('ocr_text', 'N/A')}'")
            print(f"  OCR Confidence: {result.get('ocr_confidence', 'N/A')}")
        except requests.exceptions.JSONDecodeError:
            print("  Error: Failed to decode JSON response.")
            print(f"  Raw response content: {response.text}")
        print("-----------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"\n--- ERROR ---")
        print(f"Request failed: {e}")
        print("Please ensure the reward server is running and accessible at the specified URL.")
        print("-------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for the VLM/OCR Reward Server.")
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://127.0.0.1:8000/score", 
        help="URL of the reward server's /score endpoint."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="A sign that says 'Hello'",
        help="The prompt to send with the image."
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/samples/test50_source.png",
        help="Optional path to an image file to test. If not provided, a test image will be generated."
    )
    
    args = parser.parse_args()

    # Prepare the image
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image path '{args.image_path}' does not exist.")
            exit(1)
        print(f"Using image from path: {args.image_path}")
        test_image = Image.open(args.image_path)
    else:
        print("Generating a test image with the text 'Hello'...")
        test_image = create_test_image_with_text("Hello")

    # Run the test
    test_reward_server(server_url=args.url, prompt=args.prompt, image=test_image)

    print("Test finished. Check the console output above for results.")
    print("While the request was being processed, you should have seen GPU and memory usage increase via `nvidia-smi`.")
