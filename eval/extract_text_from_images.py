import os
import openai
from openai import OpenAI
import base64
import json
import re
import sys
from PIL import Image
import glob
from tqdm import tqdm # Recommended: pip install tqdm

class ImageTextExtractor:
    def __init__(self, model="gpt-4o"):
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable not set.")
        self.client = OpenAI(
            base_url="http://35.220.164.252:3888/v1",
            api_key=api_key
        )
        self.model = model
        print(f"ImageTextExtractor initialized with model: {self.model} at custom base URL.")

    def _encode_image(self, image: Image.Image):
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_text(self, image: Image.Image):
        """
        Extracts text from an image using GPT-4o.
        """
        img_b64 = self._encode_image(image)
        
        prompt = """
        You are an Optical Character Recognition (OCR) expert. Your task is to read the text from the provided image.
        Please return only the text you see in the image. Do not add any explanation, context, or any other words like "The text is". Just return the raw text.
        If there is no text, return an empty response.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    }
                ],
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip()
            return content
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return f"Error: {e}"


def main():
    """
    Main function to extract text from images and save to a file.
    You MUST set the API_KEY environment variable to run this.
    """
    if not os.environ.get("API_KEY"):
        print("="*50)
        print("!!! ERROR: API_KEY environment variable is not set. !!!")
        print("Please set it to your API key to run this evaluation.")
        print("Example: export API_KEY='your-key'")
        print("="*50)
        return
        
    # --- IMPORTANT ---
    # Change this path to your local path
    benchmark_dir = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing"
    output_file = "self_label.txt"
    # -----------------
    
    if not os.path.exists(benchmark_dir):
        print("="*50)
        print("!!! PLEASE UPDATE THE PLACEHOLDER PATHS IN `extract_text_from_images.py` main function !!!")
        print(f"Directory path checked: {benchmark_dir}")
        print("="*50)
        return

    image_paths = sorted(glob.glob(os.path.join(benchmark_dir, "test*_ref.png")))
    
    if not image_paths:
        print(f"No images found matching 'test*_ref.png' in {benchmark_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    extractor = ImageTextExtractor()
    
    extracted_texts = []
    
    for image_path in tqdm(image_paths, desc="Extracting text from images"):
        try:
            image = Image.open(image_path).convert("RGB")
            text = extractor.extract_text(image)
            extracted_texts.append(text)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            extracted_texts.append(f"Error processing {os.path.basename(image_path)}")

    with open(output_file, 'w') as f:
        for text in extracted_texts:
            f.write(f"{text}\n")
            
    print(f"\nExtraction complete. Results saved to {output_file}")


if __name__ == '__main__':
    main()
