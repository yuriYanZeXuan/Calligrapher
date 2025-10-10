import os
import openai
from openai import OpenAI
import base64
import json
import re
import sys
from PIL import Image

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import load_images_for_evaluation


class VLMEvaluator:
    def __init__(self, model="gpt-4o"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"VLM Evaluator initialized with model: {self.model}")

    def _encode_image(self, image: Image.Image):
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _parse_score(self, response_content: str):
        """Parses score from a response like 'Score: 8/10' or 'Score: 8'."""
        match = re.search(r"Score:\s*(\d+)", response_content, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def evaluate_aesthetic(self, generated_img: Image.Image, ref_img: Image.Image):
        """
        Evaluates aesthetic quality and style consistency.
        """
        gen_b64 = self._encode_image(generated_img)
        ref_b64 = self._encode_image(ref_img)
        
        prompt = """
        You are an expert in typography and graphic design. Your task is to evaluate a generated image based on a provided reference image.
        
        Evaluate the following criteria and provide a single score from 1 (worst) to 10 (best):
        1.  **Text Aesthetics**: How visually appealing is the rendered text? Consider its form, clarity, and artistic quality.
        2.  **Style Consistency**: How well does the style of the rendered text (e.g., color, texture, font style) match the style of the text in the reference image?
        3.  **Background Integration**: How naturally is the text integrated into the background of the generated image?
        
        The first image is the generated one to be evaluated. The second image is the style reference.
        
        Please provide your response in the following format ONLY:
        Score: [Your score from 1-10]
        Justification: [A brief, one-sentence explanation for your score]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                        ],
                    }
                ],
                max_tokens=100,
            )
            content = response.choices[0].message.content
            score = self._parse_score(content)
            return score, content
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return None, str(e)


    def evaluate_text_image_match(self, generated_img: Image.Image, prompt_text: str):
        """
        Evaluates how accurately the rendered text matches the prompt.
        """
        gen_b64 = self._encode_image(generated_img)
        
        prompt = f"""
        You are an AI assistant evaluating text rendering in an image. Your task is to compare the text visible in the provided image with a ground truth prompt.

        Ground Truth Prompt: "{prompt_text}"

        Read the text in the image and score how accurately it matches the ground truth prompt from 1 (completely wrong) to 10 (perfect match). Focus only on the content and spelling of the text, not its style or legibility.

        Please provide your response in the following format ONLY:
        Score: [Your score from 1-10]
        Justification: [A brief, one-sentence explanation, stating what text you read in the image.]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
                        ],
                    }
                ],
                max_tokens=100,
            )
            content = response.choices[0].message.content
            score = self._parse_score(content)
            return score, content
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return None, str(e)


def main():
    """
    Example usage of the VLMEvaluator.
    You MUST set the OPENAI_API_KEY environment variable to run this.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("="*50)
        print("!!! ERROR: OPENAI_API_KEY environment variable is not set. !!!")
        print("Please set it to your OpenAI API key to run this evaluation.")
        print("Example: export OPENAI_API_KEY='sk-...'")
        print("="*50)
        return
        
    # --- IMPORTANT ---
    # Change these paths to your local paths
    generated_image_path = "/path/to/your/generated/images/result_0_test1_The_text_is_Calligrapher_12345678.png"
    benchmark_dir = "/Users/yanzexuan/code/dataset/Calligrapher_bench_testing"
    # -----------------
    
    if not os.path.exists(generated_image_path) or not os.path.exists(benchmark_dir):
        print("="*50)
        print("!!! PLEASE UPDATE THE PLACEHOLDER PATHS IN `eval_vlm.py` main function !!!")
        print(f"File path checked: {generated_image_path}")
        print("="*50)
        return

    generated_img, _, _, ref_img, metadata = load_images_for_evaluation(generated_image_path, benchmark_dir)

    if generated_img is None:
        print("Failed to load images.")
        return

    print(f"Evaluating file: {os.path.basename(generated_image_path)}")
    
    evaluator = VLMEvaluator()

    # --- Aesthetic Score ---
    print("\n--- Evaluating Aesthetic Score ---")
    aesthetic_score, aesthetic_justification = evaluator.evaluate_aesthetic(generated_img, ref_img)
    print(f"Score: {aesthetic_score}/10")
    print(f"Justification: {aesthetic_justification.split('Justification:')[1].strip()}")
    
    # --- Text-Image Match Score ---
    print("\n--- Evaluating Text-Image Match Score ---")
    match_score, match_justification = evaluator.evaluate_text_image_match(generated_img, metadata['prompt'])
    print(f"Score: {match_score}/10")
    print(f"Justification: {match_justification.split('Justification:')[1].strip()}")


if __name__ == '__main__':
    main()
