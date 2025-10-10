from paddleocr import PaddleOCR
import Levenshtein
from PIL import Image
import os
import sys
import numpy as np

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import load_images_for_evaluation


class OCREvaluator:
    def __init__(self, lang='en'):
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        print("PaddleOCR initialized.")

    def calculate_ocr_accuracy(self, image: Image.Image, ground_truth_text: str):
        """
        Calculates character-level OCR accuracy based on concatenated text.
        It compares the recognized text (with all parts concatenated and no spaces)
        against the ground truth text (with spaces removed). This focuses on
        character correctness, ignoring spacing.
        Accuracy = 1 - (Levenshtein_Distance / length_of_ground_truth_no_spaces)
        """
        # Prepare ground truth: remove spaces and convert to lower case for consistent comparison
        ground_truth_processed = ground_truth_text.replace(" ", "").lower()

        # Convert PIL Image to numpy array for PaddleOCR
        img_np = np.array(image.convert('RGB'))

        # Ocr the image
        result = self.ocr.ocr(img_np, cls=True)

        recognized_text_parts = []
        if result and result[0]:
            # Get all recognized text parts
            recognized_text_parts = [line[1][0] for line in result[0]]
        
        # Concatenate all parts and process for comparison
        recognized_text_processed = "".join(recognized_text_parts).replace(" ", "").lower()

        # Handle cases where ground truth is empty (e.g., prompt was just spaces)
        if not ground_truth_processed:
            return 1.0 if not recognized_text_processed else 0.0
        
        # If ground truth is not empty, but we recognized nothing, accuracy is 0
        if not recognized_text_processed:
            return 0.0

        distance = Levenshtein.distance(ground_truth_processed, recognized_text_processed)
        accuracy = 1 - (distance / len(ground_truth_processed))

        return max(0.0, accuracy)  # Ensure accuracy is not negative

def main():
    """
    Example usage of the OCREvaluator.
    You need to provide paths to a generated image and the benchmark dataset directory.
    """
    # --- IMPORTANT ---
    # Change these paths to your local paths
    generated_image_path = "/path/to/your/generated/images/result_0_test1_The_text_is_Calligrapher_12345678.png"
    benchmark_dir = "/Users/yanzexuan/code/dataset/Calligrapher_bench_testing"
    # -----------------
    
    if not os.path.exists(generated_image_path) or not os.path.exists(benchmark_dir):
        print("="*50)
        print("!!! PLEASE UPDATE THE PLACEHOLDER PATHS IN `eval_ocr.py` main function !!!")
        print(f"File path checked: {generated_image_path}")
        print("="*50)
        return

    # Load images and metadata
    generated_img, _, _, _, metadata = load_images_for_evaluation(generated_image_path, benchmark_dir)

    if generated_img is None:
        print("Failed to load images.")
        return

    ground_truth = metadata['prompt']
    print(f"Evaluating file: {os.path.basename(generated_image_path)}")
    print(f"Ground Truth Text: '{ground_truth}'")

    # Evaluate
    evaluator = OCREvaluator()
    accuracy = evaluator.calculate_ocr_accuracy(generated_img, ground_truth)

    print(f"\nOCR Character Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
