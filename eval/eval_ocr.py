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
        # Aligned with step3_ocr.py, use_textline_orientation is the modern param.
        # This resolves the 'use_angle_cls' deprecation warning.
        self.ocr = PaddleOCR(use_textline_orientation=False, lang=lang)
        print("PaddleOCR initialized.")

    def calculate_ocr_accuracy(self, image: Image.Image, ground_truth_text: str, mask: Image.Image = None):
        """
        Calculates character-level OCR accuracy based on concatenated text.
        If a mask is provided, the OCR will be performed only on the masked region.
        It compares the recognized text (with all parts concatenated and no spaces)
        against the ground truth text (with spaces removed). This focuses on
        character correctness, ignoring spacing.
        Accuracy = 1 - (Levenshtein_Distance / length_of_ground_truth_no_spaces)
        """
        # Prepare ground truth: remove spaces and convert to lower case for consistent comparison
        ground_truth_processed = ground_truth_text.replace(" ", "").lower()

        # Apply mask if provided: black out the unmasked area
        if mask:
            # Resize mask to match image and ensure it's a binary mask
            mask_resized = mask.resize(image.size, Image.NEAREST)
            img_np = np.array(image.convert('RGB'))
            mask_np = np.array(mask_resized.convert('L'))
            
            # Black out the unmasked area
            img_np[mask_np == 0] = 0
            image_to_ocr = Image.fromarray(img_np)
        else:
            image_to_ocr = image

        # Convert PIL Image (RGB) to numpy array and then to BGR for PaddleOCR predict method
        img_np_rgb = np.array(image_to_ocr.convert('RGB'))
        img_np_bgr = img_np_rgb[:, :, ::-1]

        # Use the 'predict' method, which is the modern standard and aligns with step3_ocr.py
        result = self.ocr.predict(img_np_bgr)

        recognized_text_parts = []
        # The result of predict is a list containing one result object for the image
        if result and result[0] and hasattr(result[0], 'json'):
            json_res = result[0].json
            if json_res and 'res' in json_res and json_res['res']:
                res_data = json_res['res']
                # rec_texts contains the list of recognized text strings
                recognized_text_parts = res_data.get('rec_texts', [])

        # --- DEBUG: Print raw and processed text ---
        print(f"  - Recognized Raw: {recognized_text_parts}")
        
        # Concatenate all parts and process for comparison
        recognized_text_processed = "".join(recognized_text_parts).replace(" ", "").lower()
        print(f"  - Ground Truth (Processed) : '{ground_truth_processed}'")
        print(f"  - Recognized Text (Processed): '{recognized_text_processed}'")
        # --- END DEBUG ---

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
    generated_image_path = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/cli_exps/2025-10-10-17-12-59_self/result_40_test1_The_text_is_BRAVE._801648887.png"
    benchmark_dir = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing"
    # -----------------
    
    if not os.path.exists(generated_image_path) or not os.path.exists(benchmark_dir):
        print("="*50)
        print("!!! PLEASE UPDATE THE PLACEHOLDER PATHS IN `eval_ocr.py` main function !!!")
        print(f"File path checked: {generated_image_path}")
        print("="*50)
        return

    # Load images and metadata
    generated_img, _, mask_img, _, metadata = load_images_for_evaluation(generated_image_path, benchmark_dir)

    if generated_img is None:
        print("Failed to load images.")
        return

    ground_truth = metadata['prompt']
    print(f"Evaluating file: {os.path.basename(generated_image_path)}")
    print(f"Ground Truth Text: '{ground_truth}'")

    # Evaluate
    evaluator = OCREvaluator()
    # Pass the mask to the evaluator
    accuracy = evaluator.calculate_ocr_accuracy(generated_img, ground_truth, mask=mask_img)

    print(f"\nOCR Character Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
