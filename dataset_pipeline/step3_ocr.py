import os
import easyocr
from typing import List, Tuple
import config

# Initialize the reader once
try:
    reader = easyocr.Reader(config.OCR_LANGUAGES)
    print("EasyOCR reader initialized successfully.")
except Exception as e:
    print(f"Error initializing EasyOCR reader: {e}")
    reader = None

def ocr_image(image_path: str) -> List[Tuple[List[int], str]]:
    """
    Performs OCR on an image to find text and bounding boxes.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A list of tuples, where each tuple contains the bounding box and the recognized text.
        Returns an empty list if OCR fails or no text is found.
    """
    if not reader:
        print("EasyOCR reader is not available.")
        return []
        
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return []

    try:
        print(f"Performing OCR on {image_path}")
        result = reader.readtext(image_path)
        
        # result is a list of (bbox, text, prob)
        # We only need bbox and text
        ocr_results = []
        for (bbox, text, prob) in result:
            # bbox is a list of 4 points, convert to [min_x, min_y, max_x, max_y]
            (top_left, top_right, bottom_right, bottom_left) = bbox
            min_x = int(min(top_left[0], bottom_left[0]))
            min_y = int(min(top_left[1], top_right[1]))
            max_x = int(max(top_right[0], bottom_right[0]))
            max_y = int(max(bottom_left[1], bottom_right[1]))
            
            # Format required by some models might be just top-left and bottom-right points
            # (X1, Y1, X2, Y2)
            formatted_bbox = [min_x, min_y, max_x, max_y]
            ocr_results.append((formatted_bbox, text))
            
        print(f"OCR found {len(ocr_results)} text block(s).")
        return ocr_results
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return []

if __name__ == '__main__':
    # Example usage: You need to have an image file for this to work.
    # Let's assume 'generated_dataset/generated_image.png' exists from step1.
    example_image_path = os.path.join(config.OUTPUT_DIR, "generated_image.png")
    if os.path.exists(example_image_path):
        results = ocr_image(example_image_path)
        if results:
            for bbox, text in results:
                print(f"Text: '{text}', BBox: {bbox}")
    else:
        print(f"Example image not found at '{example_image_path}'. Please run step1_generate_image.py first.")
