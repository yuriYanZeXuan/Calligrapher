import os
import easyocr
from paddleocr import PaddleOCR
from typing import List, Tuple
import config

# Initialize the reader once
try:
    reader = easyocr.Reader(config.OCR_LANGUAGES)
    print("EasyOCR reader initialized successfully.")
except Exception as e:
    print(f"Error initializing EasyOCR reader: {e}")
    reader = None

try:
    # Map languages for PaddleOCR. 'ch' model supports both Chinese and English.
    paddle_lang = 'ch' if 'ch_sim' in config.OCR_LANGUAGES else 'en'
    paddle_reader = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=paddle_lang
    )
    print("PaddleOCR reader initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR reader: {e}")
    paddle_reader = None


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
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            min_x = int(min(xs))
            min_y = int(min(ys))
            max_x = int(max(xs))
            max_y = int(max(ys))
            
            # Format required by some models might be just top-left and bottom-right points
            # (X1, Y1, X2, Y2)
            formatted_bbox = [min_x, min_y, max_x, max_y]
            ocr_results.append((formatted_bbox, text))
            
        print(f"OCR found {len(ocr_results)} text block(s).")
        return ocr_results
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return []

def ocr_image_paddle(image_path: str) -> List[Tuple[List[int], str]]:
    """
    Performs OCR on an image using PaddleOCR to find text and bounding boxes.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A list of tuples, where each tuple contains the bounding box and the recognized text.
        Returns an empty list if OCR fails or no text is found.
    """
    if not paddle_reader:
        print("PaddleOCR reader is not available.")
        return []
        
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return []

    try:
        print(f"Performing OCR with PaddleOCR on {image_path}")
        result = paddle_reader.predict(input=image_path)
        
        ocr_results = []
        # The result of predict is a list containing one OCRResult object for the image
        if not result or not result[0]:
            print("PaddleOCR found 0 text block(s).")
            return ocr_results

        ocr_result_obj = result[0]
        
        # The OCRResult object contains detected polygons and recognized texts
        # in separate lists. We zip them together.
        boxes = ocr_result_obj.dt_polys
        texts = ocr_result_obj.rec_texts

        for box, text in zip(boxes, texts):
            # box is a list of 4 points
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            min_x, min_y = int(min(xs)), int(min(ys))
            max_x, max_y = int(max(xs)), int(max(ys))
            
            formatted_bbox = [min_x, min_y, max_x, max_y]
            ocr_results.append((formatted_bbox, text))
        
        print(f"PaddleOCR found {len(ocr_results)} text block(s).")
        return ocr_results
    except Exception as e:
        print(f"An error occurred during PaddleOCR: {e}")
        return []

if __name__ == '__main__':
    # Example usage: You need to have an image file for this to work.
    # Let's assume 'generated_dataset/generated_image.png' exists from step1.
    example_image_path = os.path.join(config.OUTPUT_DIR, "generated_image.png")
    if os.path.exists(example_image_path):
        print("\n--- Testing EasyOCR ---")
        results = ocr_image(example_image_path)
        if results:
            for bbox, text in results:
                print(f"Text: '{text}', BBox: {bbox}")

        print("\n--- Testing PaddleOCR ---")
        paddle_results = ocr_image_paddle(example_image_path)
        if paddle_results:
            for bbox, text in paddle_results:
                print(f"Text: '{text}', BBox: {bbox}")
    else:
        print(f"Example image not found at '{example_image_path}'. Please run step1_generate_image.py first.")
