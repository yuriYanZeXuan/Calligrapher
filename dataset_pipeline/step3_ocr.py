import os
from paddleocr import PaddleOCR
from typing import List, Tuple
import config

# Global variable to hold the initialized reader
_paddle_reader = None

def get_paddle_reader():
    """Initializes and returns the PaddleOCR reader (singleton)."""
    global _paddle_reader
    if _paddle_reader is None:
        print("Initializing PaddleOCR reader for the first time...")
        # Map languages for PaddleOCR. 'ch' model supports both Chinese and English.
        # paddle_lang = 'ch' if 'ch_sim' in config.OCR_LANGUAGES else 'en'
        _paddle_reader = PaddleOCR(
            use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
            use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
            use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
        )
        print("PaddleOCR reader initialized successfully.")
    return _paddle_reader

def ocr_image_paddle(image_path: str) -> List[Tuple[List[int], str]]:
    """
    Performs OCR on an image using PaddleOCR to find text and bounding boxes.
    """
    try:
        paddle_reader = get_paddle_reader()
    except Exception:
        # If get_paddle_reader fails, we return an empty list to indicate failure.
        return []
        
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return []

    try:
        print(f"Performing OCR with PaddleOCR on {image_path}")
        # Use the predict method as requested
        result = paddle_reader.predict(input=image_path)
        json_result = result[0].json
        
        # The result structure is a dict with 'dt_polys' (boxes) and 'res' (texts).
        boxes = json_result.get('dt_polys', [])
        texts = json_result.get('res', [])

        ocr_results = []
        for box, text in zip(boxes, texts):
            # box is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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
    example_image_path = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/dataset_pipeline/Calligrapher_bench_custom/test1_ref.png"
    if os.path.exists(example_image_path):
        print("\n--- Testing EasyOCR ---")
        # The original ocr_image function is removed, so this part will be removed.
        # results = ocr_image(example_image_path)
        # if results:
        #     for bbox, text in results:
        #         print(f"Text: '{text}', BBox: {bbox}")

        print("\n--- Testing PaddleOCR ---")
        paddle_results = ocr_image_paddle(example_image_path)
        if paddle_results:
            for bbox, text in paddle_results:
                print(f"Text: '{text}', BBox: {bbox}")
    else:
        print(f"Example image not found at '{example_image_path}'. Please run step1_generate_image.py first.")
