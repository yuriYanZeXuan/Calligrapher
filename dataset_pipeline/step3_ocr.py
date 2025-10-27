import os
from paddleocr import PaddleOCR, PaddleOCRVL
from typing import List, Tuple

# --- Configuration for PaddleOCR ---
# Set to True to use PaddleOCRVL, False to use standard PaddleOCR
USE_PADDLE_VL = True
# URL for the deployed PaddleOCRVL backend server
PADDLE_VL_SERVER_URL = "http://127.0.0.1:8118/v1"
# --- End Configuration ---

# Global variable to hold the initialized reader
_paddle_reader = None
_paddle_vl_reader = None

def get_paddle_reader():
    """Initializes and returns the PaddleOCR reader (singleton)."""
    global _paddle_reader
    if _paddle_reader is None:
        print("Initializing PaddleOCR reader for the first time...")
        # Map languages for PaddleOCR. 'ch' model supports both Chinese and English.
        # The line below is commented out as config is no longer imported.
        # paddle_lang = 'ch' if 'ch_sim' in config.OCR_LANGUAGES else 'en'
        _paddle_reader = PaddleOCR(
            use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
            use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
            use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
        )
        print("PaddleOCR reader initialized successfully.")
    return _paddle_reader

def get_paddle_vl_reader():
    """Initializes and returns the PaddleOCRVL reader (singleton)."""
    global _paddle_vl_reader
    if _paddle_vl_reader is None:
        print("Initializing PaddleOCRVL reader for the first time...")
        _paddle_vl_reader = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=PADDLE_VL_SERVER_URL
        )
        print("PaddleOCRVL reader initialized successfully.")
    return _paddle_vl_reader

def _ocr_image_paddle_v5(image_path: str) -> List[Tuple[List[int], str]]:
    """
    Performs OCR on an image using standard PaddleOCR to find text and bounding boxes.
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
        if not result or not result[0]:
            print("PaddleOCR returned no result.")
            return []
        json_result = result[0].json
        
        # The actual results are nested inside the 'res' key
        ocr_data = json_result.get('res', {})
        
        # Extract boxes and texts from the nested dictionary
        boxes = ocr_data.get('dt_polys', [])
        texts = ocr_data.get('rec_texts', [])

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

def _ocr_image_paddle_vl(image_path: str) -> List[Tuple[List[int], str]]:
    """
    Performs OCR on an image using PaddleOCRVL to find text and bounding boxes.
    """
    try:
        paddle_vl_reader = get_paddle_vl_reader()
    except Exception as e:
        print(f"Failed to initialize PaddleOCRVL reader: {e}")
        return []

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return []

    try:
        print(f"Performing OCR with PaddleOCRVL on {image_path}")
        output = paddle_vl_reader.predict(image_path)
        
        ocr_results = []
        if output:
            for res in output:
                json_result = res.json
                
                # The actual result data is nested under the 'res' key.
                if not json_result or 'res' not in json_result:
                    continue

                actual_result = json_result['res']

                if 'parsing_res_list' not in actual_result or not actual_result['parsing_res_list']:
                    continue

                for item in actual_result['parsing_res_list']:
                    text = item.get('block_content')
                    box = item.get('block_bbox')

                    if not text or not box or len(box) != 4:
                        continue
                    
                    try:
                        # block_bbox is [min_x, min_y, max_x, max_y]
                        # Ensure they are integers
                        formatted_bbox = [int(p) for p in box]
                        ocr_results.append((formatted_bbox, text))
                    except (ValueError, TypeError):
                        print(f"Warning: could not parse bbox: {box}")
                        continue

        print(f"PaddleOCRVL found {len(ocr_results)} text block(s).")
        return ocr_results
    except Exception as e:
        print(f"An error occurred during PaddleOCRVL: {e}")
        return []

def ocr_image_paddle(image_path: str) -> List[Tuple[List[int], str]]:
    """
    Performs OCR on an image using the configured PaddleOCR version.
    """
    if USE_PADDLE_VL:
        return _ocr_image_paddle_vl(image_path)
    else:
        return _ocr_image_paddle_v5(image_path)

if __name__ == '__main__':
    # Example usage: You need to have an image file for this to work.
    # Let's assume 'generated_dataset/generated_image.png' exists from step1.
    example_image_path = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/dataset_pipeline/Calligrapher_bench_custom/test1_ref.png"
    if os.path.exists(example_image_path):
        print("\n--- Testing PaddleOCR (version controlled by USE_PADDLE_VL) ---")
        paddle_results = ocr_image_paddle(example_image_path)
        if paddle_results:
            for bbox, text in paddle_results:
                print(f"Text: '{text}', BBox: {bbox}")
    else:
        print(f"Example image not found at '{example_image_path}'. Please run step1_generate_image.py first.")
