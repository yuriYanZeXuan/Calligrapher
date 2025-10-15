from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

class OCRScorer:
    def __init__(self, device=None):
        # Initialize PaddleOCR based on the reference script to disable unnecessary modules.
        # 'ch' lang model supports both Chinese and English.
        self.ocr_model = PaddleOCR(
            use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
            use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
            use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
        )
        print("Initialized PaddleOCR Model.")

    def score(self, image_pil: Image.Image) -> tuple[str, float]:
        """
        Performs OCR on a PIL image using the .predict() method
        and returns the recognized text and average confidence score.
        """
        # Convert PIL image to numpy array for PaddleOCR
        image_np = np.array(image_pil.convert('RGB'))
        
        try:
            # Use the predict method as specified in the reference
            result = self.ocr_model.predict(input=image_np)
            
            # The result is a list containing one Result object
            if not result or not result[0]:
                return "", 0.0
            
            json_result = result[0].json
            
            # The actual data is nested inside the 'res' key
            ocr_data = json_result.get('res', {})
            
            # Extract recognized texts and their scores as per the documentation
            texts = ocr_data.get('rec_texts', [])
            confidences = ocr_data.get('rec_scores', [])
            
            if texts and confidences:
                full_text = " ".join(texts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                return full_text, float(avg_confidence)
            else:
                return "", 0.0
        except Exception as e:
            print(f"An error occurred during PaddleOCR prediction: {e}")
            return "", 0.0

if __name__ == '__main__':
    # Example usage:
    # Create a dummy black image with white text
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (200, 50), color = 'black')
        d = ImageDraw.Draw(img)
        try:
            # Try to load a common font
            font = ImageFont.truetype("Arial.ttf", 20)
        except IOError:
            # Use a default font if Arial is not available
            font = ImageFont.load_default()
        d.text((10,10), "Hello World", fill='white', font=font)
        
        scorer = OCRScorer()
        text, confidence = scorer.score(img)
        print(f"Recognized Text: '{text}', Confidence: {confidence:.4f}")

    except ImportError:
        print("PIL/Pillow is required for the example. Please install it.")
    except Exception as e:
        print(f"An error occurred during the example: {e}")
        print("This might be due to missing fonts or other system dependencies for PIL.")
