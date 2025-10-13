from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

class OCRScorer:
    def __init__(self, device=None):
        # PaddleOCR uses CPU or GPU, device parameter is for compatibility
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        print("Initialized PaddleOCR Model.")

    def score(self, image_pil: Image.Image) -> tuple[str, float]:
        """
        Performs OCR on a PIL image and returns the recognized text and confidence score.
        """
        # Convert PIL image to numpy array for PaddleOCR
        image_np = np.array(image_pil.convert('RGB'))
        
        result = self.ocr_model.ocr(image_np, cls=True)
        
        if result and result[0] is not None:
            # result is a list of lists, where each inner list contains [bbox, (text, confidence)]
            recognized_texts = [res[1][0] for res in result[0]]
            confidences = [res[1][1] for res in result[0]]
            
            full_text = " ".join(recognized_texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return full_text, float(avg_confidence)
        else:
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
