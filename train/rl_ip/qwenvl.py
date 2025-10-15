import torch
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List

def extract_score(output_text: str) -> float:
    match = re.search(r'<Score>(\d+)</Score>', output_text)
    if match:
        # Normalize score to be between 0 and 1
        return float(match.group(1)) / 5.0
    return 0.0

class QwenVLScorer:
    def __init__(self, model_path: str, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True
        ).eval()
        self.task = '''
Your role is to evaluate the aesthetic quality score of given images.
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible
subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward
composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but
lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with
a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful
composition with emotional impact.

Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. Then, give a final score from 1 to 5 within the <Score> tag.
<Thought>
[Analyze the evaluation process in detail here]
</Thought>
<Score>X</Score>
'''
        print(f"Initialized Qwen-VL Model from: {model_path}")

    def score(self, image_pil: Image.Image, prompt: str) -> float:
        """
        Calculates a score based on the aesthetic quality of the image.
        The prompt is not directly used for scoring but is kept for API consistency.
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": self.task},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image_pil], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            response_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

        score = extract_score(response)
        return score

if __name__ == '__main__':
    # This example is difficult to run standalone due to model dependencies
    # and a dummy image won't work well. The following is conceptual.
    print("QwenVLScorer example usage (conceptual):")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # scorer = QwenVLScorer(device=device)
    # 
    # from PIL import Image
    # # Load a real image with text
    # try:
    #     image = Image.open("path/to/your/image_with_text.png")
    #     prompt = "The text is 'Sample Text'"
    #     score = scorer.score(image, prompt)
    #     print(f"VLM Score: {score}")
    # except FileNotFoundError:
    #     print("Please provide a valid image path to run the example.")

    pass
