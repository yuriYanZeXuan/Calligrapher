import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class QwenVLScorer:
    def __init__(self, model_path="Qwen/Qwen-VL-Chat", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True
        ).eval()
        print("Initialized Qwen-VL-Chat Model.")

    def score(self, image_pil: Image.Image, prompt: str) -> float:
        """
        Calculates a score based on how well the Qwen-VL model's response to the image
        matches the given prompt. This is a proxy for image-text alignment.
        """
        
        # This is a simplified scoring logic. A more sophisticated approach might be needed.
        # The idea is to ask the model a question about the text in the image and check its response.
        question = f"What text is written in the image? Answer with only the text content."
        
        query = self.tokenizer.from_list_format([
            {'image': image_pil},
            {'text': question},
        ])

        with torch.no_grad():
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)

        # Extract ground truth from the original prompt (e.g., "The text is 'hello'")
        try:
            ground_truth_text = prompt.split("'")[1].lower().strip()
        except IndexError:
            return 0.0 # Cannot score if the prompt format is wrong

        # Simple similarity score (e.g., Jaccard similarity or simple match)
        response_text = response.lower().strip()
        
        # For simplicity, we check for containment. Better metrics could be used.
        if ground_truth_text in response_text:
            return 1.0
        else:
            # You could use fuzzy string matching here for a more nuanced score.
            return 0.0

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
