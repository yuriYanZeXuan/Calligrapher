import torch
from torchvision.transforms.functional import to_pil_image
from .ocr import OCRScorer
from .qwenvl import QwenVLScorer

class RewardCalculator:
    def __init__(self, device, ocr_weight=0.6, vlm_weight=0.4):
        self.device = device
        self.ocr_scorer = OCRScorer(device)
        self.vlm_scorer = QwenVLScorer(device=device)
        self.ocr_weight = ocr_weight
        self.vlm_weight = vlm_weight

    def get_reward(self, generated_images_tensor: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """
        Calculates rewards for a batch of generated images.

        Args:
            generated_images_tensor (torch.Tensor): A batch of generated images (B, C, H, W), normalized to [-1, 1].
            prompts (List[str]): A list of prompts corresponding to the images.

        Returns:
            torch.Tensor: A tensor of rewards for each image in the batch.
        """
        rewards = []
        for i, image_tensor in enumerate(generated_images_tensor):
            # Denormalize tensor from [-1, 1] to [0, 1] range for PIL conversion
            image_pil = to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
            
            prompt_text = prompts[i]
            
            try:
                ground_truth_text = prompt_text.split("'")[1]
            except IndexError:
                ground_truth_text = ""

            # 1. OCR Reward (Accuracy + Confidence)
            recognized_text, ocr_confidence = self.ocr_scorer.score(image_pil)
            
            # Use a simple string similarity metric (e.g., normalized Levenshtein distance)
            # For simplicity, we'll use exact match for now.
            ocr_accuracy = 1.0 if recognized_text.strip().lower() == ground_truth_text.lower() else 0.0
            ocr_reward = ocr_accuracy * ocr_confidence

            # 2. VLM Reward
            vlm_score = self.vlm_scorer.score(image_pil, prompt_text)
            
            # Combine rewards
            total_reward = (self.ocr_weight * ocr_reward) + (self.vlm_weight * vlm_score)
            rewards.append(total_reward)

        return torch.tensor(rewards, device=self.device, dtype=torch.float32)

def compute_rewards(
    generated_images,
    prompts,
    reward_calculator,
):
    """
    A wrapper function to compute rewards for a batch.
    """
    with torch.no_grad():
        rewards = reward_calculator.get_reward(generated_images, prompts)
    return rewards
