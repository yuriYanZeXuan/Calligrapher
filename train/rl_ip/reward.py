import torch
from torchvision.transforms.functional import to_pil_image
from .ocr import OCRScorer
from .vlm_scorer_api import VLMScorerAPI

class RewardCalculator:
    def __init__(self, device, ocr_weight=0.6, vlm_weight=0.4, vlm_scorer_api: VLMScorerAPI = None):
        self.device = device
        self.ocr_scorer = OCRScorer(device)
        self.vlm_scorer_api = vlm_scorer_api
        self.ocr_weight = ocr_weight
        self.vlm_weight = vlm_weight

    async def get_reward_async(self, generated_images_tensor: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """
        Calculates rewards for a batch of generated images asynchronously.

        Args:
            generated_images_tensor (torch.Tensor): A batch of generated images (B, C, H, W), normalized to [-1, 1].
            prompts (List[str]): A list of prompts corresponding to the images.

        Returns:
            torch.Tensor: A tensor of rewards for each image in the batch.
        """
        # --- 1. VLM Score (Async) ---
        # Convert tensors to PIL images for the API
        images_pil = [to_pil_image(t.add(1).div(2).clamp(0, 1)) for t in generated_images_tensor]
        
        # Get VLM scores concurrently
        vlm_scores = await self.vlm_scorer_api.score_batch(images_pil, prompts)
        
        # --- 2. OCR Score (Sync) ---
        # This part remains synchronous as it's a local, fast computation.
        ocr_rewards = []
        for i, image_pil in enumerate(images_pil):
            prompt_text = prompts[i]
            try:
                ground_truth_text = prompt_text.split("'")[1]
            except IndexError:
                ground_truth_text = ""

            recognized_text, ocr_confidence = self.ocr_scorer.score(image_pil)
            ocr_accuracy = 1.0 if recognized_text.strip().lower() == ground_truth_text.lower() else 0.0
            ocr_reward = ocr_accuracy * ocr_confidence
            ocr_rewards.append(ocr_reward)

        # --- 3. Combine Rewards ---
        total_rewards = []
        for ocr_r, vlm_s in zip(ocr_rewards, vlm_scores):
            total_reward = (self.ocr_weight * ocr_r) + (self.vlm_weight * vlm_s)
            total_rewards.append(total_reward)

        return torch.tensor(total_rewards, device=self.device, dtype=torch.float32)

# The synchronous version might still be useful for testing or non-RL modes.
# For simplicity, we'll remove it. If needed, it can be added back.
# def compute_rewards(...)

# Keeping a wrapper might still be useful, but it needs to be async.
async def compute_rewards_async(
    generated_images,
    prompts,
    reward_calculator,
):
    """
    An async wrapper function to compute rewards for a batch.
    """
    # The `no_grad` is less critical here since the main computation is an API call,
    # but good practice for the OCR part.
    with torch.no_grad():
        rewards = await reward_calculator.get_reward_async(generated_images, prompts)
    return rewards
