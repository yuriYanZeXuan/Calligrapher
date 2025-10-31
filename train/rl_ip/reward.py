import requests
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardClient:
    def __init__(self, server_urls: List[str], ocr_weight: float, vlm_weight: float):
        if not server_urls:
            raise ValueError("Reward server URL list cannot be empty.")
        self.server_urls = server_urls
        self.ocr_weight = ocr_weight
        self.vlm_weight = vlm_weight
        self.executor = ThreadPoolExecutor(max_workers=len(server_urls))
        # Start at a random server index to better distribute load when multiple clients start
        self.next_server_idx = random.randint(0, len(self.server_urls) - 1) if self.server_urls else 0
        logger.info(f"RewardClient initialized for {len(server_urls)} server(s). OCR weight: {ocr_weight}, VLM weight: {vlm_weight}")

    def get_reward_single(
        self,
        server_url: str,
        image: Image.Image,
        prompt: str,
        mask: Optional[Image.Image] = None,
        timestep: Optional[str] = None,
    ) -> dict:
        """Sends a single request to a specific server."""
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            payload = {"image": img_str, "prompt": prompt}
            if mask is not None:
                mask_buffered = BytesIO()
                mask.convert("L").save(mask_buffered, format="PNG")
                payload["mask"] = base64.b64encode(mask_buffered.getvalue()).decode("utf-8")
            if timestep is not None:
                payload["timestep"] = timestep
            
            response = requests.post(server_url, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list):
                data = data[0] if data else {}
            
            vlm_score = float(data.get('vlm_score', 0.0))
            ocr_confidence = float(data.get('ocr_confidence', 0.0))
            
            # Combine scores using the specified weights
            combined_score = (vlm_score * self.vlm_weight) + (ocr_confidence * self.ocr_weight)
            
            return {
                'vlm_score': vlm_score,
                'ocr_confidence': ocr_confidence,
                'combined_score': combined_score
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get reward from server {server_url}: {e}")
            return {'vlm_score': 0.0, 'ocr_confidence': 0.0, 'combined_score': 0.0}
        except Exception as e:
            logger.error(f"An unexpected error occurred with server {server_url}: {e}")
            return {'vlm_score': 0.0, 'ocr_confidence': 0.0, 'combined_score': 0.0}

    def get_rewards_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        masks: Optional[List[Optional[Image.Image]]] = None,
        timesteps: Optional[List[Optional[str]]] = None,
    ) -> List[dict]:
        """
        Sends a batch of images and prompts to the reward servers in parallel
        using a round-robin distribution.
        """
        futures = []
        # The loop iterates through each image/prompt pair in the batch.
        if masks is None:
            masks = [None] * len(images)

        if timesteps is None:
            timesteps = [None] * len(images)

        for image, prompt, mask, timestep in zip(images, prompts, masks, timesteps):
            # Distribute requests to servers in a round-robin fashion.
            # We use a stateful index `self.next_server_idx` to ensure that
            # requests are distributed across *multiple calls* to this method,
            # which is crucial when the batch size is small.
            server_url = self.server_urls[self.next_server_idx]
            self.next_server_idx = (self.next_server_idx + 1) % len(self.server_urls)
            
            futures.append(
                self.executor.submit(
                    self.get_reward_single,
                    server_url,
                    image,
                    prompt,
                    mask,
                    timestep,
                )
            )
        
        results = [future.result() for future in futures]
        return results

# Example usage
if __name__ == '__main__':
    try:
        from PIL import Image, ImageDraw

        # Example with multiple servers
        server_urls = ["http://127.0.0.1:8000/score", "http://127.0.0.1:8001/score"]
        client = RewardClient(server_urls=server_urls, ocr_weight=0.7, vlm_weight=0.3)

        # Create dummy images and prompts
        images_to_score = []
        prompts_to_score = []
        for i in range(4): # Test with more images than servers
            img = Image.new('RGB', (100, 50), color='blue')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Test {i+1}", fill="white")
            images_to_score.append(img)
            prompts_to_score.append(f"a test image number {i+1}")
        
        # Get rewards for the batch
        reward_data_batch = client.get_rewards_batch(images_to_score, prompts_to_score)
        
        for i, data in enumerate(reward_data_batch):
            print(f"Image {i+1} -> Received reward data: {data}")

    except ImportError:
        print("Pillow is not installed. Please install it to run the example.")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection to a reward server failed. Are servers running? Details: {e}")
