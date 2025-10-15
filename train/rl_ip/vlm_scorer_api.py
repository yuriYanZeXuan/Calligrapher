import aiohttp
import asyncio
import base64
import io
from typing import List
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class VLMScorerAPI:
    def __init__(self, api_url: str, session: aiohttp.ClientSession):
        self.api_url = api_url
        self.session = session

    async def _score_single(self, image_pil: Image.Image, prompt: str) -> float:
        """Helper to score a single image asynchronously."""
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        payload = {"image": img_str, "prompt": prompt}
        
        try:
            async with self.session.post(self.api_url, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("score", 0.0)
                else:
                    error_text = await response.text()
                    logger.error(f"Error from VLM API: {response.status} - {error_text}")
                    return 0.0
        except aiohttp.ClientError as e:
            logger.error(f"AIOHTTP client error calling VLM API: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"An unexpected error occurred when calling VLM API: {e}")
            return 0.0

    async def score_batch(self, images_pil: List[Image.Image], prompts: List[str]) -> List[float]:
        """Scores a batch of images and prompts concurrently."""
        if not images_pil:
            return []
            
        tasks = [
            self._score_single(image, prompt) 
            for image, prompt in zip(images_pil, prompts)
        ]
        
        scores = await asyncio.gather(*tasks)
        return scores

async def main():
    # Example usage
    api_url = "http://localhost:8000/score" # Change if your server is elsewhere
    
    # Create a dummy image for testing
    try:
        dummy_image = Image.new('RGB', (100, 100), color = 'red')
        prompts = ["a red square", "another red square"]
        images = [dummy_image, dummy_image]

        async with aiohttp.ClientSession() as session:
            scorer = VLMScorerAPI(api_url=api_url, session=session)
            scores = await scorer.score_batch(images, prompts)
            print(f"Received scores: {scores}")

    except aiohttp.ClientConnectorError as e:
        print(f"Connection error: Could not connect to the reward server at {api_url}.")
        print("Please ensure the reward server is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # This can be run to test the client against a running server
    asyncio.run(main())
