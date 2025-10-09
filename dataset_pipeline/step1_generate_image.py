import os
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import config

client = OpenAI(
    base_url=config.API_BASE_URL,
    api_key=config.API_KEY
)

def generate_image(prompt: str, output_path: str) -> bool:
    """
    Generates an image based on a prompt using a T2I model and saves it.

    Args:
        prompt (str): The text prompt for image generation.
        output_path (str): The path to save the generated image.

    Returns:
        bool: True if the image was generated and saved successfully, False otherwise.
    """
    try:
        print(f"Generating image with prompt: '{prompt}'")
        response = client.images.generate(
            model=config.IMAGE_GEN_MODEL,
            prompt=prompt,
            size=config.IMAGE_SIZE,
            response_format="url", # or "b64_json"
            extra_body={
                "watermark": False,
            },
        )
        
        image_url = response.data[0].url
        
        # Download and save the image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            image = Image.open(BytesIO(image_response.content))
            image.save(output_path)
            print(f"Image saved to {output_path}")
            return True
        else:
            print(f"Error: Could not download image from {image_url}")
            return False

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        return False

if __name__ == '__main__':
    # Example usage
    human_instruction = "A colorful graffiti of the word 'NGANGUR' on a brick wall"
    # In a real scenario, this would be an augmented prompt
    augmented_prompt = human_instruction 
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    image_filename = "generated_image.png"
    image_path = os.path.join(config.OUTPUT_DIR, image_filename)
    
    generate_image(augmented_prompt, image_path)
