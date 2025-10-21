import os
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import config

def generate_image(prompt: str, output_path: str, service: str = "remote", image_port: int = None) -> bool:
    """
    Generates an image based on a prompt using a T2I model and saves it.

    Args:
        prompt (str): The text prompt for image generation.
        output_path (str): The path to save the generated image.
        service (str): The service to use ('local' or 'remote').
        image_port (int): The specific port for the image service (for local).

    Returns:
        bool: True if the image was generated and saved successfully, False otherwise.
    """
    try:
        service_config = config.SERVICES[service]
        
        # Determine the correct base URL
        if service == "local":
            if image_port is None:
                raise ValueError("image_port must be provided for local service.")
            base_url = f"http://127.0.0.1:{image_port}/v1"
        else: # remote
            base_url = service_config["api_base_url"]

        client = OpenAI(
            base_url=base_url,
            api_key=service_config["api_key"]
        )

        print(f"Generating image with prompt: '{prompt}' using '{service}' service at {base_url}.")
        response = client.images.generate(
            model=service_config["image_gen_model"],
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
