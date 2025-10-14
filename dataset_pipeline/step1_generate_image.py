import os
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import config

def generate_image(prompt: str, output_path: str, service: str = "remote") -> bool:
    """
    Generates an image based on a prompt using a T2I model and saves it.

    Args:
        prompt (str): The text prompt for image generation.
        output_path (str): The path to save the generated image.
        service (str): The service to use ('local' or 'remote').

    Returns:
        bool: True if the image was generated and saved successfully, False otherwise.
    """
    try:
        service_config = config.SERVICES[service]
        
        # Determine the correct base URL
        if service == "local":
            base_url = service_config["api_base_url_image"]
        else: # remote
            base_url = service_config["api_base_url"]

        client = OpenAI(
            base_url=base_url,
            api_key=service_config["api_key"]
        )
        # HACK: The openai library might incorrectly use "/generations" plural.
        # We manually patch the resource path to ensure it's correct.
        client.images.with_raw_response.generate = client.images.with_raw_response.generate.__class__(
            client.images.with_raw_response.generate,
            resource_cls=client.images.with_raw_response.generate.resource_cls,
            client=client.images.with_raw_response.generate.client,
            options={'path': '/images/generate'},
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
