import os
import torch
import argparse
from PIL import Image
from diffusers.pipelines.glm_image import GlmImagePipeline

class GlmImageGenerator:
    def __init__(self, model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/glm_image", device="cuda"):
        print("Initializing GLM-Image pipeline...")
        self.device = device
        
        # Using device_map=device if it's a string like "cuda:0", or just loading and moving.
        # The user example used device_map="cuda". 
        # To support parallel execution on specific GPUs, we should pass the specific device.
        
        # If device is a torch.device, convert to string for device_map if needed, 
        # or just use .to() after loading if device_map is not supported or we want manual control.
        # However, large models often use device_map="auto" or specific device to load directly to GPU.
        
        # Let's try loading without device_map and then .to(device) to be safe and consistent with others,
        # unless it's too big. But user used device_map="cuda".
        # If we pass device_map=device (e.g. "cuda:1"), it should work.
        
        device_str = str(device)
        
        try:
            self.pipe = GlmImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_str
            )
        except Exception as e:
            print(f"Failed to load with device_map={device_str}, trying without device_map and .to(). Error: {e}")
            self.pipe = GlmImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
            self.pipe.to(device)
            
        print(f"GLM-Image pipeline initialized on {device}.")

    def generate(
        self,
        prompt: str,
        image: Image.Image = None,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        height: int = 1024,
        width: int = 1024,
        output_path: str = "output/glm_image/output.png"
    ):
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure dimensions are multiples of 32
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        kwargs = {
            "prompt": prompt,
            "height": new_height,
            "width": new_width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if image is not None:
            # Image editing mode
            # User snippet: image=[image]
            kwargs["image"] = [image]
            print(f"Running in image editing mode with input image size: {image.size}")
        else:
            print(f"Running in text-to-image mode with size: {new_width}x{new_height}")

        result = self.pipe(**kwargs).images[0]
        
        result.save(output_path)
        print(f"Image saved to {output_path}")
        return result

def main():
    parser = argparse.ArgumentParser(description="GLM-Image Generation Script")
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/glm_image",
                       help="Path to the GLM-Image model.")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="The prompt describing the image to generate.")
    parser.add_argument("--image_path", type=str, default=None,
                       help="Path to input image for editing.")
    parser.add_argument("--output_path", type=str, default="output/glm_image.png", 
                       help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="Guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    
    args = parser.parse_args()

    generator = GlmImageGenerator(model_path=args.model_path)
    
    input_image = None
    if args.image_path:
        input_image = Image.open(args.image_path).convert("RGB")
    
    generator.generate(
        prompt=args.prompt,
        image=input_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
