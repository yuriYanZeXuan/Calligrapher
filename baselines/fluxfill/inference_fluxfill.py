import os
import torch
import argparse
from PIL import Image
from diffusers.utils import load_image
from diffusers import FluxFillPipeline

class FluxFillGenerator:
    def __init__(self, model_path="black-forest-labs/FLUX.1-Fill-dev", device="cuda"):
        print("Initializing Flux-Fill pipeline...")
        self.device = device
        self.pipe = FluxFillPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
        print("Flux-Fill pipeline initialized.")

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: str = "output/fluxfill/output.png"
    ):
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        width, height = image.size
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        
        image = image.resize((new_width, new_height))
        mask_image = mask_image.resize((new_width, new_height))
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # The FluxFillPipeline uses a prompt and a prompt_2. For general inpainting,
        # we can often use the same prompt for both.
        result = self.pipe(
            prompt=prompt,
            prompt_2=prompt,
            image=image,
            mask_image=mask_image,
            height=new_height,
            width=new_width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]
        
        result.save(output_path)
        print(f"Image saved to {output_path}")
        return result

def main():
    parser = argparse.ArgumentParser(description="Flux-Fill Inpainting Script")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev", help="Path to the Flux-Fill model.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt describing what to inpaint.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the source image.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image.")
    parser.add_argument("--output_path", type=str, default="output/fluxfill.png", help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=30.0, help="Guidance scale.")
    args = parser.parse_args()

    generator = FluxFillGenerator(model_path=args.model_path)
    
    input_image = load_image(args.image_path).convert("RGB")
    mask_image = load_image(args.mask_path).convert("RGB")

    generator.generate(
        prompt=args.prompt,
        image=input_image,
        mask_image=mask_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
