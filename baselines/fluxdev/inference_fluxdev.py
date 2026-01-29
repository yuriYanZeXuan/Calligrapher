import os
import torch
import argparse
from PIL import Image
from diffusers import FluxPipeline

class FluxDevGenerator:
    def __init__(self, model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/FLUX.1-dev", device="cuda"):
        print("Initializing Flux-Dev pipeline...")
        self.device = device
        self.pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
        print("Flux-Dev pipeline initialized.")

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        output_path: str = "output/fluxdev/output.png"
    ):
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure dimensions are multiples of 32
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipe(
            prompt=prompt,
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
    parser = argparse.ArgumentParser(description="Flux-Dev Text-to-Image Generation Script")
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/FLUX.1-dev",
                       help="Path to the Flux-Dev model.")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="The prompt describing the image to generate.")
    parser.add_argument("--output_path", type=str, default="output/fluxdev.png", 
                       help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    args = parser.parse_args()

    generator = FluxDevGenerator(model_path=args.model_path)
    
    generator.generate(
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
