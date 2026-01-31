import os

# CRITICAL: Must set BEFORE importing torch/diffusers to avoid "duplicate template name" error
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch
import argparse
from PIL import Image

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from diffusers import ZImagePipeline
class ZImageGenerator:
    def __init__(self, model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image", device="cuda"):
        print("Initializing Z-Image pipeline...")
        self.device = device
        
        self.pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.pipe.to(device)
        
        # Optional: Set attention backend if needed, but keeping default for now or user can modify.
        # self.pipe.transformer.set_attention_backend("flash")
        
        print(f"Z-Image pipeline initialized on {device}.")

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        output_path: str = "output/z_image/output.png"
    ):
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Z-Image specific parameters from user snippet
        # num_inference_steps=9 results in 8 DiT forwards
        # guidance_scale=0.0 for Turbo models
        
        print(f"Generating with Z-Image: {width}x{height}, steps={num_inference_steps}, cfg={guidance_scale}")
        
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        result.save(output_path)
        print(f"Image saved to {output_path}")
        return result

def main():
    parser = argparse.ArgumentParser(description="Z-Image Generation Script")
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image",
                       help="Path to the Z-Image model.")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="The prompt describing the image to generate.")
    parser.add_argument("--output_path", type=str, default="output/z_image.png", 
                       help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=9, help="Number of inference steps (default 9).")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale (default 0.0).")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    
    args = parser.parse_args()

    generator = ZImageGenerator(model_path=args.model_path)
    
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
