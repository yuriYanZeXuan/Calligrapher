import os
import torch
import argparse
from PIL import Image
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

class FluxKleinGenerator:
    def __init__(self, model_path="black-forest-labs/FLUX.2-klein-base-9B", device="cuda", enable_cpu_offload=True):
        print("Initializing Flux-Klein pipeline...")
        self.device = device
        self.dtype = torch.bfloat16
        
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype
        )
        
        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("Flux-Klein pipeline initialized with CPU offload.")
        else:
            self.pipe.to(device)
            print(f"Flux-Klein pipeline initialized on {device}.")

    def generate(
        self,
        prompt: str,
        image: Image.Image = None,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        height: int = 1024,
        width: int = 1024,
        output_path: str = "output/fluxklein/output.png"
    ):
        """
        Generate or edit an image using Flux-Klein.
        
        Args:
            prompt: Text description of the image to generate
            image: Optional input image for image-to-image editing. If None, generates from scratch.
            seed: Random seed for reproducibility
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Output image height (only used when image is None)
            width: Output image width (only used when image is None)
            output_path: Path to save the generated image
        
        Returns:
            PIL.Image: Generated image
        """
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare kwargs
        kwargs = {
            "prompt": prompt,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add image for editing mode, or height/width for generation mode
        if image is not None:
            # Image editing mode (image-to-image)
            kwargs["image"] = [image]  # multi-image input format
            print(f"Running in image editing mode with input image size: {image.size}")
        else:
            # Text-to-image generation mode
            new_width = (width // 32) * 32
            new_height = (height // 32) * 32
            kwargs["height"] = new_height
            kwargs["width"] = new_width
            print(f"Running in text-to-image mode with size: {new_width}x{new_height}")
        
        result = self.pipe(**kwargs).images[0]
        
        result.save(output_path)
        print(f"Image saved to {output_path}")
        return result

def main():
    parser = argparse.ArgumentParser(description="Flux-Klein Text-to-Image and Image Editing Script")
    parser.add_argument("--model_path", type=str, 
                       default="black-forest-labs/FLUX.2-klein-base-9B",
                       help="Path to the Flux-Klein model.")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="The prompt describing the image to generate or edit.")
    parser.add_argument("--image_path", type=str, default=None,
                       help="Path to input image for editing. If not provided, generates from scratch.")
    parser.add_argument("--output_path", type=str, default="output/fluxklein.png", 
                       help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Image height (for text-to-image mode).")
    parser.add_argument("--width", type=int, default=1024, help="Image width (for text-to-image mode).")
    parser.add_argument("--enable_cpu_offload", action="store_true", 
                       help="Enable CPU offload (requires more VRAM).")
    args = parser.parse_args()

    generator = FluxKleinGenerator(
        model_path=args.model_path,
        enable_cpu_offload=args.enable_cpu_offload
    )
    
    # Load input image if provided
    input_image = None
    if args.image_path:
        input_image = load_image(args.image_path).convert("RGB")
        print(f"Loaded input image from {args.image_path}")
    
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
