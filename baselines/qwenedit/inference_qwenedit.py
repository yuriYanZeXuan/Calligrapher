import os
import torch
import argparse
from PIL import Image
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image

class QwenEditGenerator:
    def __init__(self, model_path="Qwen/Qwen-Image-Edit", device="cuda"):
        print("Initializing Qwen-Image-Edit pipeline...")
        self.device = device
        self.pipe = QwenImageEditPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        self.pipe.to(device)
        print("Qwen-Image-Edit pipeline initialized.")

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        seed: int = 42,
        num_inference_steps: int = 50,
        output_path: str = "output/qwenedit/output.png"
    ):
        """
        Generates an edited image based on a prompt and an input image.
        """
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # The QwenImageEditPipeline takes the image and prompt directly
        edited_image = self.pipe(
            image=image,
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        edited_image.save(output_path)
        print(f"Image saved to {output_path}")
        return edited_image

def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit Inference Script")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit", help="Path to the Qwen-Image-Edit model.")
    parser.add_argument("--prompt", type=str, required=True, help="The editing instruction.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the source image.")
    parser.add_argument("--output_path", type=str, default="output/qwenedit.png", help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    args = parser.parse_args()

    generator = QwenEditGenerator(model_path=args.model_path)
    
    input_image = load_image(args.image_path).convert("RGB")

    generator.generate(
        prompt=args.prompt,
        image=input_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
