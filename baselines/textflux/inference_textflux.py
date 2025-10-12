import os
import torch
import argparse
from PIL import Image
from diffusers.utils import load_image
from diffusers import FluxFillPipeline, FluxTransformer2DModel

class TextFluxGenerator:
    def __init__(self, pipeline_path="black-forest-labs/FLUX.1-Fill-dev", transformer_path="black-forest-labs/FLUX.1-Fill-dev/transformer", lora_path="yyyyyxie/textflux-beta-lora", device="cuda"):
        print("Initializing TextFlux pipeline...")
        self.device = device
        self.pipeline_path = pipeline_path
        self.transformer_path = transformer_path
        self.lora_path = lora_path
        self.pipe = self._load_pipeline()
        print("TextFlux pipeline initialized.")

    def _load_pipeline(self):
        transformer = FluxTransformer2DModel.from_pretrained(
            self.transformer_path,
            torch_dtype=torch.bfloat16
        )
        
        # The official TextFlux implementation uses LoRA weights.
        # We directly load them into the transformer.
        state_dict, network_alphas = FluxFillPipeline.lora_state_dict(
            self.lora_path, return_alphas=True
        )
        FluxFillPipeline.load_lora_into_transformer(
            state_dict=state_dict,
            network_alphas=network_alphas,
            transformer=transformer,
        )

        pipe = FluxFillPipeline.from_pretrained(
            self.pipeline_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        pipe.transformer.to(torch.bfloat16)
        return pipe

    def generate(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: str = "output/textflux/output.png"
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
        
        result = self.pipe(
            height=new_height,
            width=new_width,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt=prompt,
        ).images[0]
        
        result.save(output_path)
        print(f"Image saved to {output_path}")
        return result

def main():
    parser = argparse.ArgumentParser(description="TextFlux Inference Script")
    parser.add_argument("--pipeline_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev", help="Path to the pipeline model.")
    parser.add_argument("--transformer_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev/transformer", help="Path to the base transformer model.")
    parser.add_argument("--lora_path", type=str, default="yyyyyxie/textflux-beta-lora", help="Path to the TextFlux LoRA weights.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the source image.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for inpainting.")
    parser.add_argument("--output_path", type=str, default="output/textflux.png", help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=30.0, help="Guidance scale.")
    args = parser.parse_args()

    generator = TextFluxGenerator(
        pipeline_path=args.pipeline_path,
        transformer_path=args.transformer_path,
        lora_path=args.lora_path
    )
    
    input_image = load_image(args.image_path).convert("RGB")
    mask_image = load_image(args.mask_path).convert("RGB")
    
    generator.generate(
        image=input_image,
        mask_image=mask_image,
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
