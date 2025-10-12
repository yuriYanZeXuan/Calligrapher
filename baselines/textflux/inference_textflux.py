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

    def _read_words(self, words_input):
        if isinstance(words_input, list):
            return words_input
        if os.path.exists(words_input):
            with open(words_input, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        return [line.strip() for line in words_input.splitlines() if line.strip()]

    def _generate_prompts(self, words):
        words_str = ', '.join(f"'{word}'" for word in words)
        prompt_template = (
            "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
            "[IMAGE1] is a template image rendering the text, with the words {words}; "
            "[IMAGE2] shows the text content {words} naturally and correspondingly integrated into the image."
        )
        prompt_2 = prompt_template.format(words=words_str)

        # As seen in the original script, prompt_1 is a more generic version.
        prompt_1 = (
            "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
            "[IMAGE1] is a template image rendering the text, with the words; "
            "[IMAGE2] shows the text content naturally and correspondingly integrated into the image."
        )
        return prompt_1, prompt_2

    def generate(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        words: list,
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
        
        prompt_1, prompt_2 = self._generate_prompts(words)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipe(
            height=new_height,
            width=new_width,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt=prompt_1,
            prompt_2=prompt_2,
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
    parser.add_argument("--words_path", type=str, required=True, help="Path to a text file containing the words to inpaint.")
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
    
    with open(args.words_path, 'r') as f:
        words = [line.strip() for line in f.readlines() if line.strip()]

    generator.generate(
        image=input_image,
        mask_image=mask_image,
        words=words,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
