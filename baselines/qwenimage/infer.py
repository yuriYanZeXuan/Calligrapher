from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_suffix = ", Ultra HD, 4K, cinematic composition."

aspect_ratios = {
    "1:1": (1024, 1024),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}


def generate(prompt: str, ratio: str = "1:1", steps: int = 50, seed: int = 42) -> torch.Tensor:
    width, height = aspect_ratios[ratio]
    
    result = pipe(
        prompt=prompt + positive_suffix,
        negative_prompt="",
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(seed)
    )
    
    return result.images[0]


if __name__ == "__main__":
    prompt = "A coffee shop entrance features a chalkboard sign, with neon light beside it"
    image = generate(prompt, ratio="16:9")
    image.save("output.png")
