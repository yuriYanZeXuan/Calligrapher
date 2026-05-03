import os
import torch
import argparse
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from tqdm.auto import tqdm
import string
import re

from diffusers.utils import load_image
from torchvision.transforms.functional import to_pil_image, to_tensor


# Define alphabet for custom tokens
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '

class TextDiffuser2Inpainter:
    def __init__(self, diffusion_model_path, base_model_path="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.weight_dtype = torch.float16

        print("Loading Diffusion models...")
        # Load components from base model
        self.tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

        # Add custom tokens to tokenizer
        for i in range(520):
            self.tokenizer.add_tokens(['l' + str(i)])
            self.tokenizer.add_tokens(['t' + str(i)])
            self.tokenizer.add_tokens(['r' + str(i)])
            self.tokenizer.add_tokens(['b' + str(i)])
        for c in alphabet:
            self.tokenizer.add_tokens([f'[{c}]'])

        # Load fine-tuned components
        self.text_encoder = CLIPTextModel.from_pretrained(
            diffusion_model_path, subfolder="text_encoder"
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        self.unet = UNet2DConditionModel.from_pretrained(
            diffusion_model_path, subfolder="unet", in_channels=9
        )
        
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device, dtype=self.weight_dtype)
        
        print("Diffusion models loaded.")

    @staticmethod
    def extract_text(prompt, text=None):
        if isinstance(text, list) and text:
            return " ".join(str(t) for t in text if str(t).strip())
        if isinstance(text, str) and text.strip():
            return text
        quoted = [m.group(1) for m in re.finditer(r'"([^"]+)"', prompt)]
        if quoted:
            return " ".join(quoted)
        quoted = [m.group(1) for m in re.finditer(r"'([^']+)'", prompt)]
        return " ".join(quoted) if quoted else prompt[:64]

    @staticmethod
    def make_text2image_condition(image_size=512, min_area=0.18):
        """Create blank source and centered text-region mask for direct T2I use."""
        source = Image.new("RGB", (image_size, image_size), "white")
        mask = Image.new("L", (image_size, image_size), 0)
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(mask)
        box_w = int(image_size * 0.78)
        box_h = int(image_size * max(min_area, 0.10))
        x1 = (image_size - box_w) // 2
        y1 = int(image_size * 0.42)
        draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=255)
        return source, mask

    def inpaint(
        self,
        prompt,
        source_img,
        mask_img,
        caption=None,
        seed=42,
        cfg_scale=7.0,
        sample_steps=50,
        image_size=512,
        output_path="outputs/textdiffuser2_inpaint/output.png"
    ):
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Prepare images (Official demo logic)
        if isinstance(source_img, str):
            source_img = load_image(source_img).resize((image_size, image_size))
        if isinstance(mask_img, str):
            mask_img = load_image(mask_img).resize((image_size, image_size)).convert("L")

        source_tensor = to_tensor(source_img).unsqueeze(0).to(self.device).sub_(0.5).div_(0.5)
        mask_tensor = to_tensor(mask_img).unsqueeze(0).to(self.device)

        source_tensor = source_tensor.to(dtype=self.weight_dtype)
        mask_tensor = mask_tensor.to(dtype=self.weight_dtype)
        
        masked_source_tensor = source_tensor * (1 - mask_tensor)
        
        with torch.no_grad():
            masked_feature = self.vae.encode(masked_source_tensor).latent_dist.sample()
            masked_feature = masked_feature * self.vae.config.scaling_factor
            
        feature_mask = torch.nn.functional.interpolate(mask_tensor, size=(image_size // 8, image_size // 8), mode='nearest')

        # 2. Prepare prompt with bounding box from mask (Corrected Coordinate System)
        mask_np = np.array(mask_img)
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not np.any(rows) or not np.any(cols):
            print("Warning: Mask is empty. Defaulting to full image.")
            l, t, r, b = 0, 0, 127, 127 # Use 128x128 space
        else:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            # Scale 512-space coordinates to 128-space for the model
            l, t, r, b = int(xmin / 4), int(ymin / 4), int(xmax / 4), int(ymax / 4)
        
        ocr_ids = [f'l{l}', f't{t}', f'r{r}', f'b{b}']
        char_list = [f'[{c}]' for c in prompt]
        ocr_ids.extend(char_list)
        ocr_ids.append(self.tokenizer.eos_token)
        
        # Direct T2I uses the real scene prompt as caption; legacy inpainting
        # calls can keep the generic caption by passing caption=None.
        caption = caption or "a sign with text"
        caption_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0].tolist()

        try:
            encoded_ocr = self.tokenizer.convert_tokens_to_ids(ocr_ids)
            full_prompt_ids = caption_ids + encoded_ocr
        except Exception as e:
            print(f"Tokenization error for OCR data: {e}. Proceeding with caption only.")
            full_prompt_ids = caption_ids
        
        max_length = 77 # As per the demo
        full_prompt_ids = full_prompt_ids[:max_length]
        while len(full_prompt_ids) < max_length:
            full_prompt_ids.append(self.tokenizer.pad_token_id)
        
        prompt_cond = torch.tensor([full_prompt_ids], dtype=torch.long, device=self.device)
        prompt_nocond = torch.tensor([[self.tokenizer.pad_token_id] * max_length], dtype=torch.long, device=self.device)

        # 3. Diffusion process (Corrected UNet call)
        torch.manual_seed(seed)
        noise = torch.randn((1, 4, image_size // 8, image_size // 8), device=self.device, dtype=self.weight_dtype)
        latents = noise

        self.scheduler.set_timesteps(sample_steps)
        
        encoder_hidden_states_cond = self.text_encoder(prompt_cond)[0].to(self.weight_dtype)
        encoder_hidden_states_nocond = self.text_encoder(prompt_nocond)[0].to(self.weight_dtype)

        for t in tqdm(self.scheduler.timesteps):
            with torch.no_grad():
                # Correct UNet call for the modified TextDiffuser-2 UNet
                noise_pred_cond = self.unet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states_cond,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature
                ).sample
                noise_pred_uncond = self.unet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states_nocond,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature
                ).sample
                
                noisy_residual = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                
                latents = self.scheduler.step(noisy_residual, t, latents).prev_sample

        # 4. Decode and save image
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        
        image.save(output_path)
        print(f"Image saved to {output_path}")

        return image

    def generate(
        self,
        prompt,
        output_path,
        text=None,
        seed=42,
        cfg_scale=7.0,
        sample_steps=50,
        image_size=512,
        min_area=0.18,
    ):
        """Text-to-image wrapper for benchmark integration.

        TextDiffuser-2 is an inpainting-style model, so this method creates a
        blank canvas and a centered text mask automatically, then uses the input
        prompt as the caption condition and the extracted quoted text as OCR
        content.
        """
        target_text = self.extract_text(prompt, text)
        source, mask = self.make_text2image_condition(image_size=image_size, min_area=min_area)
        return self.inpaint(
            prompt=target_text,
            caption=prompt,
            source_img=source,
            mask_img=mask,
            seed=seed,
            cfg_scale=cfg_scale,
            sample_steps=sample_steps,
            image_size=image_size,
            output_path=output_path,
        )

def main():
    parser = argparse.ArgumentParser(description="TextDiffuser-2 Inpainting Inference Script")
    parser.add_argument("--base_model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to the base Stable Diffusion v1.5 model.")
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to the fine-tuned inpainting model checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Scene prompt or text to render.")
    parser.add_argument("--text", type=str, default="", help="Target text. If empty, quoted text is extracted from prompt.")
    parser.add_argument("--source_path", type=str, default="", help="Optional source image for legacy inpainting.")
    parser.add_argument("--mask_path", type=str, default="", help="Optional mask image for legacy inpainting.")
    parser.add_argument("--text2image", action="store_true", help="Run direct text-to-image wrapper.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sample_steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--image_size", type=int, default=512, help="Output image size.")
    parser.add_argument("--output_path", type=str, default="output/textdiffuser2_inpaint.png", help="Path to save the generated image.")
    args = parser.parse_args()

    inpainter = TextDiffuser2Inpainter(
        diffusion_model_path=args.diffusion_model_path,
        base_model_path=args.base_model_path
    )

    if args.text2image or not (args.source_path and args.mask_path):
        inpainter.generate(
            prompt=args.prompt,
            text=args.text,
            seed=args.seed,
            sample_steps=args.sample_steps,
            image_size=args.image_size,
            output_path=args.output_path,
        )
    else:
        inpainter.inpaint(
            prompt=args.text or args.prompt,
            caption=args.prompt,
            source_img=args.source_path,
            mask_img=args.mask_path,
            seed=args.seed,
            sample_steps=args.sample_steps,
            image_size=args.image_size,
            output_path=args.output_path
        )

if __name__ == "__main__":
    main()
