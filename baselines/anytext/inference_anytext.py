import os
import cv2
import numpy as np
from PIL import Image
import argparse

from ms_wrapper import AnyText2Model
from util import save_images

class AnyTextInpainter:
    def __init__(self, model_dir, use_fp16=True):
        print("Initializing AnyText model for inpainting...")
        infer_params = {"use_fp16": use_fp16}
        self.model = AnyText2Model(model_dir=model_dir, **infer_params).cuda(0)
        print("AnyText model initialized.")

    def inpaint(
        self,
        img_prompt,
        text_prompt,
        source_img,
        mask_img,
        seed=-1,
        img_count=1,
        ddim_steps=20,
        strength=1.0,
        cfg_scale=7.5,
        a_prompt='best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks',
        n_prompt='low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture',
        output_dir="outputs/anytext_inpaint"
    ):
        if isinstance(source_img, str):
            source_img = np.array(Image.open(source_img).convert("RGB"))
        if isinstance(mask_img, str):
            mask_img = np.array(Image.open(mask_img).convert("L"))

        h, w = source_img.shape[:2]
        
        # In edit mode, pos_imgs are created by combining the inverted source and the mask
        pos_imgs = 255 - source_img
        # Ensure mask is single channel
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        edit_mask = cv2.resize(mask_img, (w, h))[..., None]
        pos_imgs = pos_imgs.astype(np.float32) + edit_mask.astype(np.float32)
        pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)

        params = {
            "image_count": img_count,
            "ddim_steps": ddim_steps,
            "image_width": w,
            "image_height": h,
            "strength": strength,
            "cfg_scale": cfg_scale,
            "a_prompt": a_prompt,
            "n_prompt": n_prompt,
        }
        input_data = {
            "img_prompt": img_prompt,
            "text_prompt": text_prompt,
            "seed": seed,
            "draw_pos": pos_imgs,
            "ori_image": source_img.clip(1, 255), # ori_image is the source image for inpainting
        }

        results, rtn_code, rtn_warning, _ = self.model(input_data, **params)
        
        if rtn_code >= 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_images(results, output_dir)
            print(f'AnyText inpainting results saved in: {output_dir}')
            if rtn_warning:
                print(f"Warning: {rtn_warning}")
        else:
            raise Exception(f"AnyText inference failed: {rtn_warning}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="AnyText Inpainting Script")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where the AnyText models are stored.")
    parser.add_argument("--img_prompt", type=str, required=True, help="Image prompt.")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt, e.g., '\"Hello\" \"World\"'")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source image for inpainting.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image for inpainting.")
    parser.add_argument("--output_dir", type=str, default="output/anytext_inpaint", help="Directory to save the generated images.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    
    inpainter = AnyTextInpainter(model_dir=args.model_dir)

    source_image = Image.open(args.source_path).convert("RGB")
    mask_image = Image.open(args.mask_path).convert("L")
    inpainter.inpaint(
        img_prompt=args.img_prompt,
        text_prompt=args.text_prompt,
        source_img=np.array(source_image),
        mask_img=np.array(mask_image),
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
