import os
import cv2
import numpy as np
from PIL import Image
import argparse

from ms_wrapper import AnyText2Model
from util import save_images

class AnyText2Generator:
    def __init__(self, model_dir, use_fp16=True):
        print("Initializing AnyText2 model...")
        infer_params = {"use_fp16": use_fp16}
        self.model = AnyText2Model(model_dir=model_dir, **infer_params).cuda(0)
        print("AnyText2 model initialized.")

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
            "mode": 'edit',
            "sort_priority": '↕',
            "show_debug": False,
            "revise_pos": False, # disabled in edit mode
            "image_count": img_count,
            "ddim_steps": ddim_steps,
            "image_width": w,
            "image_height": h,
            "strength": strength,
            "attnx_scale": 1.0,
            "font_hollow": True,
            "cfg_scale": cfg_scale,
            "eta": 0.0,
            "a_prompt": a_prompt,
            "n_prompt": n_prompt,
            "glyline_font_path": ['None'] * 5,
            "font_hint_image": [None] * 5,
            "font_hint_mask": [None] * 5,
            "text_colors": ' '.join(['500,500,500']*5)
        }
        input_data = {
            "img_prompt": img_prompt,
            "text_prompt": text_prompt,
            "seed": seed,
            "draw_pos": pos_imgs,
            "ori_image": source_img.clip(1, 255), # ori_image is the source image for inpainting
        }

        results, rtn_code, rtn_warning, debug_info = self.model(input_data, **params)
        
        if rtn_code >= 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_images(results, output_dir)
            print(f'AnyText2 inpainting results saved in: {output_dir}')
            if rtn_warning:
                print(f"Warning: {rtn_warning}")
        else:
            raise Exception(f"AnyText2 inference failed: {rtn_warning}")
        
        return results

    def generate(
        self,
        img_prompt,
        text_prompt,
        ref_img,
        seed=-1,
        img_count=1,
        ddim_steps=20,
        image_width=512,
        image_height=512,
        strength=1.0,
        cfg_scale=7.5,
        a_prompt='best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks',
        n_prompt='low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture',
        output_dir="outputs/anytext"
    ):
        # In AnyText2, the reference image is used to derive the position map.
        # The white areas in the reference become the drawing canvas for the text.
        if isinstance(ref_img, str):
            ref_img = np.array(Image.open(ref_img).convert("RGB"))

        # Assuming the ref_img mask is provided where text should be.
        # AnyText expects a position map where non-white areas are for drawing.
        # We simulate this from the reference mask.
        pos_imgs = 255 - ref_img

        params = {
            "mode": 'gen',
            "sort_priority": '↕',
            "show_debug": False,
            "revise_pos": False,
            "image_count": img_count,
            "ddim_steps": ddim_steps,
            "image_width": image_width,
            "image_height": image_height,
            "strength": strength,
            "attnx_scale": 1.0,
            "font_hollow": True,
            "cfg_scale": cfg_scale,
            "eta": 0.0,
            "a_prompt": a_prompt,
            "n_prompt": n_prompt,
            "glyline_font_path": ['None'] * 5,
            "font_hint_image": [None] * 5,
            "font_hint_mask": [None] * 5,
            "text_colors": ' '.join(['500,500,500']*5)
        }
        input_data = {
            "img_prompt": img_prompt,
            "text_prompt": text_prompt,
            "seed": seed,
            "draw_pos": pos_imgs,
            "ori_image": None, # Not used in generation mode
        }

        results, rtn_code, rtn_warning, debug_info = self.model(input_data, **params)

        if rtn_code >= 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_images(results, output_dir)
            print(f'AnyText2 images saved in: {output_dir}')
            if rtn_warning:
                print(f"Warning: {rtn_warning}")
        else:
            raise Exception(f"AnyText2 inference failed: {rtn_warning}")
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="gen", choices=["gen", "inpaint"], help="Mode of operation.")
    parser.add_argument("--model_dir", type=str, default="../../AnyText2/models", help="Directory where the AnyText2 models are stored.")
    parser.add_argument("--img_prompt", type=str, required=True, help="Image prompt.")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt, e.g., '\"Hello\" \"World\"'")
    parser.add_argument("--ref_path", type=str, help="Path to the reference image/mask for 'gen' mode.")
    parser.add_argument("--source_path", type=str, help="Path to the source image for 'inpaint' mode.")
    parser.add_argument("--mask_path", type=str, help="Path to the mask image for 'inpaint' mode.")
    parser.add_argument("--output_dir", type=str, default="output/anytext_test", help="Directory to save the generated images.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    
    generator = AnyText2Generator(model_dir=args.model_dir)

    if args.mode == "gen":
        if not args.ref_path:
            raise ValueError("--ref_path is required for 'gen' mode.")
        ref_image = Image.open(args.ref_path).convert("RGB")
        generator.generate(
            img_prompt=args.img_prompt,
            text_prompt=args.text_prompt,
            ref_img=np.array(ref_image),
            seed=args.seed,
            output_dir=args.output_dir
        )
    elif args.mode == "inpaint":
        if not args.source_path or not args.mask_path:
            raise ValueError("--source_path and --mask_path are required for 'inpaint' mode.")
        source_image = Image.open(args.source_path).convert("RGB")
        mask_image = Image.open(args.mask_path).convert("L")
        generator.inpaint(
            img_prompt=args.img_prompt,
            text_prompt=args.text_prompt,
            source_img=np.array(source_image),
            mask_img=np.array(mask_image),
            seed=args.seed,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
