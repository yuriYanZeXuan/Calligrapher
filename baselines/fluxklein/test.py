#!/usr/bin/env python3
"""
FluxKlein Mask-Guided Text Style Editing Test

使用 FluxKlein 将 mask 内部的文字改为与背景风格协调的粉笔字，
通过 soft mask 控制背景区域完全不变。

用法:
    python test.py
    python test.py --image <path> --mask <path> --model_path <path>
    python test.py --prompt "custom prompt" --seed 123
"""

import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image, ImageFilter

# 确保能导入同目录下的 FluxKlein 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_fluxklein import FluxKleinGenerator


# ============ 工具函数 ============


def load_soft_mask(
    mask_path: str, target_size: tuple, blur_radius: int = 8,
) -> np.ndarray:
    """加载 mask 并生成 soft mask（高斯模糊平滑边缘）。

    Args:
        mask_path: mask 图像路径（亮=文字区域，暗=背景）
        target_size: (width, height) 目标尺寸
        blur_radius: 高斯模糊半径

    Returns:
        float32 (H, W)，1.0=文字区域，0.0=背景
    """
    mask = Image.open(mask_path).convert("L").resize(target_size, Image.LANCZOS)
    mask_arr = np.array(mask).astype(np.float32) / 255.0

    # 高斯模糊平滑边缘，避免混合时出现硬边
    if blur_radius > 0:
        mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask_arr = np.array(mask_pil).astype(np.float32) / 255.0

    return mask_arr


def masked_blend(
    original: Image.Image, edited: Image.Image, mask: np.ndarray,
) -> Image.Image:
    """Soft mask 像素混合：mask 内用编辑结果，mask 外保持原图。

    blended = mask * edited + (1 - mask) * original
    """
    # 确保尺寸匹配：以 edited 为基准 resize original 和 mask
    if original.size != edited.size:
        original = original.resize(edited.size, Image.LANCZOS)
    
    orig_arr = np.array(original).astype(np.float32)
    edit_arr = np.array(edited).astype(np.float32)
    
    # resize mask 到 edited 尺寸
    if mask.shape[1] != edited.width or mask.shape[0] != edited.height:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(edited.size, Image.LANCZOS)
        mask = np.array(mask_pil).astype(np.float32) / 255.0
    
    mask_3ch = mask[:, :, np.newaxis]
    blended = mask_3ch * edit_arr + (1 - mask_3ch) * orig_arr
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ============ 主流程 ============


def main():
    parser = argparse.ArgumentParser(
        description="FluxKlein mask-guided text style editing",
    )

    # 输入
    parser.add_argument(
        "--image", type=str,
        default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/logs/"
                "20260210_214601_zimage_inference/images/harmonize/pass3_iter1.png",
        help="参考图路径（Pass 3 输出）",
    )
    parser.add_argument(
        "--mask", type=str,
        default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/logs/"
                "20260210_205155_zimage_inference/images/harmonize/pass3_soft_mask.png",
        help="Mask 路径（白=文字区域）",
    )

    # 模型
    parser.add_argument(
        "--model_path", type=str,
        default="black-forest-labs/FLUX.2-klein-base-9B",
        help="FluxKlein 模型路径",
    )

    # 生成参数
    parser.add_argument(
        "--prompt", type=str,
        default=(
            "The text on the surface is rewritten in beautiful chalk style, "
            "with natural chalk texture, soft powdery edges, and subtle imperfections. "
            "The chalk lettering harmonizes perfectly with the background aesthetic. "
            "No other changes to the image."
        ),
        help="编辑 prompt（描述粉笔字风格）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument(
        "--blur_radius", type=int, default=8,
        help="Soft mask 高斯模糊半径（像素），控制混合过渡宽度",
    )
    parser.add_argument("--enable_cpu_offload", action="store_true")

    # 输出
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认为脚本同级 output/）",
    )

    args = parser.parse_args()

    # 输出目录
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. 加载图像和 mask ----
    print(f"Loading image: {args.image}")
    print(f"Loading mask:  {args.mask}")
    original = Image.open(args.image).convert("RGB")
    soft_mask = load_soft_mask(args.mask, original.size, blur_radius=args.blur_radius)

    # 保存处理后的 soft mask 以供检查
    mask_vis_path = os.path.join(output_dir, "soft_mask_prepared.png")
    Image.fromarray((soft_mask * 255).astype(np.uint8)).save(mask_vis_path)
    print(f"Image size: {original.size}, mask coverage: {soft_mask.mean():.3f}")
    print(f"Prepared soft mask saved to {mask_vis_path}")

    # ---- 2. FluxKlein 图像编辑 ----
    print(f"\nPrompt: {args.prompt[:120]}...")
    generator = FluxKleinGenerator(
        model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein",
        enable_cpu_offload=args.enable_cpu_offload,
    )

    raw_edit_path = os.path.join(output_dir, "klein_raw_edit.png")
    edited = generator.generate(
        prompt=args.prompt,
        image=original,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        output_path=raw_edit_path,
    )
    print(f"FluxKlein raw edit saved to {raw_edit_path}")

    # ---- 3. Mask 混合：文字区域用编辑结果，背景保持不变 ----
    print("Applying mask blending...")
    result = masked_blend(original, edited, soft_mask)
    result_path = os.path.join(output_dir, "klein_masked_result.png")
    result.save(result_path)
    print(f"Final result saved to {result_path}")

    # ---- 4. 保存对比拼图 ----
    comparison = Image.new("RGB", (original.width * 3, original.height))
    comparison.paste(original, (0, 0))
    comparison.paste(edited, (original.width, 0))
    comparison.paste(result.resize(original.size, Image.LANCZOS), (original.width * 2, 0))
    comp_path = os.path.join(output_dir, "comparison.png")
    comparison.save(comp_path)
    print(f"Comparison (original | raw edit | masked result) saved to {comp_path}")


if __name__ == "__main__":
    main()
