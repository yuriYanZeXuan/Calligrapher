"""
Z-Image 频率分解 (F.D.) 定性消融实验

对比:
  w/ F.D.  — freq_decompose=True  (默认)，只注入高频笔画结构，保留模型低频风格
  w/o F.D. — freq_decompose=False，全频 latent 直接替换，文字区域与背景不融合

数据: ablation/qualitative_FD/exm.jsonl
输出: ablation/qualitative_FD/output_zimage/
      ├── {id}_wo_fd.png    (w/o F.D.)
      ├── {id}_w_fd.png     (w/ F.D.)
      └── {id}_compare.png  (并排对比)
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from PIL import Image, ImageDraw, ImageFont

from zimage_inference import ZImageInference, GenerationConfig
from infer.glyph_injector import InjectionConfig
from infer.mylogger import TTSLogger


def load_dataset(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def make_comparison(img_wo: Image.Image, img_w: Image.Image, prompt: str) -> Image.Image:
    """拼接 w/o F.D. 和 w/ F.D. 的对比图，底部加标注。"""
    w, h = img_wo.size
    bar_h = 50
    canvas = Image.new("RGB", (w * 2, h + bar_h), "white")
    canvas.paste(img_wo, (0, 0))
    canvas.paste(img_w, (w, 0))

    draw = ImageDraw.Draw(canvas)
    font = TTSLogger._load_font(size=20)
    draw.text((w // 2 - 40, h + 8), "w/o F.D.", fill="red", font=font)
    draw.text((w + w // 2 - 30, h + 8), "w/ F.D.", fill="green", font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Z-Image F.D. ablation")
    parser.add_argument("--model-path", type=str,
                        default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--harmonizer", type=str, default="klein",
                        choices=["klein", "qwenedit", "none"])
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), "exm.jsonl")
    out_dir = os.path.join(os.path.dirname(__file__), "output_zimage")
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset(data_path)
    print(f"加载 {len(dataset)} 条数据，输出目录: {out_dir}")

    logger = TTSLogger(run_name="zimage_fd_ablation")
    engine = ZImageInference(
        model_path=args.model_path,
        device=args.device,
        logger=logger,
    )

    use_harmonization = args.harmonizer != "none"

    for item in dataset:
        pid = item["prompt_id"]
        prompt = item["prompt"]
        texts = item["text"]
        print(f"\n{'='*60}")
        print(f"[{pid}] {prompt[:80]}")

        # --- w/o F.D. ---
        config_wo = GenerationConfig(
            num_inference_steps=args.steps,
            seed=args.seed,
            use_harmonization=use_harmonization,
            harmonizer_type=args.harmonizer if use_harmonization else "klein",
            injection_config=InjectionConfig(freq_decompose=False),
        )
        print("  >>> w/o F.D. (freq_decompose=False)")
        img_wo = engine.generate(
            prompt=prompt,
            text_contents=texts,
            config=config_wo,
            run_name=f"{pid}_wo_fd",
        )

        # --- w/ F.D. ---
        config_w = GenerationConfig(
            num_inference_steps=args.steps,
            seed=args.seed,
            use_harmonization=use_harmonization,
            harmonizer_type=args.harmonizer if use_harmonization else "klein",
            injection_config=InjectionConfig(freq_decompose=True),
        )
        print("  >>> w/ F.D. (freq_decompose=True)")
        img_w = engine.generate(
            prompt=prompt,
            text_contents=texts,
            config=config_w,
            run_name=f"{pid}_w_fd",
        )

        # --- 保存 ---
        img_wo.save(os.path.join(out_dir, f"{pid}_wo_fd.png"))
        img_w.save(os.path.join(out_dir, f"{pid}_w_fd.png"))

        comp = make_comparison(img_wo, img_w, prompt)
        comp.save(os.path.join(out_dir, f"{pid}_compare.png"))
        print(f"  保存: {out_dir}/{pid}_*.png")

    print(f"\n全部完成，共 {len(dataset)} 组对比图，保存在 {out_dir}")


if __name__ == "__main__":
    main()
