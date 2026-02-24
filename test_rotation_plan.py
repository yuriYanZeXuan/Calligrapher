#!/usr/bin/env python3
"""
测试带旋转角度的文字渲染 plan

测试内容：
  - 加载 plan_rotation_chinese.json
  - 逐 region 渲染文字模板（含旋转）
  - 合成到一张全图并保存

无需 GPU / 模型，纯 CPU 即可运行。

用法:
  python test_rotation_plan.py
  python test_rotation_plan.py --plan infer/test_plans/plan_rotation_chinese.json
"""

import json
import argparse
from pathlib import Path
from PIL import Image

from infer.formula_helper import render_formula


def main():
    parser = argparse.ArgumentParser(description="测试带 rotation 的文字渲染 plan")
    parser.add_argument(
        "--plan", type=str,
        default="infer/test_plans/plan_rotation_chinese.json",
        help="plan JSON 路径",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="output/rotation_test")
    args = parser.parse_args()

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = plan.get("text_regions", [])

    print(f"Plan: {args.plan}")
    print(f"Canvas: {args.width}x{args.height}, background: black")
    print(f"Regions: {len(regions)}")
    print("=" * 60)

    canvas = Image.new("RGB", (args.width, args.height), "black")

    for i, r in enumerate(regions):
        content = r["content"]
        bbox = r["bbox"]
        rotation = r.get("rotation", 0)
        text_color = r.get("color", "#FFFFFF")
        font_weight = r.get("font_weight", "regular")

        x1 = int(bbox[0] * args.width)
        y1 = int(bbox[1] * args.height)
        x2 = int(bbox[2] * args.width)
        y2 = int(bbox[3] * args.height)
        rw, rh = max(x2 - x1, 1), max(y2 - y1, 1)

        print(f"  [{i}] \"{content}\" bbox={bbox} size={rw}x{rh} "
              f"rotation={rotation}° weight={font_weight}")

        img = render_formula(
            content, rw, rh,
            text_color=text_color,
            font_weight=font_weight,
            rotation=rotation,
        )

        region_path = output_dir / f"region_{i}_{content[:4]}.png"
        img.save(region_path)
        print(f"       -> {region_path}")

        img_no_rot = render_formula(
            content, rw, rh,
            text_color=text_color,
            font_weight=font_weight,
            rotation=0,
        )
        no_rot_path = output_dir / f"region_{i}_{content[:4]}_no_rot.png"
        img_no_rot.save(no_rot_path)

        if img.size != (rw, rh):
            img = img.resize((rw, rh), Image.LANCZOS)
        canvas.paste(img, (x1, y1))

    canvas_path = output_dir / "combined.png"
    canvas.save(canvas_path)
    print(f"\n合成图保存到: {canvas_path}")
    print("测试完成!")


if __name__ == "__main__":
    main()
