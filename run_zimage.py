#!/usr/bin/env python3
"""
Z-Image 推理启动脚本

支持三种模式:
  1. 正式模式 (默认):  prompt + text_contents → VLM 自主规划布局 → 三阶段推理
  2. 调试旁路模式 (--debug-bypass):  手动指定 text_regions，跳过 VLM 规划
  3. JSON plan 测试 (--plan):  直接从 JSON 文件加载 typography_plan 进行渲染

三阶段推理架构:
  Pass 1: 用完整 prompt 生成参考图 → VLM 自主规划排版
  Pass 2: 从相同噪声用 clean prompt + 字形注入生成文字图
  Pass 3: FluxKlein img2img + soft mask 背景融合循环（VLM 评判停止条件）

用法:
  python run_zimage.py                                  # 正式模式
  python run_zimage.py --debug-bypass                   # 调试旁路
  python run_zimage.py --plan infer/test_plans/plan_multi_region.json  # JSON plan 测试
  python run_zimage.py --tts --beam 8                   # 启用 TTS beam search
  python run_zimage.py --no-refiner --no-inject         # 纯生图（无注入无优化）
  python run_zimage.py --no-harmonize                   # 跳过 Pass 3 背景融合
  python run_zimage.py --klein-steps 30 --klein-iters 2 # 自定义 FluxKlein 参数
"""

import os
import json
import argparse

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch


def main():
    parser = argparse.ArgumentParser(description="Z-Image 两阶段推理")

    # 基础参数
    parser.add_argument("--prompt", default="爱因斯坦站在黑板前，黑板上写着著名的质能方程")
    parser.add_argument("--text", nargs="+", default=["E = mc^2"],
                        help="待渲染的文本/公式列表（正式模式使用）")
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    # 模式选择
    parser.add_argument("--debug-bypass", action="store_true",
                        help="调试旁路：手动指定 text_regions，跳过 VLM 规划")
    parser.add_argument("--plan", type=str, default=None,
                        help="从 JSON 文件加载 typography_plan（如 infer/test_plans/plan_multi_region.json）")

    # 功能开关
    parser.add_argument("--tts", action="store_true", help="启用 Test Time Scaling")
    parser.add_argument("--beam", type=int, default=8, help="TTS beam size")
    parser.add_argument("--no-refiner", action="store_true", help="禁用 prompt refiner")
    parser.add_argument("--no-inject", action="store_true", help="禁用 glyph injection")

    # 注入配置
    parser.add_argument("--freq-decompose", action="store_true", default=True,
                        help="频率分解注入（默认开启）")
    parser.add_argument("--no-freq-decompose", action="store_true", help="关闭频率分解注入")
    parser.add_argument("--strength-schedule", default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="注入强度衰减策略")
    parser.add_argument("--mask-strength", type=float, default=1.0, help="mask 注入强度")
    parser.add_argument("--timestep-ratio", type=float, default=0.9, help="注入时间步比例")
    parser.add_argument("--attn-enhance", type=float, default=2.0,
                        help="注意力增强倍率（1.0=不增强）")
    parser.add_argument("--attn-suppress", type=float, default=0.1,
                        help="反向注意力抑制倍率（1.0=不抑制）")

    # Pass 3: FluxKlein img2img 背景融合
    parser.add_argument("--no-harmonize", action="store_true",
                        help="禁用 Pass 3 FluxKlein 背景融合")
    parser.add_argument("--klein-model-path", type=str,
                        default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein",
                        help="FluxKlein 模型路径")
    parser.add_argument("--klein-prompt", type=str,
                        default=("The text on the surface is rewritten in beautiful style, "
                                 "with natural texture, soft edges, and subtle imperfections. "
                                 "The lettering harmonizes perfectly with the background aesthetic. "
                                 "No other changes to the image."),
                        help="FluxKlein 编辑 prompt")
    parser.add_argument("--klein-steps", type=int, default=50,
                        help="FluxKlein 推理步数")
    parser.add_argument("--klein-guidance", type=float, default=4.0,
                        help="FluxKlein guidance scale")
    parser.add_argument("--klein-seed", type=int, default=None,
                        help="FluxKlein 随机种子（默认跟随主 seed）")
    parser.add_argument("--klein-iters", type=int, default=3,
                        help="FluxKlein refine 最大循环次数")
    parser.add_argument("--klein-target", type=float, default=9.5,
                        help="FluxKlein VLM 评分目标阈值 (0-10)")
    parser.add_argument("--klein-blur-radius", type=int, default=8,
                        help="FluxKlein soft mask 高斯模糊半径 (像素)")
    parser.add_argument("--klein-cpu-offload", action="store_true",
                        help="FluxKlein 启用 CPU offload")

    # GPU
    parser.add_argument("--gpus", type=str, default=None, help="GPU 列表，如 0,1,2,3")

    args = parser.parse_args()

    # 解析 GPU
    if args.gpus:
        devices = [f"cuda:{i}" for i in args.gpus.split(",")]
    else:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"使用 GPU: {devices}")

    # 构建注入配置
    from infer.glyph_injector import InjectionConfig

    freq_on = args.freq_decompose and not args.no_freq_decompose
    injection_config = InjectionConfig(
        mask_strength=args.mask_strength,
        timestep_ratio=args.timestep_ratio,
        freq_decompose=freq_on,
        freq_kernel_size=5,
        strength_schedule=args.strength_schedule,
        attn_suppress_scale=args.attn_suppress,
        attn_enhance_scale=args.attn_enhance,
        attn_enhance_timestep_ratio=0.6,
        attn_enhance_text_to_image=True,
        attn_enhance_image_to_text=True,
    )

    print(f"注入配置: freq_decompose={freq_on}, schedule={args.strength_schedule}, "
          f"mask_strength={args.mask_strength}, attn_enhance={args.attn_enhance}, "
          f"attn_suppress={args.attn_suppress}")

    # 构建生成配置
    from zimage_inference import ZImageInference, GenerationConfig

    gen_config = GenerationConfig(
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        seed=args.seed,
        use_prompt_refiner=not args.no_refiner,
        use_glyph_injection=not args.no_inject,
        injection_config=injection_config,
        use_tts=args.tts,
        beam_size=args.beam,
        # Pass 3: FluxKlein img2img 背景融合
        use_harmonization=not args.no_harmonize,
        klein_model_path=args.klein_model_path,
        klein_prompt=args.klein_prompt,
        klein_steps=args.klein_steps,
        klein_guidance_scale=args.klein_guidance,
        klein_seed=args.klein_seed,
        klein_max_iters=args.klein_iters,
        klein_target_score=args.klein_target,
        klein_blur_radius=args.klein_blur_radius,
        klein_enable_cpu_offload=args.klein_cpu_offload,
    )

    # 创建推理实例
    inference = ZImageInference(device=devices[0])

    # ---- 模式分支 ----

    if args.plan:
        # JSON plan 测试模式：直接从文件加载 plan，走调试旁路
        print(f"\n=== JSON Plan 测试模式 ===")
        print(f"加载 plan: {args.plan}")
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = json.load(f)

        # 将 plan 中的 text_regions 作为 debug bypass 传入
        text_regions = plan.get("text_regions", [])
        print(f"Plan 包含 {len(text_regions)} 个 text region:")
        for i, r in enumerate(text_regions):
            print(f"  [{i}] \"{r['content'][:30]}\" bbox={r['bbox']} "
                  f"latex={r.get('is_latex', False)} weight={r.get('font_weight', 'regular')}")

        image = inference.generate(
            prompt=args.prompt,
            text_regions=text_regions,
            config=gen_config,
            run_name="plan_test",
        )

    elif args.debug_bypass:
        # 调试旁路模式：手动构造 text_regions
        print(f"\n=== 调试旁路模式 ===")
        text_regions = [
            {
                "content": args.text[0] if args.text else "E = mc^2",
                "bbox": [0.15, 0.25, 0.85, 0.55],
                "font_weight": "bold",
                "font_size_ratio": 0.7,
                "color": "#FFFFFF",
                "background_color": "#000000",
                "is_latex": True,
                "alignment": "center",
            }
        ]
        # 如果有多个 text，追加额外 region
        for i, t in enumerate(args.text[1:], 1):
            y_start = 0.55 + i * 0.15
            text_regions.append({
                "content": t,
                "bbox": [0.2, y_start, 0.8, min(y_start + 0.12, 0.95)],
                "font_weight": "regular",
                "font_size_ratio": 0.4,
                "color": "#CCCCCC",
                "background_color": "#000000",
                "is_latex": False,
                "alignment": "center",
            })

        print(f"手动 text_regions: {len(text_regions)} 个")
        image = inference.generate(
            prompt=args.prompt,
            text_regions=text_regions,
            config=gen_config,
            run_name="debug_bypass",
        )

    else:
        # 正式模式：VLM 自主规划
        print(f"\n=== 正式模式（VLM 自主规划）===")
        print(f"Prompt: {args.prompt}")
        print(f"Text contents: {args.text}")

        image = inference.generate(
            prompt=args.prompt,
            text_contents=args.text,
            config=gen_config,
            run_name="two_pass",
        )

    image.save(args.output)
    print(f"\n最终图像保存到: {args.output}")


if __name__ == "__main__":
    main()
