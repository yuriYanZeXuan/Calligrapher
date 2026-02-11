#!/usr/bin/env python3
"""
Glyph Injection 方案消融实验

控制变量地测试 6 种方案组合，所有实验共享相同的 seed、prompt、text_regions。
输出保存在 output/ablation/ 下，每个方案一张图 + 一份配置 json。

用法:
    python test_ablation.py                          # 跑全部 6 组
    python test_ablation.py --cases baseline A C+D   # 只跑指定组
    python test_ablation.py --seed 123               # 换 seed
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer.glyph_injector import InjectionConfig

# ============ 实验配置 ============

# 可选 prompt 模板（通过 --formula 切换）
FORMULAS = {
    "quadratic": {
        "prompt": '一幅展示爱因斯坦站在黑板前的图片，黑板上写着"$x=\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$"，教室里光线柔和',
        "content": r"$x=\frac{-b \pm \sqrt{b^2-4ac}}{2a}$",
    },
    "euler": {
        "prompt": '一块大学教室的黑板上写着"$e^{i\\pi} + 1 = 0$"，粉笔字迹清晰',
        "content": r"$e^{i\pi} + 1 = 0$",
    },
    "integral": {
        "prompt": '数学教材页面上印着"$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$"，排版整洁',
        "content": r"$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$",
    },
    "maxwell": {
        "prompt": '物理课讲义上写着"$\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}$"，字体清晰',
        "content": r"$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$",
    },
    "plain": {
        "prompt": '一幅展示爱因斯坦站在黑板前的图片，黑板上写着"x=(-b±√(b²-4ac))/2a"，教室里光线柔和',
        "content": "x=(-b±√(b²-4ac))/2a",
    },
}

DEFAULT_FORMULA = "quadratic"

TEXT_REGIONS_BBOX = [0.25, 0.15, 0.75, 0.45]  # 黑板区域

CASES = {
    # ---- 基线 ----
    "baseline": {
        "desc": "原始方案: 全频 latent 替换, 恒定 strength, 无注意力增强",
        "config": InjectionConfig(
            mask_strength=0.8,
            timestep_ratio=1.0,
        ),
    },

    # ---- 单方案消融 ----
    "A": {
        "desc": "方案A: 频率分解注入 — 只注入高频笔画结构，保留模型风格",
        "config": InjectionConfig(
            mask_strength=0.8,
            timestep_ratio=1.0,
            freq_decompose=True,
            freq_kernel_size=5,
        ),
    },

    "C": {
        "desc": "方案C: cosine 递减注入强度 — 早期强注入，后期放手让模型融合风格",
        "config": InjectionConfig(
            mask_strength=1.0,
            timestep_ratio=1.0,
            strength_schedule="cosine",
        ),
    },

    "D": {
        "desc": "方案D: 反向注意力抑制 — 非 glyph 区域压制文字生成",
        "config": InjectionConfig(
            mask_strength=0.8,
            timestep_ratio=1.0,
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
        ),
    },

    "E": {
        "desc": "方案E: 双路 Prompt — glyph 区域用文字 prompt, 其余用无文字 prompt",
        "config": InjectionConfig(
            mask_strength=0.8,
            timestep_ratio=1.0,
            dual_prompt=True,
        ),
    },

    # ---- 组合方案 ----
    "A+C": {
        "desc": "方案A+C: 频率分解 + cosine 递减 — 结构引导逐步淡出",
        "config": InjectionConfig(
            mask_strength=1.0,
            timestep_ratio=1.0,
            freq_decompose=True,
            freq_kernel_size=5,
            strength_schedule="cosine",
        ),
    },

    "C+D": {
        "desc": "方案C+D: cosine 递减 + 反向抑制 — 风格融合 + 去重复",
        "config": InjectionConfig(
            mask_strength=1.0,
            timestep_ratio=1.0,
            strength_schedule="cosine",
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
        ),
    },

    "A+C+D": {
        "desc": "方案A+C+D: 频率分解 + cosine 递减 + 反向抑制",
        "config": InjectionConfig(
            mask_strength=1.0,
            timestep_ratio=1.0,
            freq_decompose=True,
            freq_kernel_size=5,
            strength_schedule="cosine",
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
        ),
    },

    "A+D": {
        "desc": "方案A+D: 频率分解 + 反向抑制",
        "config": InjectionConfig(
            mask_strength=0.8,
            timestep_ratio=1.0,
            freq_decompose=True,
            freq_kernel_size=5,
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
        ),
    },

    "C+D+E": {
        "desc": "方案C+D+E: cosine递减 + 反向抑制 + 双路prompt",
        "config": InjectionConfig(
            mask_strength=1.0,
            timestep_ratio=1.0,
            strength_schedule="cosine",
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
            dual_prompt=True,
        ),
    },

    "A+D+E": {
        "desc": "方案A+D+E: 频率分解 + 反向抑制 + 双路prompt",
        "config": InjectionConfig(
            mask_strength=0.8,
            timestep_ratio=1.0,
            freq_decompose=True,
            freq_kernel_size=5,
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
            dual_prompt=True,
        ),
    },

    "A+C+D+E": {
        "desc": "方案A+C+D+E: 频率分解 + cosine递减 + 反向抑制 + 双路prompt",
        "config": InjectionConfig(
            mask_strength=1.0,
            timestep_ratio=1.0,
            freq_decompose=True,
            freq_kernel_size=5,
            strength_schedule="cosine",
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
            dual_prompt=True,
        ),
    },

    "all": {
        "desc": "全方案: A+C+D+E (频率分解+递减+抑制+双路)",
        "config": InjectionConfig(
            mask_strength=1.0,
            timestep_ratio=1.0,
            freq_decompose=True,
            freq_kernel_size=5,
            strength_schedule="cosine",
            attn_enhance_scale=2.0,
            attn_enhance_timestep_ratio=0.5,
            attn_suppress_scale=0.1,
            dual_prompt=True,
        ),
    },
}


def run_case(inference, case_name: str, case_info: dict, output_dir: Path, seed: int,
             prompt: str, text_regions: list):
    """运行单个实验。"""
    print(f"\n{'='*60}")
    print(f"实验: {case_name}")
    print(f"描述: {case_info['desc']}")
    print(f"{'='*60}")

    cfg = case_info["config"]

    # 打印关键开关状态
    flags = []
    if cfg.freq_decompose:
        flags.append(f"freq_decompose(k={cfg.freq_kernel_size})")
    if cfg.strength_schedule != "constant":
        flags.append(f"schedule={cfg.strength_schedule}")
    if cfg.attn_suppress_scale < 1.0:
        flags.append(f"suppress={cfg.attn_suppress_scale}")
    if cfg.dual_prompt:
        flags.append("dual_prompt")
    if cfg.attn_enhance_scale > 1.0:
        flags.append(f"attn_enhance={cfg.attn_enhance_scale}x")
    print(f"开关: {', '.join(flags) or 'none (baseline)'}")

    image = inference.generate(
        prompt=prompt,
        text_regions=text_regions,
        use_prompt_refiner=False,
        use_glyph_injection=True,
        injection_config=cfg,

        seed=seed,
        run_name=case_name,
    )

    # 保存图片
    img_path = output_dir / f"{case_name}.png"
    image.save(img_path)
    print(f"保存到: {img_path}")

    # 保存配置 json
    cfg_dict = {
        "case": case_name,
        "desc": case_info["desc"],
        "seed": seed,
        "prompt": prompt,
        "config": {
            k: v for k, v in cfg.__dict__.items()
        },
    }
    json_path = output_dir / f"{case_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2, default=str)

    return image


def main():
    parser = argparse.ArgumentParser(description="Glyph Injection 方案消融实验")
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help="指定要运行的实验名称，如 baseline A C+D。不指定则跑全部。"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--formula", type=str, default=DEFAULT_FORMULA,
        choices=list(FORMULAS.keys()),
        help=f"公式模板: {list(FORMULAS.keys())}"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image",
        help="模型路径"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument(
        "--output_dir", type=str, default="output/ablation",
        help="输出目录"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择公式模板
    formula_info = FORMULAS[args.formula]
    prompt = formula_info["prompt"]
    text_regions = [{"bbox": TEXT_REGIONS_BBOX, "content": formula_info["content"]}]

    # 选择实验
    if args.cases:
        cases_to_run = {k: CASES[k] for k in args.cases if k in CASES}
        unknown = [k for k in args.cases if k not in CASES]
        if unknown:
            print(f"未知实验: {unknown}")
            print(f"可选: {list(CASES.keys())}")
            return
    else:
        cases_to_run = CASES

    print(f"将运行 {len(cases_to_run)} 个实验: {list(cases_to_run.keys())}")
    print(f"公式: {args.formula} → {formula_info['content'][:60]}...")
    print(f"Seed: {args.seed}")
    print(f"输出: {output_dir}")

    # 加载模型（只加载一次）
    from zimage_inference import ZImageInference
    inference = ZImageInference(model_path=args.model_path, device=args.device)

    # 逐个运行
    for name, info in cases_to_run.items():
        run_case(inference, name, info, output_dir, args.seed, prompt, text_regions)

    print(f"\n{'='*60}")
    print(f"全部完成！结果保存在 {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
