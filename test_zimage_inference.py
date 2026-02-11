#!/usr/bin/env python3
"""
Z-Image 推理测试脚本

测试用例：让爱因斯坦写二次方程求根公式
"""

import os
import sys
import argparse
from pathlib import Path

# 禁用 torch.compile
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch


def test_basic_generation(inference, output_dir: Path):
    """测试基础生成（无文字注入）"""
    print("\n" + "="*60)
    print("测试 1: 基础图像生成")
    print("="*60)
    
    prompt = "一幅展示爱因斯坦站在黑板前的照片，他穿着灰色西装，头发蓬松，表情专注地看着黑板"
    
    image = inference.generate(
        prompt=prompt,
        use_prompt_refiner=False,
        use_glyph_injection=False,
        seed=42
    )
    
    output_path = output_dir / "test_basic.png"
    image.save(output_path)
    print(f"保存到: {output_path}")
    return image


def test_with_prompt_refiner(inference, output_dir: Path):
    """测试 Prompt Refiner"""
    print("\n" + "="*60)
    print("测试 2: 使用 Prompt Refiner")
    print("="*60)
    
    prompt = "爱因斯坦在黑板前写公式"
    text_content = "E = mc²"
    
    # 只测试 prompt refiner
    from infer.prompt_refiner import refine_prompt
    
    refined = refine_prompt(prompt, text_content, num_variants=2)
    print(f"原始 prompt: {prompt}")
    print(f"文字内容: {text_content}")
    print(f"\n优化后的 prompts:")
    for i, r in enumerate(refined, 1):
        print(f"  变体 {i}: {r[:100]}...")
    
    # 使用优化后的 prompt 生成
    image = inference.generate(
        prompt=prompt,
        text_regions=[{"bbox": [0.3, 0.2, 0.8, 0.5], "content": text_content}],
        use_prompt_refiner=True,
        use_glyph_injection=False,
        seed=42
    )
    
    output_path = output_dir / "test_prompt_refiner.png"
    image.save(output_path)
    print(f"保存到: {output_path}")
    return image


def test_with_glyph_injection(inference, output_dir: Path):
    """测试 Glyph Injection"""
    print("\n" + "="*60)
    print("测试 3: 使用 Glyph Injection")
    print("="*60)
    
    prompt = '一幅展示爱因斯坦站在黑板前的图片，黑板上写着"x=(-b±√(b²-4ac))/2a"，教室里光线柔和'
    
    text_regions = [
        {
            "bbox": [0.25, 0.15, 0.75, 0.45],  # 黑板区域
            "content": "x=(-b±√(b²-4ac))/2a"
        }
    ]
    
    from infer.glyph_injector import InjectionConfig
    
    image = inference.generate(
        prompt=prompt,
        text_regions=text_regions,
        use_prompt_refiner=False,
        use_glyph_injection=True,
        injection_config=InjectionConfig(mask_strength=0.),
        seed=42
    )
    
    output_path = output_dir / "test_glyph_injection.png"
    image.save(output_path)
    print(f"保存到: {output_path}")
    return image


def test_full_pipeline(inference, output_dir: Path):
    """测试完整流程"""
    print("\n" + "="*60)
    print("测试 5: 完整流程（Prompt Refiner + Glyph Injection）")
    print("="*60)
    
    prompt = "爱因斯坦在教室黑板前讲课，写着二次方程求根公式"
    
    text_regions = [
        {
            "bbox": [0.2, 0.1, 0.8, 0.5],
            "content": "x = (-b ± √(b²-4ac)) / 2a"
        }
    ]
    
    from infer.glyph_injector import InjectionConfig
    
    image = inference.generate(
        prompt=prompt,
        text_regions=text_regions,
        use_prompt_refiner=True,
        use_glyph_injection=True,
        injection_config=InjectionConfig(mask_strength=0.6),
        seed=42
    )
    
    output_path = output_dir / "test_full_pipeline.png"
    image.save(output_path)
    print(f"保存到: {output_path}")
    return image


def test_parallel_generation(model_path: str, output_dir: Path):
    """测试多 GPU 并行生成"""
    print("\n" + "="*60)
    print("测试 6: 多 GPU 并行生成")
    print("="*60)
    
    num_gpus = torch.cuda.device_count()
    print(f"可用 GPU 数量: {num_gpus}")
    
    if num_gpus < 2:
        print("GPU 数量不足，跳过并行测试")
        return None
    
    from zimage_inference import ParallelZImageInference, GenerationConfig
    
    parallel_inference = ParallelZImageInference(
        model_path=model_path,
        devices=[f"cuda:{i}" for i in range(min(num_gpus, 8))]
    )
    
    prompts = [
        "爱因斯坦在黑板前写 E=mc²",
        "牛顿在苹果树下思考",
        "达芬奇在画蒙娜丽莎",
        "居里夫人在实验室做实验"
    ]
    
    config = GenerationConfig(
        height=1024,
        width=1024,
        num_inference_steps=9,
        use_prompt_refiner=False,
        use_glyph_injection=False,
        seed=42
    )
    
    images = parallel_inference.generate_batch(prompts, config=config)
    
    for i, img in enumerate(images):
        output_path = output_dir / f"test_parallel_{i}.png"
        img.save(output_path)
        print(f"保存到: {output_path}")
    
    parallel_inference.close()
    return images


def test_text_rendering():
    """测试文字渲染功能"""
    print("\n" + "="*60)
    print("测试 7: 文字渲染和 Mask 提取")
    print("="*60)
    
    from infer.glyph_injector import GlyphInjector, get_available_font
    import numpy as np
    import cv2
    from PIL import Image
    
    # 创建临时 injector
    class MockInjector:
        pass
    
    injector = MockInjector()
    
    # 测试文字渲染
    text = "x = (-b ± √(b²-4ac)) / 2a"
    img = GlyphInjector.render_text_template(injector, text, 512, 128)
    img.save("/tmp/test_text_render.png")
    print(f"文字渲染完成: /tmp/test_text_render.png")
    
    # 测试 mask 提取
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    mask = GlyphInjector.extract_text_mask(injector, img_bgr)
    cv2.imwrite("/tmp/test_text_mask.png", mask)
    print(f"Mask 提取完成: /tmp/test_text_mask.png")
    
    # 显示统计
    text_ratio = np.sum(mask == 255) / mask.size
    print(f"文字区域占比: {text_ratio*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Z-Image 推理测试")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image",
        help="模型路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/zimage_test",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "basic", "refiner", "injection", "full", "parallel", "render"],
        default="render",
        help="运行的测试类型"
    )
    parser.add_argument(
        "--skip_model",
        action="store_true",
        help="跳过需要加载模型的测试"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Z-Image 推理测试")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {args.device}")
    print(f"可用 GPU: {torch.cuda.device_count()}")
    
    # 测试文字渲染（不需要模型）
    if args.test in ["all", "render"]:
        test_text_rendering()
    
    if args.skip_model:
        print("\n跳过需要模型的测试")
        return
    
    # 创建推理实例
    if args.test not in ["render"]:
        from zimage_inference import ZImageInference
        inference = ZImageInference(
            model_path=args.model_path,
            device=args.device
        )
    
    # 运行测试
    if args.test in ["all", "basic"]:
        test_basic_generation(inference, output_dir)
    
    if args.test in ["all", "refiner"]:
        test_with_prompt_refiner(inference, output_dir)
    
    if args.test in ["all", "injection"]:
        test_with_glyph_injection(inference, output_dir)
    
    if args.test in ["all", "full"]:
        test_full_pipeline(inference, output_dir)
    
    if args.test in ["all", "parallel"]:
        test_parallel_generation(args.model_path, output_dir)
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
