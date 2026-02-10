#!/usr/bin/env python3
"""
测试 formula_helper 渲染 plan_multi_region.json 中的文本区域
"""

import json
import sys
from pathlib import Path

# 确保能导入 formula_helper
sys.path.insert(0, str(Path(__file__).parent))

from formula_helper import render_formula, is_latex
from PIL import Image


def hex_to_rgb(hex_color: str) -> str:
    """将 HEX 颜色转换为 PIL 可用的颜色字符串"""
    hex_color = hex_color.lstrip('#')
    return f"#{hex_color}"


def render_plan_text_regions(plan_path: str, output_dir: str = "./formula_test_output"):
    """渲染 plan JSON 中的所有文本区域"""
    
    # 读取 plan
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    text_regions = plan.get("text_regions", [])
    
    print(f"找到 {len(text_regions)} 个文本区域")
    print("=" * 60)
    
    for i, region in enumerate(text_regions, 1):
        content = region["content"]
        bbox = region["bbox"]  # [x_min, y_min, x_max, y_max] 归一化坐标
        is_latex_flag = region.get("is_latex", False)
        color = region.get("color", "#000000")
        bg_color = region.get("background_color", "#FFFFFF")
        
        # 计算尺寸 (使用固定宽度 1024，按比例计算高度)
        canvas_width = 1024
        x1, y1, x2, y2 = bbox
        region_width = int((x2 - x1) * canvas_width)
        region_height = int((y2 - y1) * canvas_width)  # 假设正方形像素
        
        print(f"\n区域 {i}:")
        print(f"  内容: {content[:60]}{'...' if len(content) > 60 else ''}")
        print(f"  类型: {'LaTeX' if is_latex_flag or is_latex(content) else '纯文本'}")
        print(f"  尺寸: {region_width}x{region_height}")
        print(f"  文字颜色: {color}, 背景: {bg_color}")
        
        # 转换颜色
        text_color = hex_to_rgb(color)
        background_color = hex_to_rgb(bg_color)
        
        # 渲染
        try:
            img = render_formula(
                content,
                width=region_width,
                height=region_height,
                text_color=text_color,
                background_color=background_color,
                force_latex=is_latex_flag
            )
            
            # 保存
            save_path = output_path / f"region_{i}_{content[:20].replace(' ', '_').replace('$', '')}.png"
            img.save(save_path)
            print(f"  ✅ 已保存: {save_path}")
            
        except Exception as e:
            print(f"  ❌ 渲染失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"全部完成！输出目录: {output_path.absolute()}")


if __name__ == "__main__":
    # 默认使用 plan_multi_region.json
    default_plan = Path(__file__).parent / "test_plans" / "plan_multi_region.json"
    
    plan_file = sys.argv[1] if len(sys.argv) > 1 else str(default_plan)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./formula_test_output"
    
    print(f"使用 plan: {plan_file}")
    render_plan_text_regions(plan_file, output_dir)
