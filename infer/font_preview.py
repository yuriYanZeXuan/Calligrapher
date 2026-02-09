#!/usr/bin/env python3
"""
字体预览生成器 - 批量生成 TTF 字体样张

用法:
    python font_preview.py /path/to/font/dir
    python font_preview.py /path/to/font/dir --sample "你好世界 Hello World" --output ./previews
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_font_preview(
    font_path: str,
    sample_text: str = "The quick brown fox jumps over the lazy dog.\n"
                       "1234567890 一二三四五六七八九十\n"
                       "Hello World 你好世界",
    size: int = 60,
    canvas_width: int = 1200,
    line_height: int = 80,
    bg_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    """为单个字体文件生成预览图"""
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception as e:
        print(f"⚠️  无法加载字体 {font_path}: {e}")
        return None
    
    # 计算需要的画布高度
    lines = sample_text.split('\n')
    padding = 60
    canvas_height = len(lines) * line_height + padding * 2 + 40  # 额外空间给文件名
    
    # 创建画布
    img = Image.new('RGB', (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 获取字体名称（从文件路径提取）
    font_name = Path(font_path).stem
    font_filename = Path(font_path).name
    
    # 绘制文件名
    try:
        title_font = ImageFont.truetype(font_path, size=24)
    except:
        title_font = ImageFont.load_default()
    
    draw.text((padding, 20), f"{font_name}", fill=text_color, font=title_font)
    draw.text((padding, 45), f"({font_filename})", fill="gray", font=title_font)
    
    # 绘制样本文本
    y_offset = padding + 40
    for line in lines:
        draw.text((padding, y_offset), line, fill=text_color, font=font)
        y_offset += line_height
    
    return img


def main():
    parser = argparse.ArgumentParser(description="生成字体预览图")
    parser.add_argument("font_dir", help="字体文件夹路径")
    parser.add_argument("--sample", "-s", default=None, help="自定义样本文本")
    parser.add_argument("--output", "-o", default="./font_previews", help="输出目录")
    parser.add_argument("--size", type=int, default=60, help="字体大小")
    parser.add_argument("--limit", "-l", type=int, default=None, help="最多处理 N 个字体")
    args = parser.parse_args()
    
    font_dir = Path(args.font_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 TTF/TTC 字体
    font_extensions = {'.ttf', '.ttc', '.otf'}
    font_files = [
        f for f in font_dir.rglob("*")
        if f.suffix.lower() in font_extensions and f.is_file()
    ]
    
    if not font_files:
        print(f"❌ 在 {font_dir} 中未找到字体文件")
        sys.exit(1)
    
    print(f"📁 找到 {len(font_files)} 个字体文件")
    print(f"📸 正在生成预览图到: {output_dir}")
    print()
    
    # 默认样本文本
    if args.sample:
        sample_text = args.sample
    else:
        sample_text = (
            "The quick brown fox jumps over the lazy dog.\n"
            "1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
            "一二三四五六七八九十 你好世界 字体预览"
        )
    
    # 限制数量
    if args.limit:
        font_files = font_files[:args.limit]
    
    # 生成预览
    success_count = 0
    for i, font_path in enumerate(font_files, 1):
        print(f"[{i}/{len(font_files)}] {font_path.name}...", end=" ")
        
        img = create_font_preview(
            str(font_path),
            sample_text=sample_text,
            size=args.size,
        )
        
        if img:
            # 保存预览图
            safe_name = "".join(c for c in font_path.stem if c.isalnum() or c in ('-', '_')).rstrip()
            output_path = output_dir / f"{safe_name}.png"
            img.save(output_path)
            print(f"✅ -> {output_path.name}")
            success_count += 1
        else:
            print("❌ 失败")
    
    print()
    print(f"🎉 完成! 成功生成 {success_count}/{len(font_files)} 张预览图")
    print(f"📂 输出目录: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
