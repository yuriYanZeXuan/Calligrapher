#!/usr/bin/env python3
"""
Web_rendered 数据集构建主脚本
集成：提取 -> LLM处理 -> 渲染 -> 保存
支持 resume 机制
"""

import json
import random
from pathlib import Path
from dataclasses import asdict

import yaml

from extract_content import extract_all, filter_by_length, count_words
from llm_processor import translate, generate_image_prompt, generate_text_array, rewrite_to_length
from render_richtext import render_content_to_image

# ============ 写死的配置 ============
PROJECT_ROOT = Path(__file__).parent.parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers-content"
OUTPUT_DIR = PROJECT_ROOT / "eval" / "Web_rendered"
IMAGES_DIR = OUTPUT_DIR / "images"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# 长度分布配置
LENGTH_DISTRIBUTION = [
    {"range": [0, 50], "ratio": 0.10},
    {"range": [50, 100], "ratio": 0.10},
    {"range": [100, 200], "ratio": 0.20},
    {"range": [200, 400], "ratio": 0.30},
    {"range": [400, 800], "ratio": 0.20},
    {"range": [800, 1600], "ratio": 0.10},
]

# 语言配置
LANGUAGES = ["en", "zh", "ko", "ja", "ar", "fr"]

# 每语种目标样本数
TOTAL_SAMPLES_PER_LANG = 1000  # 可调整

# ============ 进度管理 ============
def load_progress() -> dict:
    """加载进度"""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
    return {"completed_ids": [], "current_lang": None, "current_idx": 0}


def save_progress(progress: dict):
    """保存进度"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')


# ============ 数据采样 ============
def sample_by_distribution(contents: list, total: int) -> list:
    """按长度分布采样"""
    sampled = []
    
    for dist in LENGTH_DISTRIBUTION:
        min_len, max_len = dist["range"]
        ratio = dist["ratio"]
        target_count = int(total * ratio)
        
        # 过滤该长度范围的内容
        candidates = filter_by_length(contents, min_len, max_len)
        
        # 采样
        if len(candidates) >= target_count:
            selected = random.sample(candidates, target_count)
        else:
            selected = candidates  # 不够则全部使用
        
        sampled.extend(selected)
    
    return sampled


# ============ 主流程 ============
def process_single_item(item, lang: str, prompt_id: int) -> dict:
    """处理单条数据"""
    content = item.content
    paper_id = item.paper_id
    section = item.section
    text_length = item.text_length
    
    # 生成唯一ID
    item_id = f"WR_{lang}_{prompt_id}"
    
    # 翻译（如果不是英文）
    if lang == "en":
        translated = content
    else:
        translated = translate(content, lang)
    
    # 生成图像prompt
    prompt = generate_image_prompt(translated, lang)
    
    # 提取关键文本
    text_array = generate_text_array(translated, lang)
    
    # 渲染图片
    image_filename = f"{item_id}.png"
    image_path = IMAGES_DIR / lang / image_filename
    render_content_to_image(
        content=translated,
        output_path=image_path,
        text_length=count_words(translated),
        lang=lang
    )
    
    # 构建输出数据
    return {
        "category": "academic",
        "length": get_length_category(text_length),
        "prompt": prompt,
        "text": text_array,
        "text_length": count_words(translated),
        "prompt_id": prompt_id,
        "source_paper": paper_id,
        "source_section": section,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
    }


def get_length_category(length: int) -> str:
    """获取长度类别"""
    if length < 100:
        return "short"
    elif length < 400:
        return "medium"
    else:
        return "long"


def build_dataset():
    """构建数据集主函数"""
    print("=" * 60)
    print("开始构建 Web_rendered 数据集")
    print("=" * 60)
    
    # 加载进度
    progress = load_progress()
    completed_ids = set(progress.get("completed_ids", []))
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGUAGES:
        (IMAGES_DIR / lang).mkdir(parents=True, exist_ok=True)
    
    # 提取所有内容
    print(f"\n[1/4] 从 {PAPERS_DIR} 提取内容...")
    all_contents = extract_all(PAPERS_DIR, max_files=500)  # 限制文件数量加快测试
    print(f"  提取了 {len(all_contents)} 条原始内容")
    
    # 按长度分布采样
    print(f"\n[2/4] 按长度分布采样...")
    sampled = sample_by_distribution(all_contents, TOTAL_SAMPLES_PER_LANG)
    print(f"  采样了 {len(sampled)} 条内容")
    
    # 为每种语言处理
    for lang in LANGUAGES:
        print(f"\n[3/4] 处理语言: {lang}")
        
        output_file = OUTPUT_DIR / f"web_rendered_{lang}.jsonl"
        
        # 如果文件存在，加载已有数据
        existing_data = []
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = [json.loads(line) for line in f if line.strip()]
        
        existing_ids = {d.get("prompt_id") for d in existing_data}
        
        # 处理每条内容
        new_data = []
        for idx, item in enumerate(sampled):
            prompt_id = idx
            
            # 跳过已完成的
            if prompt_id in existing_ids:
                continue
            
            item_id = f"WR_{lang}_{prompt_id}"
            if item_id in completed_ids:
                continue
            
            print(f"  处理 [{idx+1}/{len(sampled)}] {item_id}...")
            
            result = process_single_item(item, lang, prompt_id)
            new_data.append(result)
            
            # 更新进度
            completed_ids.add(item_id)
            progress["completed_ids"] = list(completed_ids)
            progress["current_lang"] = lang
            progress["current_idx"] = idx
            save_progress(progress)
            
            # 追加写入
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"  {lang} 完成，新增 {len(new_data)} 条数据")
    
    print("\n[4/4] 构建完成!")
    print(f"  输出目录: {OUTPUT_DIR}")
    
    # 统计
    for lang in LANGUAGES:
        output_file = OUTPUT_DIR / f"web_rendered_{lang}.jsonl"
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            print(f"  {lang}: {count} 条")


if __name__ == '__main__':
    build_dataset()
