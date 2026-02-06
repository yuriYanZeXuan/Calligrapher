#!/usr/bin/env python3
"""
Web_rendered 数据集构建主脚本
集成：提取 -> LLM处理 -> 渲染 -> 保存
支持 resume 机制
支持 --debug 模式（10条样本）
"""

import argparse
import json
import random
from pathlib import Path
from dataclasses import asdict

import yaml

from extract_content import extract_all, filter_by_length, count_words
from llm_processor import translate, generate_image_prompt, generate_text_array, rewrite_to_length, clean_extracted_content
from render_richtext import render_content_to_image

# ============ 写死的配置 ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
TOTAL_SAMPLES_PER_LANG = 1000
DEBUG_SAMPLES_PER_LANG = 10

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
def process_single_item(item, lang: str, prompt_id: int, use_llm_clean: bool = True) -> dict:
    """处理单条数据
    
    Args:
        item: 提取的内容项
        lang: 目标语言
        prompt_id: 提示词ID
        use_llm_clean: 是否使用 LLM 清洗内容（去除噪声）
    """
    content = item.content
    paper_id = item.paper_id
    section = item.section
    text_length = item.text_length
    
    # 生成唯一ID
    item_id = f"WR_{lang}_{prompt_id}"
    
    # 【步骤0】使用 LLM 清洗原始内容（去除引用、符号等噪声）
    if use_llm_clean:
        cleaned_content = clean_extracted_content(content)
        # 如果清洗后为空，跳过此项
        if not cleaned_content:
            return None
        content = cleaned_content
    
    # 【步骤1】翻译（如果不是英文）
    if lang == "en":
        translated = content
    else:
        translated = translate(content, lang)
    
    # 生成图像prompt
    prompt = generate_image_prompt(translated, lang)
    
    # 提取关键文本
    # 使用 LLM 提取关键文本片段，并验证它们是否是原文的子集
    llm_extracted_texts = generate_text_array(translated, lang)
    
    # 简单的清理 markdown 函数 (用于验证)
    def clean_md_for_verify(text):
        return text.replace('**', '').replace('__', '').replace('`', '').replace('#', '').strip()
    
    full_text_clean = clean_md_for_verify(translated)
    
    # 验证并过滤
    verified_texts = []
    for t in llm_extracted_texts:
        t_clean = clean_md_for_verify(t)
        if t_clean and t_clean in full_text_clean:
            verified_texts.append(t_clean)
            
    # 如果验证后为空（LLM幻觉严重），回退到使用全文的前几句
    if not verified_texts:
        lines = [clean_md_for_verify(line) for line in translated.split('\n') if clean_md_for_verify(line)]
        verified_texts = lines[:5] # 取前5句作为 fallback
    
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
    # 构造包含文本的 prompt
    # 为了满足 "jsonl内保存的prompt也没有完全将text包括" 的要求，我们在 prompt 中包含全文
    # 这样 prompt 是对图片的完整描述
    full_clean_text = clean_md_for_verify(translated)
    prompt_with_text = f"Generate an image of a document with the following text content:\n\n\"{full_clean_text}\"\n\nVisual style: {prompt}"

    return {
        "category": "academic",
        "length": get_length_category(text_length),
        "prompt": prompt_with_text,
        "text": verified_texts, 
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


def build_dataset(debug: bool = False, use_llm_clean: bool = True):
    """构建数据集主函数
    
    Args:
        debug: 是否为 debug 模式（仅生成10条样本）
        use_llm_clean: 是否使用 LLM 清洗内容（去除噪声）
    """
    total_samples = DEBUG_SAMPLES_PER_LANG if debug else TOTAL_SAMPLES_PER_LANG
    max_files = 10 if debug else 500
    
    print("=" * 60)
    print("开始构建 Web_rendered 数据集")
    if debug:
        print(f"[DEBUG 模式] 每语种仅生成 {total_samples} 条样本")
    if use_llm_clean:
        print("[LLM 清洗] 启用 - 使用大模型去除内容噪声")
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
    all_contents = extract_all(PAPERS_DIR, max_files=max_files)
    print(f"  提取了 {len(all_contents)} 条原始内容")
    
    # 按长度分布采样
    print(f"\n[2/4] 按长度分布采样...")
    sampled = sample_by_distribution(all_contents, total_samples)
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
            
            result = process_single_item(item, lang, prompt_id, use_llm_clean=use_llm_clean)
            
            # 如果清洗后内容为空，跳过
            if result is None:
                print(f"    [跳过] 内容清洗后为空")
                continue
            
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
    parser = argparse.ArgumentParser(description='构建 Web_rendered 数据集')
    parser.add_argument('--debug', action='store_true', help='Debug模式，仅生成10条样本')
    parser.add_argument('--no-clean', action='store_true', help='禁用LLM内容清洗（默认启用）')
    args = parser.parse_args()
    
    build_dataset(debug=args.debug, use_llm_clean=not args.no_clean)
