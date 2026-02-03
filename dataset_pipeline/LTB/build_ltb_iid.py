#!/usr/bin/env python3
"""
LTB_iid 数据集构建脚本
仿照 LongText-Bench 的风格，按配置的文本长度分布生成新数据
支持 resume 机制
"""

import json
import random
from pathlib import Path

from llm_processor import call_llm, translate

# ============ 写死的配置 ============
PROJECT_ROOT = Path(__file__).parent.parent.parent
LTB_INPUT_DIR = PROJECT_ROOT / "eval" / "LongText-Bench"
LTB_OUTPUT_DIR = PROJECT_ROOT / "eval" / "LTB_iid"
PROGRESS_FILE = LTB_OUTPUT_DIR / "progress.json"

# 长度分布配置（text_length 词数范围）
LENGTH_DISTRIBUTION = [
    {"range": [0, 50], "ratio": 0.10, "label": "short"},
    {"range": [50, 100], "ratio": 0.10, "label": "short"},
    {"range": [100, 200], "ratio": 0.20, "label": "medium"},
    {"range": [200, 400], "ratio": 0.30, "label": "medium"},
    {"range": [400, 800], "ratio": 0.20, "label": "long"},
    {"range": [800, 1600], "ratio": 0.10, "label": "long"},
]

# 语言配置
LANGUAGES = ["en", "zh", "ko", "ja", "ar", "fr"]

# 每语种目标样本数
TOTAL_SAMPLES_PER_LANG = 2000

# 类别列表（参考 LongText-Bench）
CATEGORIES = ["sign", "caption", "poster", "document", "article", "advertisement"]

# ============ 示例数据加载 ============
def load_examples(lang: str = "en", num_examples: int = 10) -> list[dict]:
    """加载示例数据作为 few-shot 参考"""
    if lang == "en":
        input_file = LTB_INPUT_DIR / "text_prompts.jsonl"
    else:
        input_file = LTB_INPUT_DIR / "text_prompts_zh.jsonl"
    
    if not input_file.exists():
        input_file = LTB_INPUT_DIR / "text_prompts.jsonl"
    
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    # 随机采样
    if len(examples) > num_examples:
        examples = random.sample(examples, num_examples)
    
    return examples


# ============ 进度管理 ============
def load_progress() -> dict:
    """加载进度"""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
    return {"completed": {}}


def save_progress(progress: dict):
    """保存进度"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')


# ============ LLM 生成 ============
def generate_prompt_batch(target_text_length: int, category: str, 
                          examples: list[dict], lang: str, batch_size: int = 5) -> list[dict]:
    """使用 LLM 批量生成 prompt"""
    
    # 构建示例字符串
    examples_str = ""
    for i, ex in enumerate(examples[:3]):
        examples_str += f"""
示例{i+1}:
- category: {ex.get('category', '')}
- text_length: {ex.get('text_length', 0)}
- text: {json.dumps(ex.get('text', []), ensure_ascii=False)}
- prompt: {ex.get('prompt', '')}
"""
    
    lang_map = {
        "en": "英语", "zh": "中文", "ko": "韩语", 
        "ja": "日语", "ar": "阿拉伯语", "fr": "法语"
    }
    lang_name = lang_map.get(lang, lang)
    
    system_prompt = f"""你是一个专业的文本渲染数据集生成专家。你需要仿照给定的示例，生成新的文本渲染场景描述。

参考示例格式：
{examples_str}

要求：
1. 生成 {batch_size} 条新数据，每条数据包含 category, text, prompt 三个字段
2. category 必须是: {category}
3. text 字段是一个字符串数组，包含场景中需要渲染的文字内容
4. text 数组中所有文字的总词数（word count）应该约为 {target_text_length} 词
5. prompt 是详细的场景描述，描述文字出现的环境、样式、排版等
6. 使用{lang_name}输出所有内容
7. 输出格式为 JSON 数组，每个元素包含 category, text, prompt 三个字段

直接输出 JSON 数组，不要添加任何其他内容。"""

    user_prompt = f"请生成 {batch_size} 条 category={category}、text_length 约为 {target_text_length} 词的数据。"
    
    result = call_llm(system_prompt, user_prompt, max_tokens=8192, temperature=0.9)
    
    # 解析 JSON
    # 尝试提取 JSON 数组
    result = result.strip()
    if result.startswith("```"):
        # 去除 markdown 代码块
        lines = result.split('\n')
        result = '\n'.join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    
    items = json.loads(result)
    return items


def count_text_length(text_array: list[str]) -> int:
    """计算 text 数组的总词数"""
    import re
    total = 0
    for t in text_array:
        # 中文字符
        chinese = len(re.findall(r'[\u4e00-\u9fff]', t))
        # 非中文单词
        non_chinese = re.sub(r'[\u4e00-\u9fff]', ' ', t)
        words = len([w for w in non_chinese.split() if w.strip()])
        total += chinese + words
    return total


# ============ 主流程 ============
def build_ltb_iid():
    """构建 LTB_iid 数据集主函数"""
    print("=" * 60)
    print("开始构建 LTB_iid 数据集（LLM 生成模式）")
    print("=" * 60)
    
    # 加载进度
    progress = load_progress()
    completed = progress.get("completed", {})
    
    # 创建输出目录
    LTB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for lang in LANGUAGES:
        print(f"\n{'='*40}")
        print(f"处理语言: {lang}")
        print(f"{'='*40}")
        
        # 检查已完成数量
        lang_key = f"ltb_iid_{lang}"
        completed_count = completed.get(lang_key, 0)
        
        if completed_count >= TOTAL_SAMPLES_PER_LANG:
            print(f"  [跳过] {lang} 已完成 {completed_count} 条")
            continue
        
        # 加载示例
        examples = load_examples(lang)
        
        # 输出文件
        output_file = LTB_OUTPUT_DIR / f"ltb_iid_{lang}.jsonl"
        
        # 如果文件存在，加载已有数据获取最大 ID
        existing_ids = set()
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        existing_ids.add(item.get("id", ""))
        
        prompt_id = len(existing_ids)
        
        # 按长度分布生成
        for dist in LENGTH_DISTRIBUTION:
            min_len, max_len = dist["range"]
            ratio = dist["ratio"]
            label = dist["label"]
            target_count = int(TOTAL_SAMPLES_PER_LANG * ratio)
            
            # 计算该范围已有数量
            existing_in_range = sum(1 for _ in existing_ids if True)  # 简化，按总数计算
            remaining = max(0, target_count - int(completed_count * ratio))
            
            if remaining <= 0:
                continue
            
            target_len = (min_len + max_len) // 2
            
            print(f"\n  生成 text_length={min_len}-{max_len} 的数据，目标 {remaining} 条")
            
            # 分批生成
            batch_size = 5
            generated = 0
            
            for category in CATEGORIES:
                if generated >= remaining:
                    break
                
                cat_target = remaining // len(CATEGORIES) + 1
                
                while generated < remaining and cat_target > 0:
                    print(f"    生成 category={category}, batch...")
                    
                    items = generate_prompt_batch(
                        target_text_length=target_len,
                        category=category,
                        examples=examples,
                        lang=lang,
                        batch_size=min(batch_size, cat_target)
                    )
                    
                    # 处理生成的数据
                    for item in items:
                        text_array = item.get("text", [])
                        text_length = count_text_length(text_array)
                        
                        output_item = {
                            "id": f"LTB_{lang}_{prompt_id}",
                            "prompt": item.get("prompt", ""),
                            "category": item.get("category", category),
                            "text": text_array,
                            "text_length": text_length,
                            "length": label,
                            "image_path": "",
                        }
                        
                        # 追加写入
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                        
                        prompt_id += 1
                        generated += 1
                        cat_target -= 1
                        
                        # 更新进度
                        completed[lang_key] = prompt_id
                        progress["completed"] = completed
                        save_progress(progress)
                    
                    if generated >= remaining:
                        break
        
        print(f"\n  {lang} 完成，共 {prompt_id} 条数据")
    
    print("\n" + "=" * 60)
    print("构建完成!")
    print(f"  输出目录: {LTB_OUTPUT_DIR}")
    
    # 统计输出文件
    for lang in LANGUAGES:
        output_file = LTB_OUTPUT_DIR / f"ltb_iid_{lang}.jsonl"
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            print(f"  {lang}: {count} 条")


if __name__ == '__main__':
    build_ltb_iid()
