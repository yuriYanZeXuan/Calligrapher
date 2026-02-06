#!/usr/bin/env python3
"""
后处理脚本：为已生成的 JSONL 文件中的 prompt 添加文本双引号
"""

import json
import re
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "eval" / "Web_rendered"


def fix_prompt_quotes(prompt: str) -> str:
    """为 prompt 中的文本内容添加双引号"""
    # 匹配模式: "...text content:\n\n" + 文本 + "\n\nVisual style:"
    pattern = r'(Generate an image of a document with the following text content:\n\n)(.*?)(\n\nVisual style:)'
    
    def add_quotes(match):
        prefix = match.group(1)
        text = match.group(2)
        suffix = match.group(3)
        # 如果已经有双引号，不重复添加
        if text.startswith('"') and text.endswith('"'):
            return match.group(0)
        return f'{prefix}"{text}"{suffix}'
    
    return re.sub(pattern, add_quotes, prompt, flags=re.DOTALL)


def process_file(jsonl_path: Path):
    """处理单个 JSONL 文件"""
    if not jsonl_path.exists():
        return
    
    # 读取所有数据
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 修正 prompt
    modified = 0
    for item in data:
        old_prompt = item.get("prompt", "")
        new_prompt = fix_prompt_quotes(old_prompt)
        if old_prompt != new_prompt:
            item["prompt"] = new_prompt
            modified += 1
    
    # 写回文件
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  {jsonl_path.name}: 修正了 {modified} 条记录")


def main():
    print("修正 prompt 中的文本双引号...")
    
    # 处理所有语言的 JSONL 文件
    for lang in ["en", "zh", "ko", "ja", "ar", "fr"]:
        jsonl_path = OUTPUT_DIR / f"web_rendered_{lang}.jsonl"
        process_file(jsonl_path)
    
    print("完成!")


if __name__ == '__main__':
    main()
