#!/usr/bin/env python3
"""
从 papers-content 提取混合内容（文本+公式+表格）的模块
"""

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExtractedContent:
    """提取的内容数据结构"""
    paper_id: str
    section: str
    content: str  # 清洗后的 Markdown 内容
    text_length: int  # 词数（中文按字数，其他语言按空格分词）
    has_formula: bool
    has_table: bool


def count_words(text: str) -> int:
    """统计词数：中文按字数，其他语言按空格分词"""
    # 去除公式和特殊标记
    clean_text = re.sub(r'\\\(.*?\\\)', '', text)
    clean_text = re.sub(r'\\\[.*?\\\]', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'\$\$.*?\$\$', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'\$.*?\$', '', clean_text)
    
    # 中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_text))
    # 非中文单词
    non_chinese = re.sub(r'[\u4e00-\u9fff]', ' ', clean_text)
    words = len([w for w in non_chinese.split() if w.strip()])
    
    return chinese_chars + words


def clean_markdown(content: str) -> str:
    """清洗 Markdown 内容，保留公式和表格"""
    # 去除图片引用
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # 去除链接但保留文字
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    # 去除 HTML 标签
    content = re.sub(r'<[^>]+>', '', content)
    # 去除引用编号 [[1], [2,3]] 等
    content = re.sub(r'\[\[[\d,\s]+\]\]', '', content)
    content = re.sub(r'\[[\d,\s]+\]', '', content)
    # 去除脚注标记
    content = re.sub(r'\s*\d+\s*footnotemark:\s*\d+', '', content)
    # 清理多余空行
    content = re.sub(r'\n{3,}', '\n\n', content)
    # 清理行首行尾空白
    lines = [line.strip() for line in content.split('\n')]
    content = '\n'.join(lines)
    
    return content.strip()


def extract_sections(md_content: str) -> list[tuple[str, str]]:
    """从 Markdown 中提取各个章节"""
    sections = []
    current_section = "abstract"
    current_content = []
    
    for line in md_content.split('\n'):
        # 检测标题
        if line.startswith('#'):
            if current_content:
                text = '\n'.join(current_content).strip()
                if text:
                    sections.append((current_section, text))
            # 提取新章节名
            current_section = re.sub(r'^#+\s*', '', line).strip()
            current_section = re.sub(r'^\d+\.?\s*', '', current_section)
            current_content = []
        else:
            current_content.append(line)
    
    # 最后一个章节
    if current_content:
        text = '\n'.join(current_content).strip()
        if text:
            sections.append((current_section, text))
    
    return sections


def extract_paragraphs(section_content: str, min_length: int = 20) -> list[str]:
    """从章节内容中提取段落"""
    paragraphs = []
    current = []
    
    for line in section_content.split('\n'):
        line = line.strip()
        if not line:
            if current:
                para = ' '.join(current)
                if count_words(para) >= min_length:
                    paragraphs.append(para)
                current = []
        else:
            current.append(line)
    
    if current:
        para = ' '.join(current)
        if count_words(para) >= min_length:
            paragraphs.append(para)
    
    return paragraphs


def has_formula(content: str) -> bool:
    """检测是否包含公式"""
    patterns = [
        r'\\\(.*?\\\)',  # \( \)
        r'\\\[.*?\\\]',  # \[ \]
        r'\$\$.*?\$\$',  # $$...$$
        r'\$[^$]+\$',    # $...$
    ]
    for p in patterns:
        if re.search(p, content, re.DOTALL):
            return True
    return False


def has_table(content: str) -> bool:
    """检测是否包含表格"""
    # Markdown 表格检测
    return bool(re.search(r'\|.*\|.*\|', content))


def extract_from_file(md_path: Path) -> list[ExtractedContent]:
    """从单个 MD 文件提取内容"""
    paper_id = md_path.stem
    content = md_path.read_text(encoding='utf-8')
    
    # 跳过元数据部分
    if 'Markdown Content:' in content:
        content = content.split('Markdown Content:', 1)[1]
    
    results = []
    sections = extract_sections(content)
    
    for section_name, section_content in sections:
        cleaned = clean_markdown(section_content)
        paragraphs = extract_paragraphs(cleaned)
        
        for i, para in enumerate(paragraphs):
            results.append(ExtractedContent(
                paper_id=paper_id,
                section=f"{section_name}_{i}",
                content=para,
                text_length=count_words(para),
                has_formula=has_formula(para),
                has_table=has_table(para),
            ))
    
    return results


def extract_all(papers_dir: Path, max_files: int = None) -> list[ExtractedContent]:
    """从整个 papers-content 目录提取内容"""
    all_results = []
    md_files = sorted(papers_dir.glob('**/*.md'))
    
    if max_files:
        md_files = md_files[:max_files]
    
    for md_path in md_files:
        results = extract_from_file(md_path)
        all_results.extend(results)
    
    return all_results


def filter_by_length(contents: list[ExtractedContent], 
                     min_len: int, max_len: int) -> list[ExtractedContent]:
    """按词数范围过滤内容"""
    return [c for c in contents if min_len <= c.text_length < max_len]


if __name__ == '__main__':
    # 测试
    papers_dir = Path(__file__).parent.parent.parent / 'papers-content'
    results = extract_all(papers_dir, max_files=5)
    
    print(f"提取了 {len(results)} 条内容")
    for r in results[:3]:
        print(f"  [{r.paper_id}] {r.section}: {r.text_length} words, formula={r.has_formula}")
        print(f"    {r.content[:100]}...")
