"""
Evaluation utilities for Calligrapher project.

This module provides functions for calculating text statistics,
including character count for text with mathematical formulas.
"""

import json
from typing import List, Dict, Union, Tuple


def calculate_text_length(text: Union[str, List[str]]) -> int:
    """
    Calculate the total character length of text.
    
    For mathematical formulas, each character (including operators,
    brackets, symbols) is counted individually.
    
    Args:
        text: A single string or a list of strings
        
    Returns:
        Total character count
        
    Examples:
        >>> calculate_text_length("Hello World")
        11
        >>> calculate_text_length("C=AB with A=[a11 a12; a21 a22]")
        30
        >>> calculate_text_length(["Line 1", "Line 2"])
        12
    """
    if isinstance(text, list):
        return sum(len(t) for t in text)
    return len(text)


def analyze_text_composition(text: str) -> Dict[str, int]:
    """
    Analyze the character composition of text.
    
    Breaks down text into categories: alphabetic, numeric, 
    whitespace, and special characters (including formula symbols).
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with character type counts
        
    Example:
        >>> analyze_text_composition("Matrix C=A+B where x=1")
        {'alphabetic': 14, 'numeric': 1, 'whitespace': 3, 'special': 4, 'total': 22}
    """
    return {
        'alphabetic': sum(1 for c in text if c.isalpha()),
        'numeric': sum(1 for c in text if c.isdigit()),
        'whitespace': sum(1 for c in text if c.isspace()),
        'special': sum(1 for c in text if not c.isalnum() and not c.isspace()),
        'total': len(text)
    }


def validate_jsonl_text_length(filepath: str, fix: bool = False) -> Tuple[int, int]:
    """
    Validate and optionally fix text_length fields in a JSONL file.
    
    Checks if the 'text_length' field matches the actual character count
    of all text items. For files with mathematical formulas (like SCI_Hard_L2),
    the character count includes all formula symbols.
    
    Args:
        filepath: Path to the JSONL file
        fix: If True, fix mismatched text_length fields in place
        
    Returns:
        Tuple of (total_samples, mismatch_count)
        
    Example:
        >>> validate_jsonl_text_length("unseen_hardL2_sci.jsonl", fix=True)
        (20, 0)  # All 20 samples now have correct text_length
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    mismatches = []
    fixed_lines = []
    
    for i, line in enumerate(lines):
        data = json.loads(line.strip())
        actual_length = calculate_text_length(data['text'])
        recorded_length = data.get('text_length', 0)
        
        if actual_length != recorded_length:
            mismatches.append({
                'line': i + 1,
                'recorded': recorded_length,
                'actual': actual_length,
                'diff': actual_length - recorded_length
            })
            if fix:
                data['text_length'] = actual_length
        
        fixed_lines.append(json.dumps(data, ensure_ascii=False))
    
    if fix and mismatches:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines) + '\n')
    
    return len(lines), len(mismatches)


def get_dataset_statistics(filepath: str) -> Dict[str, Union[int, float]]:
    """
    Calculate comprehensive statistics for a dataset JSONL file.
    
    Returns statistics including average text_length, prompt_length,
    and composition breakdown.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_samples': 0,
        'total_text_length': 0,
        'total_prompt_length': 0,
        'avg_text_length': 0.0,
        'avg_prompt_length': 0.0,
        'max_text_length': 0,
        'min_text_length': float('inf'),
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text_len = calculate_text_length(data['text'])
            prompt_len = len(data.get('prompt', ''))
            
            stats['total_samples'] += 1
            stats['total_text_length'] += text_len
            stats['total_prompt_length'] += prompt_len
            stats['max_text_length'] = max(stats['max_text_length'], text_len)
            stats['min_text_length'] = min(stats['min_text_length'], text_len)
    
    if stats['total_samples'] > 0:
        stats['avg_text_length'] = stats['total_text_length'] / stats['total_samples']
        stats['avg_prompt_length'] = stats['total_prompt_length'] / stats['total_samples']
    
    return stats


# Convenience functions for specific datasets


def fix_unseen_words_dataset(directory: str = '/Users/yanzexuan/code/Calligrapher/eval/UnseenWords'):
    """
    Fix text_length fields in all UnseenWords dataset files.
    
    This function corrects the text_length field to match the actual
    character count of text items. For SCI_Hard_L2 which contains
    mathematical formulas, the character count includes all symbols
    (e.g., =, [, ], *, +, -, etc.).
    
    Args:
        directory: Directory containing the JSONL files
        
    Returns:
        Dictionary mapping filenames to (total, mismatches) tuples
    """
    import os
    
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            total, mismatches = validate_jsonl_text_length(filepath, fix=True)
            results[filename] = {'total': total, 'mismatches': mismatches}
    
    return results


def print_text_analysis_sample(filepath: str, sample_index: int = 0):
    """
    Print detailed analysis of a specific sample.
    
    Useful for understanding how text_length is calculated for
    complex cases like mathematical formulas.
    
    Args:
        filepath: Path to the JSONL file
        sample_index: Index of the sample to analyze
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if sample_index >= len(lines):
        print(f"Sample index {sample_index} out of range (total: {len(lines)})")
        return
    
    data = json.loads(lines[sample_index].strip())
    text = data['text'][0] if data['text'] else ""
    
    print(f"Sample {sample_index} Analysis")
    print("=" * 60)
    print(f"Category: {data.get('category', 'N/A')}")
    print(f"Length type: {data.get('length', 'N/A')}")
    print()
    print(f"Text content (first 150 chars):")
    print(text[:150] + "..." if len(text) > 150 else text)
    print()
    
    composition = analyze_text_composition(text)
    print("Character composition:")
    print(f"  Alphabetic: {composition['alphabetic']}")
    print(f"  Numeric: {composition['numeric']}")
    print(f"  Whitespace: {composition['whitespace']}")
    print(f"  Special (formula symbols): {composition['special']}")
    print(f"  Total: {composition['total']}")
    print()
    print(f"text_length field: {data.get('text_length', 'N/A')}")
    print(f"Calculated length: {calculate_text_length(data['text'])}")


# ============================================================================
# CVTG Dataset Statistics
# ============================================================================

def analyze_cvtg_carrier_list(directory: str = '/Users/yanzexuan/code/Calligrapher/eval/CVTG-2K/CVTG') -> dict:
    """
    Analyze carrier_list character counts in CVTG dataset.
    
    The CVTG dataset contains carrier_list as a list of strings (e.g., ['sign', 'cup']),
    representing the types of text carriers in the image.
    
    Args:
        directory: Directory containing CVTG JSON files
        
    Returns:
        Dictionary with statistics including average character count
        
    Example:
        >>> stats = analyze_cvtg_carrier_list()
        >>> print(f"Average characters: {stats['overall']['avg']:.2f}")
        Average characters: 19.19
    """
    import os
    from collections import Counter
    
    json_files = [f for f in os.listdir(directory) 
                  if f.endswith('.json') and not f.endswith('_combined.json')]
    json_files.sort()
    
    all_char_counts = []
    file_stats = {}
    
    for json_file in json_files:
        filepath = os.path.join(directory, json_file)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data_list = data.get('data_list', [])
        file_char_counts = []
        
        for item in data_list:
            carrier_list = item.get('carrier_list', [])
            if isinstance(carrier_list, list):
                # Calculate total characters in carrier_list
                total_chars = sum(len(str(c)) for c in carrier_list)
                file_char_counts.append(total_chars)
                all_char_counts.append(total_chars)
        
        if file_char_counts:
            file_stats[json_file] = {
                'samples': len(file_char_counts),
                'avg': sum(file_char_counts) / len(file_char_counts),
                'min': min(file_char_counts),
                'max': max(file_char_counts),
                'distribution': dict(Counter(file_char_counts))
            }
    
    # Overall statistics
    overall = {
        'samples': len(all_char_counts),
        'avg': sum(all_char_counts) / len(all_char_counts) if all_char_counts else 0,
        'min': min(all_char_counts) if all_char_counts else 0,
        'max': max(all_char_counts) if all_char_counts else 0,
        'total': sum(all_char_counts)
    }
    
    return {
        'files': file_stats,
        'overall': overall
    }


def print_cvtg_carrier_stats(directory: str = '/Users/yanzexuan/code/Calligrapher/eval/CVTG-2K/CVTG'):
    """
    Print formatted statistics for CVTG carrier_list.
    
    Args:
        directory: Directory containing CVTG JSON files
    """
    stats = analyze_cvtg_carrier_list(directory)
    
    print("=" * 70)
    print("CVTG Dataset - carrier_list 字符数量统计")
    print("=" * 70)
    
    # Per-file statistics
    for filename, file_stat in sorted(stats['files'].items()):
        print(f"\n{filename}:")
        print(f"  样本数: {file_stat['samples']}")
        print(f"  平均字符数: {file_stat['avg']:.2f}")
        print(f"  范围: [{file_stat['min']}, {file_stat['max']}]")
    
    # Overall statistics
    overall = stats['overall']
    print("\n" + "=" * 70)
    print("总体统计:")
    print("=" * 70)
    print(f"  总样本数: {overall['samples']}")
    print(f"  平均字符数: {overall['avg']:.2f}")
    print(f"  最小字符数: {overall['min']}")
    print(f"  最大字符数: {overall['max']}")
    print(f"  总字符数: {overall['total']}")
