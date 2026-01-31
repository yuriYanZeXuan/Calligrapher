#!/usr/bin/env python3
"""
Simple script to evaluate LongText-Bench results.

Usage:
    python eval/scripts/eval_longtext_simple.py \
        --results_dir /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/z_image/LongText-Bench \
        --benchmark eval/LongText-Bench/text_prompts.jsonl \
        --output eval_results/z_image_results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from eval.bak.eval_ocr import OCREvaluator
from eval.core.metrics import CLIPMetrics


def load_longtext_benchmark(benchmark_path: str):
    """Load LongText-Bench data from jsonl file."""
    samples = []
    lang_prefix = 'zh' if 'zh' in benchmark_path else 'en'
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt_id = item.get('prompt_id', len(samples))
            sample_id = f"longtext_{lang_prefix}_{prompt_id}"
            samples.append({
                'id': sample_id,
                'prompt': item.get('prompt', ''),
                'text': item.get('text', []),
                'category': item.get('category', ''),
                'length': item.get('length', '')
            })
    
    return samples


def find_result_image(results_dir: str, sample_id: str):
    """Find result image for a sample."""
    results_dir = Path(results_dir)
    
    # Try exact match
    for ext in ['.png', '.jpg', '.jpeg']:
        path = results_dir / f"{sample_id}{ext}"
        if path.exists():
            return str(path)
    
    # Try pattern match
    for pattern in [
        f"result_*_{sample_id}_*.png",
        f"result_{sample_id}_*.png",
        f"*{sample_id}*.png"
    ]:
        matches = list(results_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None


def evaluate_results(results_dir: str, benchmark_path: str, metrics: list, 
                     mineru_path: str = None, vlm_path: str = None):
    """Evaluate all results."""
    print(f"Loading benchmark from: {benchmark_path}")
    samples = load_longtext_benchmark(benchmark_path)
    print(f"Loaded {len(samples)} samples")
    
    # Initialize evaluators
    from eval.core.metrics import OCRMetrics, CLIPMetrics, VLMMetrics
    
    ocr_evaluator = OCRMetrics(model_path=mineru_path) if 'ocr' in metrics else None
    clip_evaluator = CLIPMetrics() if 'clip' in metrics else None
    vlm_evaluator = VLMMetrics(model_path=vlm_path) if 'vlm' in metrics else None
    
    results = []
    
    for sample in samples:
        sample_id = sample['id']
        image_path = find_result_image(results_dir, sample_id)
        
        if not image_path:
            print(f"  Warning: Image not found for {sample_id}")
            continue
        
        # Ground truth text
        gt_text = ' '.join(sample['text']) if sample['text'] else sample['prompt']
        
        result = {
            'id': sample_id,
            'prompt': sample['prompt'],
            'ground_truth': gt_text,
            'category': sample['category'],
            'length': sample['length'],
            'image_path': image_path
        }
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"  Error loading {image_path}: {e}")
            continue
        
        # OCR evaluation
        if ocr_evaluator:
            try:
                ocr_acc = ocr_evaluator.calculate_ocr_accuracy(image, gt_text)
                result['ocr_accuracy'] = ocr_acc
                print(f"  {sample_id}: OCR = {ocr_acc:.4f}")
            except Exception as e:
                print(f"  OCR failed for {sample_id}: {e}")
                result['ocr_accuracy'] = 0.0
        
        # CLIP evaluation
        if clip_evaluator and clip_evaluator.available:
            try:
                clip_score = clip_evaluator.compute_clip_score(image_path, sample['prompt'])
                result['clip_score'] = clip_score
                print(f"  {sample_id}: CLIP = {clip_score:.2f}")
            except Exception as e:
                print(f"  CLIP failed for {sample_id}: {e}")
        
        results.append(result)
    
    return results


def compute_summary(results: list):
    """Compute summary statistics."""
    if not results:
        return {}
    
    summary = {'total_samples': len(results)}
    
    # OCR summary
    ocr_scores = [r['ocr_accuracy'] for r in results if 'ocr_accuracy' in r]
    if ocr_scores:
        summary['mean_ocr_accuracy'] = sum(ocr_scores) / len(ocr_scores)
        summary['min_ocr_accuracy'] = min(ocr_scores)
        summary['max_ocr_accuracy'] = max(ocr_scores)
    
    # CLIP summary
    clip_scores = [r['clip_score'] for r in results if 'clip_score' in r]
    if clip_scores:
        summary['mean_clip_score'] = sum(clip_scores) / len(clip_scores)
        summary['min_clip_score'] = min(clip_scores)
        summary['max_clip_score'] = max(clip_scores)
    
    # Category-wise summary
    categories = {}
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    summary['by_category'] = {}
    for cat, cat_results in categories.items():
        cat_summary = {'count': len(cat_results)}
        cat_ocr = [r['ocr_accuracy'] for r in cat_results if 'ocr_accuracy' in r]
        if cat_ocr:
            cat_summary['mean_ocr_accuracy'] = sum(cat_ocr) / len(cat_ocr)
        cat_clip = [r['clip_score'] for r in cat_results if 'clip_score' in r]
        if cat_clip:
            cat_summary['mean_clip_score'] = sum(cat_clip) / len(cat_clip)
        summary['by_category'][cat] = cat_summary
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate LongText-Bench results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing generated images')
    parser.add_argument('--benchmark', type=str, required=True,
                       help='Path to LongText-Bench jsonl file')
    parser.add_argument('--output', type=str, default='eval_results.json',
                       help='Output JSON file path')
    parser.add_argument('--metrics', nargs='+', default=['ocr', 'clip'],
                       choices=['ocr', 'clip', 'vlm'], help='Metrics to compute')
    parser.add_argument('--mineru_path', type=str,
                       default='/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM',
                       help='Path to local MinerU VLM model')
    parser.add_argument('--vlm_path', type=str,
                       default='/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B',
                       help='Path to local Qwen2.5-VL model')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_results(args.results_dir, args.benchmark, args.metrics,
                               args.mineru_path, args.vlm_path)
    
    # Compute summary
    summary = compute_summary(results)
    
    # Save results
    output_data = {
        'detailed_results': results,
        'summary': summary
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"Total samples: {summary.get('total_samples', 0)}")
    
    if 'mean_ocr_accuracy' in summary:
        print(f"Mean OCR Accuracy: {summary['mean_ocr_accuracy']:.4f}")
        print(f"  Min: {summary['min_ocr_accuracy']:.4f}")
        print(f"  Max: {summary['max_ocr_accuracy']:.4f}")
    
    if 'mean_clip_score' in summary:
        print(f"Mean CLIP Score: {summary['mean_clip_score']:.2f}")
        print(f"  Min: {summary['min_clip_score']:.2f}")
        print(f"  Max: {summary['max_clip_score']:.2f}")
    
    if summary.get('by_category'):
        print("\nBy Category:")
        for cat, cat_summary in summary['by_category'].items():
            print(f"  {cat}: n={cat_summary['count']}", end='')
            if 'mean_ocr_accuracy' in cat_summary:
                print(f", OCR={cat_summary['mean_ocr_accuracy']:.4f}", end='')
            if 'mean_clip_score' in cat_summary:
                print(f", CLIP={cat_summary['mean_clip_score']:.2f}", end='')
            print()
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
