#!/usr/bin/env python3
"""
Parallel evaluation script for text rendering tasks with multi-GPU support.

Features:
- Multi-GPU parallel evaluation (data parallelism)
- Local model weights loading
- JSONL output with append mode
- Resume from checkpoint

Usage:
    # 8-GPU parallel evaluation
    python eval/scripts/eval_parallel.py \
        --results_dir /path/to/results \
        --benchmark eval/LongText-Bench/text_prompts.jsonl \
        --benchmark_type longtext \
        --output eval_results/output.jsonl \
        --metrics ocr clip \
        --gpus 8

    # Resume from checkpoint
    python eval/scripts/eval_parallel.py \
        --results_dir /path/to/results \
        --benchmark eval/LongText-Bench/text_prompts.jsonl \
        --benchmark_type longtext \
        --output eval_results/output.jsonl \
        --resume
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
import torch.multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Default local model paths
DEFAULT_MINERU_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM"
DEFAULT_VLM_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"


def load_longtext_benchmark(benchmark_path: str) -> List[Dict]:
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


def load_oneig_benchmark(benchmark_path: str) -> List[Dict]:
    """Load OneIG-Bench format data."""
    import glob
    samples = []
    
    if os.path.isfile(benchmark_path):
        json_files = [benchmark_path]
    else:
        json_files = glob.glob(os.path.join(benchmark_path, '*.json'))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            prompt = item.get('prompt_en') or item.get('prompt_cn') or item.get('prompt', '')
            samples.append({
                'id': item.get('id', ''),
                'prompt': prompt,
                'category': item.get('category', ''),
                'class': item.get('class', '')
            })
    
    return samples


def load_cvtg_benchmark(benchmark_path: str) -> List[Dict]:
    """Load CVTG-2K format data."""
    import glob
    samples = []
    
    for subdir in ['CVTG', 'CVTG-Style']:
        subdir_path = os.path.join(benchmark_path, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        for json_file in glob.glob(os.path.join(subdir_path, '*.json')):
            if 'combined' in json_file:
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            area = os.path.basename(json_file).replace('.json', '')
            for item in data.get('data_list', []):
                samples.append({
                    'id': f"{subdir}_{area}_{item.get('index', 0)}",
                    'prompt': item.get('prompt', ''),
                    'area': area,
                    'benchmark_type': subdir,
                    'carrier_list': item.get('carrier_list', []),
                    'sentence_list': item.get('sentence_list', [])
                })
    
    return samples


def load_benchmark(benchmark_path: str, benchmark_type: str) -> List[Dict]:
    """Load benchmark data based on type."""
    if benchmark_type == 'longtext':
        return load_longtext_benchmark(benchmark_path)
    elif benchmark_type == 'oneig':
        return load_oneig_benchmark(benchmark_path)
    elif benchmark_type == 'cvtg':
        return load_cvtg_benchmark(benchmark_path)
    else:
        # Generic: try to load as json
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [{'id': item.get('id', str(i)), **item} for i, item in enumerate(data)]
        return []


def find_result_image(results_dir: str, sample_id: str) -> Optional[str]:
    """Find result image for a sample."""
    results_dir = Path(results_dir)
    
    for ext in ['.png', '.jpg', '.jpeg']:
        path = results_dir / f"{sample_id}{ext}"
        if path.exists():
            return str(path)
    
    for pattern in [f"result_*_{sample_id}_*.png", f"result_{sample_id}_*.png", f"*{sample_id}*.png"]:
        matches = list(results_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None


def load_existing_results(output_path: str) -> set:
    """Load existing sample IDs from output file."""
    existing_ids = set()
    if not os.path.exists(output_path):
        return existing_ids
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        existing_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Failed to load existing results: {e}")
    
    return existing_ids


def worker_fn(rank: int, world_size: int, args, dataset: List[Dict], output_path: str):
    """Worker function for parallel evaluation."""
    import time
    time.sleep(rank * 1.0)
    
    # Stagger initialization
    import torch
    def no_op_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda x: x
        return model
    torch.compile = no_op_compile
    
    device = f"cuda:{rank}"
    
    # Data partitioning
    total = len(dataset)
    per_gpu = math.ceil(total / world_size)
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, total)
    my_dataset = dataset[start_idx:end_idx]
    
    if not my_dataset:
        print(f"[GPU {rank}] No samples assigned")
        return
    
    # Initialize evaluators
    from eval.core.metrics import OCRMetrics, CLIPMetrics, VLMMetrics
    
    evaluators = {}
    if 'ocr' in args.metrics:
        evaluators['ocr'] = OCRMetrics(model_path=args.mineru_path)
    if 'clip' in args.metrics:
        evaluators['clip'] = CLIPMetrics(device=device)
    if 'vlm' in args.metrics:
        evaluators['vlm'] = VLMMetrics(model_path=args.vlm_path, device=device)
    
    # Resume: filter already evaluated samples
    if args.resume:
        existing_ids = load_existing_results(output_path)
        my_dataset = [s for s in my_dataset if s['id'] not in existing_ids]
        if not my_dataset:
            print(f"[GPU {rank}] All samples already evaluated")
            return
        print(f"[GPU {rank}] {len(my_dataset)} samples to evaluate (skipped {len(existing_ids)})")
    
    # Open output file in append mode
    import fcntl
    
    for sample in my_dataset:
        sample_id = sample['id']
        result = {'id': sample_id}
        
        # Find image
        image_path = find_result_image(args.results_dir, sample_id)
        if not image_path:
            result['error'] = 'Image not found'
            append_result(output_path, result)
            continue
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            result['error'] = f'Failed to load image: {e}'
            append_result(output_path, result)
            continue
        
        # Ground truth text
        gt_text = sample.get('text', [])
        gt_text = ' '.join(gt_text) if isinstance(gt_text, list) else sample['prompt']
        
        # Evaluate OCR
        if 'ocr' in evaluators:
            try:
                acc = evaluators['ocr'].compute_accuracy(image, gt_text)
                result['ocr_accuracy'] = round(acc, 4)
            except Exception as e:
                result['ocr_accuracy_error'] = str(e)
        
        # Evaluate CLIP
        if 'clip' in evaluators:
            try:
                score = evaluators['clip'].compute_clip_score(image_path, sample['prompt'])
                result['clip_score'] = round(score, 2)
            except Exception as e:
                result['clip_score_error'] = str(e)
        
        # Evaluate VLM
        if 'vlm' in evaluators:
            try:
                vlm_result = evaluators['vlm'].evaluate_text_rendering(image, sample['prompt'])
                result['vlm_text_accuracy'] = round(vlm_result['text_accuracy'], 4)
                result['vlm_image_quality'] = round(vlm_result['image_quality'], 4)
                result['vlm_overall'] = round(vlm_result['overall'], 4)
            except Exception as e:
                result['vlm_error'] = str(e)
        
        # Additional metadata
        result['prompt'] = sample['prompt']
        result['category'] = sample.get('category', '')
        result['image_path'] = image_path
        
        # Append to output file (with file locking for safety)
        append_result(output_path, result)
        
        if args.verbose:
            print(f"[GPU {rank}] Evaluated: {sample_id}")


def append_result(output_path: str, result: Dict):
    """Append a single result to JSONL file with file locking."""
    import fcntl
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'a', encoding='utf-8') as f:
        # Acquire exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def compute_summary(output_path: str) -> Dict:
    """Compute summary statistics from output file."""
    results = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not results:
        return {}
    
    summary = {'total_evaluated': len(results)}
    
    # OCR summary
    ocr_scores = [r['ocr_accuracy'] for r in results if 'ocr_accuracy' in r]
    if ocr_scores:
        summary['ocr'] = {
            'mean': round(sum(ocr_scores) / len(ocr_scores), 4),
            'min': round(min(ocr_scores), 4),
            'max': round(max(ocr_scores), 4)
        }
    
    # CLIP summary
    clip_scores = [r['clip_score'] for r in results if 'clip_score' in r]
    if clip_scores:
        summary['clip'] = {
            'mean': round(sum(clip_scores) / len(clip_scores), 2),
            'min': round(min(clip_scores), 2),
            'max': round(max(clip_scores), 2)
        }
    
    # VLM summary
    vlm_scores = [r['vlm_overall'] for r in results if 'vlm_overall' in r]
    if vlm_scores:
        summary['vlm'] = {
            'mean': round(sum(vlm_scores) / len(vlm_scores), 4),
            'min': round(min(vlm_scores), 4),
            'max': round(max(vlm_scores), 4)
        }
    
    # Category-wise
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
            cat_summary['ocr_mean'] = round(sum(cat_ocr) / len(cat_ocr), 4)
        cat_clip = [r['clip_score'] for r in cat_results if 'clip_score' in r]
        if cat_clip:
            cat_summary['clip_mean'] = round(sum(cat_clip) / len(cat_clip), 2)
        summary['by_category'][cat] = cat_summary
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Parallel evaluation with multi-GPU support')
    
    # Required arguments
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing generated images')
    parser.add_argument('--benchmark', type=str, required=True,
                       help='Path to benchmark file')
    parser.add_argument('--benchmark_type', type=str, default='longtext',
                       choices=['longtext', 'oneig', 'cvtg', 'generic'],
                       help='Benchmark type')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file path')
    
    # Model paths
    parser.add_argument('--mineru_path', type=str, default=DEFAULT_MINERU_PATH,
                       help='Path to local MinerU VLM model')
    parser.add_argument('--vlm_path', type=str, default=DEFAULT_VLM_PATH,
                       help='Path to local Qwen2.5-VL model')
    
    # Metrics
    parser.add_argument('--metrics', nargs='+', default=['ocr', 'clip'],
                       choices=['ocr', 'clip', 'vlm'],
                       help='Metrics to compute')
    
    # Parallel settings
    parser.add_argument('--gpus', type=int, default=8,
                       help='Number of GPUs to use')
    
    # Resume
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing output file (skip already evaluated)')
    
    # Other
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress for each sample')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Parallel Evaluation - Text Rendering")
    print("="*60)
    print(f"Results dir: {args.results_dir}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Type: {args.benchmark_type}")
    print(f"Output: {args.output}")
    print(f"GPUs: {args.gpus}")
    print(f"Metrics: {args.metrics}")
    print(f"MinerU: {args.mineru_path}")
    print(f"VLM: {args.vlm_path}")
    print(f"Resume: {args.resume}")
    print("="*60)
    
    # Load benchmark
    print("\nLoading benchmark...")
    dataset = load_benchmark(args.benchmark, args.benchmark_type)
    print(f"Loaded {len(dataset)} samples")
    
    # Check resume
    if args.resume and os.path.exists(args.output):
        existing = load_existing_results(args.output)
        to_eval = len([s for s in dataset if s['id'] not in existing])
        print(f"Resume mode: {len(existing)} already evaluated, {to_eval} remaining")
    
    # Check if all done
    if args.resume:
        existing = load_existing_results(args.output)
        remaining = [s for s in dataset if s['id'] not in existing]
        if not remaining:
            print("\nAll samples already evaluated!")
            summary = compute_summary(args.output)
            print("\nSummary:")
            print(json.dumps(summary, indent=2))
            return
    
    # Launch parallel workers
    print(f"\nLaunching {args.gpus} workers...")
    mp.set_start_method('spawn', force=True)
    mp.spawn(
        worker_fn,
        args=(args.gpus, args, dataset, args.output),
        nprocs=args.gpus,
        join=True
    )
    
    print("\nAll workers completed!")
    
    # Compute and save summary
    summary = compute_summary(args.output)
    summary_path = args.output.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    # Workaround for torch.compile issues
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    main()
