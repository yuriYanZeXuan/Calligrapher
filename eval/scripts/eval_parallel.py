#!/usr/bin/env python3
"""
Parallel evaluation script for text rendering tasks with multi-GPU support.

Features:
- Metric-by-metric evaluation (run all samples for one metric, then next metric)
- GPU memory released after each metric completes
- JSONL output with streaming write (one line per sample)
- Fine-grained resume: skip only if specific metric already exists for a sample

Usage:
    # 8-GPU parallel evaluation
    python eval/scripts/eval_parallel.py \
        --results_dir /path/to/results \
        --benchmark eval/LongText-Bench/text_prompts.jsonl \
        --benchmark_type longtext \
        --output eval_results/output.jsonl \
        --metrics vqa ocr clip \
        --gpus 8

    # Resume from checkpoint (checks each metric field per sample)
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
import gc
import fcntl
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from PIL import Image
import torch
import torch.multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from eval.core.metrics import extract_text_from_prompt

# Default local model paths
DEFAULT_MINERU_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM"
# DEFAULT_VLM_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"
DEFAULT_VLM_PATH = "ApiCall"

# Metric name to output field mapping
METRIC_FIELDS = {
    'vqa': ['vqa_score'],
    'ocr': ['ocr_acc', 'ocr_ned'],
    'clip': ['clip_score'],
    'vlm': ['vlm_text_accuracy', 'vlm_text_ned', 'vlm_image_quality', 'vlm_faithfulness', 'vlm_overall'],
    'vlm_quality': ['VLM_printed_like', 'VLM_sharpness', 'VLM_OCR_friendly'],
    'hpsv3': ['hpsv3_score'],
    'aesthetic': ['aesthetic_score'],
}


def load_longtext_benchmark(benchmark_path: str) -> List[Dict]:
    """Load LongText-Bench data from jsonl file or directory."""
    samples = []
    benchmark_path = Path(benchmark_path)
    
    # If directory, scan all .jsonl files
    if benchmark_path.is_dir():
        jsonl_files = list(benchmark_path.glob('*.jsonl'))
    else:
        jsonl_files = [benchmark_path]
    
    for jsonl_file in jsonl_files:
        lang_prefix = 'zh' if 'zh' in jsonl_file.name else 'en'
        
        file_prefix = os.path.splitext(os.path.basename(jsonl_file))[0]
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt_id = item.get('prompt_id', len(samples))
                sample_id = f"longtext_{lang_prefix}_{prompt_id}"
                samples.append({
                    'id': sample_id,
                    '_source': file_prefix,
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
        # Determine language from filename
        lang_prefix = 'zh' if 'ZH' in os.path.basename(json_file) else 'en'
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            prompt = item.get('prompt_en') or item.get('prompt_cn') or item.get('prompt', '')
            sample_id = str(item.get('id', ''))
            # Match id format with run_parallel_benchmark.py: f"oneig_{lang_prefix}_{sample_id}"
            full_id = f"oneig_{lang_prefix}_{sample_id}"
            samples.append({
                'id': full_id,
                'prompt': prompt,
                'text': extract_text_from_prompt(prompt),
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
                    'text': extract_text_from_prompt(item.get('prompt', '')),
                    'area': area,
                    'benchmark_type': subdir,
                    'carrier_list': item.get('carrier_list', []),
                    'sentence_list': item.get('sentence_list', [])
                })
    
    return samples


def load_unseenwords_benchmark(benchmark_path: str) -> List[Dict]:
    """Load UnseenWords benchmark data from jsonl files.
    
    Each sample is tagged with '_source' = filename stem (e.g. 'unseen_en')
    so results can be split back into per-file outputs later.
    """
    import glob
    samples = []
    
    if os.path.isfile(benchmark_path):
        jsonl_files = [benchmark_path]
    else:
        jsonl_files = glob.glob(os.path.join(benchmark_path, '*.jsonl'))
    
    for jsonl_file in sorted(jsonl_files):
        file_prefix = os.path.splitext(os.path.basename(jsonl_file))[0]
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt_id = item.get('prompt_id', len(samples))
                sample_id = f"{file_prefix}_{prompt_id}"
                samples.append({
                    'id': sample_id,
                    '_source': file_prefix,
                    'prompt': item.get('prompt', ''),
                    'text': item.get('text', []),
                    'category': item.get('category', ''),
                    'length': item.get('length', ''),
                    'text_length': item.get('text_length', 0)
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
    elif benchmark_type == 'unseenwords':
        return load_unseenwords_benchmark(benchmark_path)
    else:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [{'id': item.get('id', str(i)), **item} for i, item in enumerate(data)]
        return []


def find_result_image(results_dir: str, sample_id: str) -> Optional[str]:
    """Find result image for a sample. Returns None if not found."""
    results_dir = Path(results_dir)
    
    # Exact match: {id}.ext
    for ext in ['.png', '.jpg', '.jpeg']:
        path = results_dir / f"{sample_id}{ext}"
        if path.exists():
            return str(path)
    
    # run_parallel_benchmark convention: result_{id}.ext
    for ext in ['.png', '.jpg', '.jpeg']:
        path = results_dir / f"result_{sample_id}{ext}"
        if path.exists():
            return str(path)
    
    # Pattern matching fallback
    for pattern in [f"result_*_{sample_id}_*.png", f"result_{sample_id}_*.png", f"*{sample_id}*.png"]:
        matches = list(results_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None


def load_existing_results(output_path: str) -> Dict[str, Dict]:
    """Load existing results from output file as a dict mapping id -> result dict."""
    existing = {}
    if not os.path.exists(output_path):
        return existing
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'id' in data:
                    existing[data['id']] = data
            except json.JSONDecodeError:
                continue
    
    return existing


def get_samples_needing_metric(dataset: List[Dict], existing_results: Dict[str, Dict], metric: str) -> List[Dict]:
    """Get samples that don't have the specified metric computed yet."""
    fields = METRIC_FIELDS.get(metric, [])
    if not fields:
        return dataset
    
    samples_to_eval = []
    for sample in dataset:
        sample_id = sample['id']
        if sample_id not in existing_results:
            # Sample not in results at all, need to evaluate
            samples_to_eval.append(sample)
        else:
            # Check if all metric fields exist
            result = existing_results[sample_id]
            if not all(field in result for field in fields):
                samples_to_eval.append(sample)
    
    return samples_to_eval


def update_result_in_file(output_path: str, sample_id: str, updates: Dict):
    """Update a specific sample's result in the JSONL file."""
    # Read all results
    results = load_existing_results(output_path)
    
    # Update or create entry
    if sample_id in results:
        results[sample_id].update(updates)
    else:
        results[sample_id] = {'id': sample_id, **updates}
    
    # Write back all results (atomic write)
    temp_path = output_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        for result in results.values():
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    os.replace(temp_path, output_path)


def append_or_update_result(output_path: str, sample_id: str, updates: Dict, lock_file: str):
    """Append or update a result with file locking for multi-process safety."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Use a separate lock file for coordination
    with open(lock_file, 'a') as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            # Read existing results
            results = {}
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if 'id' in data:
                                    results[data['id']] = data
                            except json.JSONDecodeError:
                                continue
            
            # Update or create entry
            if sample_id in results:
                results[sample_id].update(updates)
            else:
                results[sample_id] = {'id': sample_id, **updates}
            
            # Write back all results
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results.values():
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def worker_fn_single_metric(rank: int, world_size: int, args, dataset: List[Dict], 
                            output_path: str, metric: str, existing_results: Dict[str, Dict]):
    """Worker function for evaluating a single metric across assigned samples."""
    import time
    time.sleep(rank * 0.5)
    device = f"cuda:{rank}"
    
    # Data partitioning
    total = len(dataset)
    per_gpu = math.ceil(total / world_size)
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, total)
    my_dataset = dataset[start_idx:end_idx]
    
    if not my_dataset:
        print(f"[GPU {rank}] No samples assigned for metric '{metric}'")
        return
    
    print(f"[GPU {rank}] Evaluating metric '{metric}' on {len(my_dataset)} samples")
    
    # Initialize only the needed evaluator
    from eval.core.metrics import OCRMetrics, CLIPMetrics, VLMMetrics, VQAScoreMetrics, AestheticScoreMetrics, HPSv3Metrics
    
    evaluator = None
    if metric == 'vqa':
        evaluator = VQAScoreMetrics(device=device)
    elif metric == 'ocr':
        evaluator = OCRMetrics(model_path=args.mineru_path)
    elif metric == 'clip':
        evaluator = CLIPMetrics(device=device)
    elif metric in ('vlm', 'vlm_quality'):
        evaluator = VLMMetrics(model_path=args.vlm_path, device=device)
    elif metric == 'hpsv3':
        evaluator = HPSv3Metrics(device=device)
    elif metric == 'aesthetic':
        evaluator = AestheticScoreMetrics(device=device)
    
    if evaluator is None:
        print(f"[GPU {rank}] Unknown metric: {metric}")
        return
    
    lock_file = output_path + '.lock'
    
    for sample in my_dataset:
        sample_id = sample['id']
        # Multi-dir mode: use per-sample dir & original id for image lookup
        sample_results_dir = sample.get('_results_dir', args.results_dir)
        original_id = sample.get('_original_id', sample_id)
        
        image_path = find_result_image(sample_results_dir, original_id)
        if not image_path:
            updates = {'error': 'Image not found'}
            append_or_update_result(output_path, sample_id, updates, lock_file)
            continue
        
        updates = {
            'prompt': sample['prompt'],
            'category': sample.get('category', ''),
            'image_path': image_path,
        }
        
        eval_prompt = sample['prompt']
        
        try:
            if metric == 'vqa':
                score = evaluator.compute_score(image_path, eval_prompt)
                updates['vqa_score'] = round(float(score), 4)
            
            elif metric == 'ocr':
                image = Image.open(image_path).convert('RGB')
                gt_text = sample.get('text', [])
                gt_text = ' '.join(gt_text) if isinstance(gt_text, list) else sample['prompt']
                ocr_metrics = evaluator.compute_accuracy(image, gt_text)
                updates['ocr_acc'] = round(ocr_metrics['ocr_acc'], 4)
                updates['ocr_ned'] = round(ocr_metrics['ocr_ned'], 4)
            
            elif metric == 'clip':
                score = evaluator.compute_clip_score(image_path, eval_prompt)
                updates['clip_score'] = round(float(score), 2)
            
            elif metric == 'vlm':
                image = Image.open(image_path).convert('RGB')
                vlm_result = evaluator.evaluate_text_rendering(image, sample['prompt'])
                updates['vlm_text_accuracy'] = round(vlm_result['text_accuracy'], 4)
                updates['vlm_text_ned'] = round(vlm_result['text_ned'], 4)
                updates['vlm_image_quality'] = round(vlm_result['image_quality'], 4)
                updates['vlm_faithfulness'] = round(vlm_result['faithfulness'], 4)
                updates['vlm_overall'] = round(vlm_result['overall'], 4)
            
            elif metric == 'vlm_quality':
                image = Image.open(image_path).convert('RGB')
                vlm_q = evaluator.evaluate_vlm_quality(image)
                updates['VLM_printed_like'] = round(vlm_q['VLM_printed_like'], 4)
                updates['VLM_sharpness'] = round(vlm_q['VLM_sharpness'], 4)
                updates['VLM_OCR_friendly'] = round(vlm_q['VLM_OCR_friendly'], 4)
            
            elif metric == 'hpsv3':
                score = evaluator.compute_score(image_path, eval_prompt)
                updates['hpsv3_score'] = round(float(score), 4)
            
            elif metric == 'aesthetic':
                score = evaluator.compute_score(image_path)
                updates['aesthetic_score'] = round(float(score), 4)
        
        except Exception as e:
            updates[f'{metric}_error'] = str(e)
        
        # Write result immediately
        append_or_update_result(output_path, sample_id, updates, lock_file)
        
        if args.verbose:
            print(f"[GPU {rank}] {metric}: {sample_id}")
    
    # Cleanup evaluator to free GPU memory
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[GPU {rank}] Completed metric '{metric}', GPU memory released")


def run_metric_evaluation(args, dataset: List[Dict], output_path: str, metric: str):
    """Run evaluation for a single metric across all GPUs."""
    print(f"\n{'='*60}")
    print(f"Evaluating metric: {metric}")
    print(f"{'='*60}")
    
    # Load existing results for resume
    existing_results = load_existing_results(output_path) if args.resume else {}
    
    # Filter samples that need this metric
    samples_to_eval = get_samples_needing_metric(dataset, existing_results, metric)
    
    if not samples_to_eval:
        print(f"All samples already have metric '{metric}' computed. Skipping.")
        return
    
    print(f"Samples to evaluate: {len(samples_to_eval)} / {len(dataset)}")
    
    # Determine number of GPUs to use
    num_gpus = min(args.gpus, len(samples_to_eval))
    if num_gpus < 1:
        num_gpus = 1
    
    print(f"Using {num_gpus} GPUs for metric '{metric}'")
    
    # Launch parallel workers
    if num_gpus > 1:
        mp.spawn(
            worker_fn_single_metric,
            args=(num_gpus, args, samples_to_eval, output_path, metric, existing_results),
            nprocs=num_gpus,
            join=True
        )
    else:
        # Single GPU mode
        worker_fn_single_metric(0, 1, args, samples_to_eval, output_path, metric, existing_results)
    
    # Force GPU memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Metric '{metric}' evaluation completed. GPU memory released.")


def compute_summary(output_path: str) -> Dict:
    """Compute summary statistics from output file."""
    results = list(load_existing_results(output_path).values())
    
    if not results:
        return {}
    
    summary = {'total_evaluated': len(results)}
    
    # OCR summary
    ocr_acc_scores = [r['ocr_acc'] for r in results if 'ocr_acc' in r]
    ocr_ned_scores = [r['ocr_ned'] for r in results if 'ocr_ned' in r]
    
    if ocr_acc_scores:
        summary['ocr_acc'] = {
            'mean': round(sum(ocr_acc_scores) / len(ocr_acc_scores), 4),
            'min': round(min(ocr_acc_scores), 4),
            'max': round(max(ocr_acc_scores), 4),
            'count': len(ocr_acc_scores)
        }
    if ocr_ned_scores:
        summary['ocr_ned'] = {
            'mean': round(sum(ocr_ned_scores) / len(ocr_ned_scores), 4),
            'min': round(min(ocr_ned_scores), 4),
            'max': round(max(ocr_ned_scores), 4),
            'count': len(ocr_ned_scores)
        }
    
    # CLIP summary
    clip_scores = [r['clip_score'] for r in results if 'clip_score' in r]
    if clip_scores:
        summary['clip'] = {
            'mean': round(sum(clip_scores) / len(clip_scores), 2),
            'min': round(min(clip_scores), 2),
            'max': round(max(clip_scores), 2),
            'count': len(clip_scores)
        }
    
    # VLM summary
    vlm_overall_scores = [r['vlm_overall'] for r in results if 'vlm_overall' in r]
    vlm_text_acc_scores = [r['vlm_text_accuracy'] for r in results if 'vlm_text_accuracy' in r]
    vlm_text_ned_scores = [r['vlm_text_ned'] for r in results if 'vlm_text_ned' in r]
    vlm_quality_scores = [r['vlm_image_quality'] for r in results if 'vlm_image_quality' in r]
    vlm_faith_scores = [r['vlm_faithfulness'] for r in results if 'vlm_faithfulness' in r]

    for key, scores in [
        ('vlm_overall', vlm_overall_scores),
        ('vlm_text_accuracy', vlm_text_acc_scores),
        ('vlm_text_ned', vlm_text_ned_scores),
        ('vlm_image_quality', vlm_quality_scores),
        ('vlm_faithfulness', vlm_faith_scores),
    ]:
        if scores:
            summary[key] = {
                'mean': round(sum(scores) / len(scores), 4),
                'min': round(min(scores), 4),
                'max': round(max(scores), 4),
                'count': len(scores),
            }

    # VLM quality summary (printed_like / sharpness / OCR_friendly)
    for vq_key in ['VLM_printed_like', 'VLM_sharpness', 'VLM_OCR_friendly']:
        vals = [r[vq_key] for r in results if vq_key in r]
        if vals:
            summary[vq_key] = {
                'mean': round(sum(vals) / len(vals), 4),
                'min': round(min(vals), 4),
                'max': round(max(vals), 4),
                'count': len(vals),
            }

    # VQA summary
    vqa_scores = [r['vqa_score'] for r in results if 'vqa_score' in r]
    if vqa_scores:
        summary['vqa'] = {
            'mean': round(sum(vqa_scores) / len(vqa_scores), 4),
            'min': round(min(vqa_scores), 4),
            'max': round(max(vqa_scores), 4),
            'count': len(vqa_scores)
        }

    # HPSv3 summary
    hpsv3_scores = [r['hpsv3_score'] for r in results if 'hpsv3_score' in r]
    if hpsv3_scores:
        summary['hpsv3'] = {
            'mean': round(sum(hpsv3_scores) / len(hpsv3_scores), 4),
            'min': round(min(hpsv3_scores), 4),
            'max': round(max(hpsv3_scores), 4),
            'count': len(hpsv3_scores)
        }

    # Aesthetic summary
    aesthetic_scores = [r['aesthetic_score'] for r in results if 'aesthetic_score' in r]
    if aesthetic_scores:
        summary['aesthetic'] = {
            'mean': round(sum(aesthetic_scores) / len(aesthetic_scores), 4),
            'min': round(min(aesthetic_scores), 4),
            'max': round(max(aesthetic_scores), 4),
            'count': len(aesthetic_scores)
        }
    
    # Category-wise breakdown
    categories = {}
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    summary['by_category'] = {}
    for cat, cat_results in categories.items():
        cat_summary = {'count': len(cat_results)}
        
        cat_ocr_acc = [r['ocr_acc'] for r in cat_results if 'ocr_acc' in r]
        cat_ocr_ned = [r['ocr_ned'] for r in cat_results if 'ocr_ned' in r]
        if cat_ocr_acc:
            cat_summary['ocr_acc_mean'] = round(sum(cat_ocr_acc) / len(cat_ocr_acc), 4)
        if cat_ocr_ned:
            cat_summary['ocr_ned_mean'] = round(sum(cat_ocr_ned) / len(cat_ocr_ned), 4)
        
        cat_clip = [r['clip_score'] for r in cat_results if 'clip_score' in r]
        if cat_clip:
            cat_summary['clip_mean'] = round(sum(cat_clip) / len(cat_clip), 2)
        
        cat_vqa = [r['vqa_score'] for r in cat_results if 'vqa_score' in r]
        if cat_vqa:
            cat_summary['vqa_mean'] = round(sum(cat_vqa) / len(cat_vqa), 4)
            
        for vlm_key in ['vlm_text_accuracy', 'vlm_text_ned', 'vlm_image_quality', 'vlm_faithfulness', 'vlm_overall',
                        'VLM_printed_like', 'VLM_sharpness', 'VLM_OCR_friendly', 'hpsv3_score']:
            vals = [r[vlm_key] for r in cat_results if vlm_key in r]
            if vals:
                cat_summary[f'{vlm_key}_mean'] = round(sum(vals) / len(vals), 4)

        cat_aesthetic = [r['aesthetic_score'] for r in cat_results if 'aesthetic_score' in r]
        if cat_aesthetic:
            cat_summary['aesthetic_mean'] = round(sum(cat_aesthetic) / len(cat_aesthetic), 4)
            
        summary['by_category'][cat] = cat_summary
    
    return summary


def split_results_by_source(output_path: str, dataset: List[Dict], split_dir: str):
    """Split merged results into per-source-file outputs.
    
    Uses the '_source' tag on each dataset sample to determine which output
    file a result belongs to. Output filenames match the original source
    (e.g. unseen_en.jsonl).
    
    If a split file already exists, new metric fields are merged into
    existing entries (keyed by 'id') so that multiple evaluation passes
    (e.g. regular metrics + hpsv3) accumulate rather than overwrite.
    """
    source_map: Dict[str, str] = {}
    for sample in dataset:
        source_map[sample['id']] = sample.get('_source', '_unknown')

    results = load_existing_results(output_path)
    buckets: Dict[str, list] = {}
    for sample_id, result in results.items():
        source = source_map.get(sample_id, '_unknown')
        buckets.setdefault(source, []).append(result)

    os.makedirs(split_dir, exist_ok=True)
    for source, entries in sorted(buckets.items()):
        out_path = os.path.join(split_dir, f"{source}.jsonl")

        existing: Dict[str, Dict] = {}
        if os.path.exists(out_path):
            existing = load_existing_results(out_path)

        for entry in entries:
            eid = entry.get('id', '')
            if eid in existing:
                existing[eid].update(entry)
            else:
                existing[eid] = entry

        with open(out_path, 'w', encoding='utf-8') as f:
            for entry in existing.values():
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        summary = compute_summary(out_path)
        summary_path = out_path.replace('.jsonl', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  {source}: {len(existing)} samples -> {out_path}")

    print(f"Split into {len(buckets)} files in {split_dir}")


def split_merged_to_per_dir(merged_output: str, dir_tags: Dict[str, str]):
    """Split a merged eval output back into per-results_dir output files.
    
    Args:
        merged_output: Path to the merged eval_detail.jsonl
        dir_tags: Mapping from dir_tag (e.g. 'd0') to results_dir path
    """
    results = load_existing_results(merged_output)

    # Group results by dir_tag (prefix before '::')
    buckets: Dict[str, list] = {tag: [] for tag in dir_tags}
    for sample_id, result in results.items():
        sep = sample_id.find('::')
        if sep == -1:
            continue
        tag = sample_id[:sep]
        if tag not in buckets:
            continue
        restored = dict(result)
        restored['id'] = sample_id[sep + 2:]
        buckets[tag].append(restored)

    for tag, entries in buckets.items():
        results_dir = dir_tags[tag]
        out_path = os.path.join(results_dir, 'eval_detail.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        summary = compute_summary(out_path)
        summary_path = out_path.replace('.jsonl', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  [{tag}] {len(entries)} samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Parallel evaluation with multi-GPU support (metric-by-metric)')
    
    dir_group = parser.add_mutually_exclusive_group(required=True)
    dir_group.add_argument('--results_dir', type=str, default=None,
                           help='Single directory containing generated images')
    dir_group.add_argument('--results_dirs', nargs='+', default=None,
                           help='Multiple results directories to evaluate together '
                                '(one metric load for all dirs)')
    
    parser.add_argument('--benchmark', type=str, required=True,
                       help='Path to benchmark file or directory')
    parser.add_argument('--benchmark_type', type=str, default='longtext',
                       choices=['longtext', 'oneig', 'cvtg', 'unseenwords', 'generic'],
                       help='Benchmark type')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSONL file path (required for single-dir mode)')
    parser.add_argument('--mineru_path', type=str, default=DEFAULT_MINERU_PATH,
                       help='Path to local MinerU VLM model')
    parser.add_argument('--vlm_path', type=str, default=DEFAULT_VLM_PATH,
                       help='Path to local Qwen2.5-VL model')
    parser.add_argument('--metrics', nargs='+', default=['vqa', 'ocr', 'clip'],
                       choices=['ocr', 'clip', 'vlm', 'vlm_quality', 'vqa', 'hpsv3', 'aesthetic'],
                       help='Metrics to compute (evaluated in order specified)')
    parser.add_argument('--gpus', type=int, default=8,
                       help='Number of GPUs to use')
    parser.add_argument('--resume', action='store_true',
                       help='Resume: skip samples that already have the metric computed')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress for each sample')
    parser.add_argument('--split_output_dir', type=str, default=None,
                       help='After evaluation, split merged results into per-source-file '
                            'outputs in this directory (e.g. unseen_en.jsonl, unseen_zh.jsonl)')
    
    args = parser.parse_args()

    multi_dir_mode = args.results_dirs is not None

    # Load benchmark once
    print("\nLoading benchmark...")
    base_dataset = load_benchmark(args.benchmark, args.benchmark_type)
    print(f"Loaded {len(base_dataset)} samples from benchmark")

    if multi_dir_mode:
        # ----- Multi-dir mode: merge all dirs into one dataset -----
        dir_tags: Dict[str, str] = {}
        merged_dataset: List[Dict] = []
        for i, rdir in enumerate(args.results_dirs):
            tag = f"d{i}"
            dir_tags[tag] = rdir
            for sample in base_dataset:
                merged = dict(sample)
                merged['_results_dir'] = rdir
                merged['_original_id'] = sample['id']
                merged['id'] = f"{tag}::{sample['id']}"
                merged_dataset.append(merged)

        # Merged output file: next to first results_dir or use --output
        if args.output:
            merged_output = args.output
        else:
            merged_output = os.path.join(
                os.path.dirname(args.results_dirs[0]),
                '_merged_eval_detail.jsonl')
        os.makedirs(os.path.dirname(merged_output) or '.', exist_ok=True)

        print("=" * 60)
        print("Parallel Evaluation - Multi-Dir Mode")
        print("=" * 60)
        for tag, rdir in dir_tags.items():
            print(f"  [{tag}] {rdir}")
        print(f"Benchmark: {args.benchmark} ({args.benchmark_type})")
        print(f"Samples per dir: {len(base_dataset)}, Total: {len(merged_dataset)}")
        print(f"Merged output: {merged_output}")
        print(f"Metrics: {args.metrics}")
        print(f"GPUs: {args.gpus}")
        print("=" * 60)

        if args.resume and os.path.exists(merged_output):
            existing = load_existing_results(merged_output)
            print(f"\nResume mode: Found {len(existing)} existing results")
            for metric in args.metrics:
                remaining = len(get_samples_needing_metric(merged_dataset, existing, metric))
                total = len(merged_dataset)
                print(f"  - {metric}: {total - remaining}/{total} done, {remaining} remaining")

        mp.set_start_method('spawn', force=True)

        for metric in args.metrics:
            run_metric_evaluation(args, merged_dataset, merged_output, metric)

        print("\n" + "=" * 60)
        print("All metrics completed! Splitting results to per-dir outputs...")
        print("=" * 60)
        split_merged_to_per_dir(merged_output, dir_tags)

    else:
        # ----- Single-dir mode (original behavior) -----
        if not args.output:
            parser.error("--output is required when using --results_dir")

        dataset = base_dataset

        print("=" * 60)
        print("Parallel Evaluation - Metric-by-Metric Mode")
        print("=" * 60)
        print(f"Results dir: {args.results_dir}")
        print(f"Benchmark: {args.benchmark}")
        print(f"Type: {args.benchmark_type}")
        print(f"Output: {args.output}")
        print(f"GPUs: {args.gpus}")
        print(f"Metrics (in order): {args.metrics}")
        print(f"Resume: {args.resume}")
        print("=" * 60)

        if args.resume and os.path.exists(args.output):
            existing = load_existing_results(args.output)
            print(f"\nResume mode: Found {len(existing)} existing results")
            for metric in args.metrics:
                remaining = len(get_samples_needing_metric(dataset, existing, metric))
                print(f"  - {metric}: {len(dataset) - remaining}/{len(dataset)} done, {remaining} remaining")

        mp.set_start_method('spawn', force=True)

        for metric in args.metrics:
            run_metric_evaluation(args, dataset, args.output, metric)

        print("\n" + "=" * 60)
        print("All metrics evaluation completed!")
        print("=" * 60)

        summary = compute_summary(args.output)
        summary_path = args.output.replace('.jsonl', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\nEvaluation Summary:")
        print(json.dumps(summary, indent=2))
        print(f"\nResults: {args.output}")
        print(f"Summary: {summary_path}")

        if args.split_output_dir:
            print(f"\nSplitting results by source into {args.split_output_dir} ...")
            split_results_by_source(args.output, dataset, args.split_output_dir)


if __name__ == '__main__':
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    main()
