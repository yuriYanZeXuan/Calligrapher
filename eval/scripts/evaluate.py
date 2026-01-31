#!/usr/bin/env python3
"""
Unified evaluation script for text rendering tasks.

This is the main entry point for all evaluations in the Calligrapher project.
It supports both generation and editing evaluation modes.

Usage:
    # Generation evaluation (text-to-image)
    python eval/scripts/evaluate.py \\
        --mode generation \\
        --config eval/configs/generation_config.yaml
    
    # Editing evaluation (image editing with mask)
    python eval/scripts/evaluate.py \\
        --mode editing \\
        --config eval/configs/editing_config.yaml
    
    # Quick evaluation with minimal arguments
    python eval/scripts/evaluate.py \\
        --mode generation \\
        --benchmark eval/OneIG-Bench/OneIG-Bench.json \\
        --generated outputs/generation \\
        --metrics ocr clip
    
    python eval/scripts/evaluate.py \\
        --mode editing \\
        --benchmark eval/Calligrapher_bench_testing \\
        --generated outputs/editing \\
        --metrics ocr dino

Available Modes:
    - generation: Evaluate text-to-image generation
    - editing: Evaluate text rendering with masks (editing/inpainting)

Available Metrics:
    - ocr: Character-level OCR accuracy (requires MinerU)
    - clip: CLIP score for text-image alignment
    - dino: DINOv2 similarity (for editing with reference)
    - fid: Frechet Inception Distance (distribution quality)
    - vlm: Vision-Language Model evaluation (requires API_KEY)

Examples:
    # OneIG-Bench evaluation
    python eval/scripts/evaluate.py \\
        --mode generation \\
        --benchmark eval/OneIG-Bench/OneIG-Bench.json \\
        --benchmark_type oneig \\
        --generated outputs/oneig_results \\
        --output eval_results/oneig.json \\
        --metrics ocr clip
    
    # CVTG-2K evaluation
    python eval/scripts/evaluate.py \\
        --mode generation \\
        --benchmark eval/CVTG-2K \\
        --benchmark_type cvtg \\
        --generated outputs/cvtg_results \\
        --metrics ocr clip
    
    # Calligrapher editing evaluation
    python eval/scripts/evaluate.py \\
        --mode editing \\
        --benchmark eval/Calligrapher_bench_testing \\
        --benchmark_format directory \\
        --generated outputs/editing_results \\
        --output eval_results/editing.json \\
        --metrics ocr dino clip
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from eval.core import GenerationEvaluator, EditingEvaluator


def print_banner():
    """Print evaluation banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         Calligrapher Text Rendering Evaluation               ║
    ║                                                              ║
    ║  Unified evaluation framework for text rendering tasks       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_config(args) -> dict:
    """Build configuration from arguments and config file."""
    # Start with config file if provided
    if args.config and os.path.exists(args.config):
        config = load_yaml_config(args.config)
    else:
        config = {}
    
    # Override with command line arguments
    if args.benchmark:
        config.setdefault('benchmark', {})['path'] = args.benchmark
    if args.benchmark_type:
        config.setdefault('benchmark', {})['type'] = args.benchmark_type
    if args.benchmark_format:
        config.setdefault('benchmark', {})['format'] = args.benchmark_format
    if args.generated:
        config.setdefault('generation', {})['output_dir'] = args.generated
    if args.output:
        config.setdefault('output', {})['result_file'] = args.output
    if args.metrics:
        config['metrics'] = args.metrics
    if args.device:
        config.setdefault('device', {})['type'] = args.device
    if args.filter_categories:
        config.setdefault('benchmark', {})['filter_categories'] = args.filter_categories
    
    # Local model paths
    config['mineru_path'] = args.mineru_path
    config['vlm_path'] = args.vlm_path
    
    return config


def validate_config(config: dict, mode: str) -> bool:
    """Validate configuration."""
    if 'benchmark' not in config or 'path' not in config['benchmark']:
        print("Error: Benchmark path not specified!")
        return False
    
    if 'generation' not in config or 'output_dir' not in config['generation']:
        print("Error: Generated directory not specified!")
        return False
    
    benchmark_path = config['benchmark']['path']
    if not os.path.exists(benchmark_path):
        print(f"Error: Benchmark path does not exist: {benchmark_path}")
        return False
    
    generated_dir = config['generation']['output_dir']
    if not os.path.exists(generated_dir):
        print(f"Warning: Generated directory does not exist: {generated_dir}")
    
    return True


def run_generation_eval(config: dict):
    """Run generation evaluation."""
    print("\n[Mode] Text Rendering Generation Evaluation\n")
    
    # Setup evaluator
    evaluator_config = {
        'metrics': config.get('metrics', ['ocr', 'clip']),
        'device': config.get('device', {}).get('type', 'auto'),
        'benchmark_type': config.get('benchmark', {}).get('type', 'oneig'),
        'mineru_path': config.get('mineru_path'),
        'vlm_path': config.get('vlm_path'),
    }
    
    evaluator = GenerationEvaluator(evaluator_config)
    
    # Load benchmark
    benchmark_path = config['benchmark']['path']
    benchmark_data = evaluator.load_benchmark(benchmark_path)
    
    # Filter by categories
    filter_categories = config.get('benchmark', {}).get('filter_categories', [])
    if filter_categories:
        filtered_data = [s for s in benchmark_data if s.get('category') in filter_categories]
        print(f"Filtered to {len(filtered_data)} samples in categories: {filter_categories}")
        benchmark_data = filtered_data
    
    if not benchmark_data:
        print("No benchmark samples found!")
        return
    
    # Run evaluation
    generated_dir = config['generation']['output_dir']
    df = evaluator.evaluate_batch(benchmark_data, generated_dir)
    
    # Save results
    output_file = config.get('output', {}).get('result_file', 'eval_results/generation_results.json')
    summary = evaluator.compute_summary(df)
    evaluator.save_results(df, output_file, summary)
    
    return df, summary


def run_editing_eval(config: dict):
    """Run editing evaluation."""
    print("\n[Mode] Text Rendering Editing Evaluation\n")
    
    # Setup evaluator
    evaluator_config = {
        'metrics': config.get('metrics', ['ocr', 'dino']),
        'device': config.get('device', {}).get('type', 'auto'),
        'mask_required': config.get('mask', {}).get('required', True),
        'use_masked_metrics': config.get('mask', {}).get('use_masked_metrics', True),
        'benchmark_dir': config['benchmark']['path'],
        'mineru_path': config.get('mineru_path'),
        'vlm_path': config.get('vlm_path'),
    }
    
    evaluator = EditingEvaluator(evaluator_config)
    
    # Load benchmark
    benchmark_path = config['benchmark']['path']
    benchmark_data = evaluator.load_benchmark(benchmark_path)
    
    if not benchmark_data:
        print("No benchmark samples found!")
        return
    
    print(f"Loaded {len(benchmark_data)} samples from benchmark")
    
    # Run evaluation
    generated_dir = config['generation']['output_dir']
    df = evaluator.evaluate_batch(benchmark_data, generated_dir)
    
    if df.empty:
        print("No evaluation results generated!")
        return
    
    # Save results
    output_file = config.get('output', {}).get('result_file', 'eval_results/editing_results.json')
    summary = evaluator.compute_summary(df)
    evaluator.save_results(df, output_file, summary)
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description='Unified evaluation for text rendering tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main arguments
    parser.add_argument('--mode', type=str, required=True,
                        choices=['generation', 'editing'],
                        help='Evaluation mode: generation or editing')
    parser.add_argument('--config', type=str,
                        help='Path to configuration YAML file')
    
    # Benchmark settings
    parser.add_argument('--benchmark', type=str,
                        help='Path to benchmark file or directory')
    parser.add_argument('--benchmark_type', type=str,
                        help='Benchmark type (for generation mode)')
    parser.add_argument('--benchmark_format', type=str,
                        choices=['directory', 'txt', 'json'],
                        help='Benchmark format (for editing mode)')
    parser.add_argument('--filter_categories', nargs='+',
                        help='Filter by categories (generation mode only)')
    
    # Generation settings
    parser.add_argument('--generated', type=str,
                        help='Directory containing generated images')
    
    # Metrics
    parser.add_argument('--metrics', nargs='+',
                        choices=['ocr', 'clip', 'dino', 'fid', 'vlm', 'vqa', 'aesthetic'],
                        help='Metrics to compute')
    
    # Output settings
    parser.add_argument('--output', type=str,
                        help='Output file path for results')
    
    # Device settings
    parser.add_argument('--device', default="cuda",type=str,
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device for model inference')
    
    # Local model paths
    parser.add_argument('--mineru_path', type=str,
                        default='/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM',
                        help='Path to local MinerU VLM model for OCR')
    parser.add_argument('--vlm_path', type=str,
                        default='/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B',
                        help='Path to local VLM model (Qwen2.5-VL)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Build configuration
    config = build_config(args)
    
    # Validate configuration
    if not validate_config(config, args.mode):
        parser.print_help()
        return 1
    
    if args.mode == 'generation':
        run_generation_eval(config)
    elif args.mode == 'editing':
        run_editing_eval(config)
    
    print("\n✓ Evaluation completed successfully!")
    return 0
        


if __name__ == '__main__':
    sys.exit(main())
