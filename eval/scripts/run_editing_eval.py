#!/usr/bin/env python3
"""
Script to run text rendering editing evaluation.

Usage:
    # Using config file
    python eval/scripts/run_editing_eval.py --config eval/configs/editing_config.yaml
    
    # Using command line arguments
    python eval/scripts/run_editing_eval.py \\
        --benchmark_path eval/Calligrapher_bench_testing \\
        --benchmark_format directory \\
        --generated_dir outputs/editing \\
        --output_file eval_results/editing_results.json \\
        --metrics ocr dino clip

    # With mask settings
    python eval/scripts/run_editing_eval.py \\
        --benchmark_path eval/benchmark.txt \\
        --benchmark_format txt \\
        --generated_dir outputs/editing \\
        --use_masked_metrics \\
        --metrics ocr dino
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from eval.core import EditingEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(args) -> dict:
    """Merge command line arguments with config file."""
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {}
    
    # Override with command line arguments
    if args.benchmark_path:
        config.setdefault('benchmark', {})['path'] = args.benchmark_path
    if args.benchmark_format:
        config.setdefault('benchmark', {})['format'] = args.benchmark_format
    if args.generated_dir:
        config.setdefault('generation', {})['output_dir'] = args.generated_dir
    if args.output_file:
        config.setdefault('output', {})['result_file'] = args.output_file
    if args.metrics:
        config['metrics'] = args.metrics
    if args.device:
        config.setdefault('device', {})['type'] = args.device
    if args.use_masked_metrics is not None:
        config.setdefault('mask', {})['use_masked_metrics'] = args.use_masked_metrics
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Run text rendering editing evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file
    parser.add_argument('--config', type=str,
                        help='Path to configuration YAML file')
    
    # Benchmark settings
    parser.add_argument('--benchmark_path', type=str,
                        help='Path to benchmark directory or file')
    parser.add_argument('--benchmark_format', type=str,
                        choices=['directory', 'txt', 'json'],
                        help='Format of benchmark')
    
    # Generation settings
    parser.add_argument('--generated_dir', type=str,
                        help='Directory containing generated images')
    
    # Metrics
    parser.add_argument('--metrics', nargs='+',
                        choices=['ocr', 'dino', 'clip', 'fid', 'vlm'],
                        help='Metrics to compute')
    
    # Mask settings
    parser.add_argument('--use_masked_metrics', action='store_true',
                        help='Use mask for metrics computation')
    parser.add_argument('--no_masked_metrics', dest='use_masked_metrics', action='store_false',
                        help='Do not use mask for metrics computation')
    
    # Output settings
    parser.add_argument('--output_file', type=str,
                        help='Output file path for results')
    
    # Device settings
    parser.add_argument('--device', type=str,
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device for model inference')
    
    args = parser.parse_args()
    
    # Check required arguments
    if not args.config:
        if not args.benchmark_path or not args.generated_dir:
            parser.error("Either --config or both --benchmark_path and --generated_dir must be provided")
    
    # Merge configuration
    config = merge_config(args)
    
    # Validate configuration
    if 'benchmark' not in config or 'path' not in config['benchmark']:
        parser.error("Benchmark path must be specified in config or via --benchmark_path")
    if 'generation' not in config or 'output_dir' not in config['generation']:
        parser.error("Generated directory must be specified in config or via --generated_dir")
    
    # Setup evaluator configuration
    evaluator_config = {
        'metrics': config.get('metrics', ['ocr', 'dino']),
        'device': config.get('device', {}).get('type', 'auto'),
        'mask_required': config.get('mask', {}).get('required', True),
        'use_masked_metrics': config.get('mask', {}).get('use_masked_metrics', True),
        'benchmark_dir': config['benchmark']['path'],
    }
    
    # Create evaluator
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
    
    # Compute summary
    summary = evaluator.compute_summary(df)
    
    # Save results
    output_file = config.get('output', {}).get('result_file', 'eval_results/editing_results.json')
    evaluator.save_results(df, output_file, summary)
    
    print("\n=== Evaluation Complete ===")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
