#!/usr/bin/env python3
"""
Script to run text rendering generation evaluation.

Usage:
    # Using config file
    python eval/scripts/run_generation_eval.py --config eval/configs/generation_config.yaml
    
    # Using command line arguments
    python eval/scripts/run_generation_eval.py \\
        --benchmark_path eval/OneIG-Bench/OneIG-Bench.json \\
        --benchmark_type oneig \\
        --generated_dir outputs/generation \\
        --output_file eval_results/generation_results.json \\
        --metrics ocr clip

    # Filter by category
    python eval/scripts/run_generation_eval.py \\
        --benchmark_path eval/OneIG-Bench/OneIG-Bench.json \\
        --filter_categories Text_Rendering \\
        --generated_dir outputs/generation
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from eval.core import GenerationEvaluator


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
    if args.benchmark_type:
        config.setdefault('benchmark', {})['type'] = args.benchmark_type
    if args.generated_dir:
        config.setdefault('generation', {})['output_dir'] = args.generated_dir
    if args.output_file:
        config.setdefault('output', {})['result_file'] = args.output_file
    if args.metrics:
        config['metrics'] = args.metrics
    if args.filter_categories:
        config.setdefault('benchmark', {})['filter_categories'] = args.filter_categories
    if args.device:
        config.setdefault('device', {})['type'] = args.device
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Run text rendering generation evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file
    parser.add_argument('--config', type=str,
                        help='Path to configuration YAML file')
    
    # Benchmark settings
    parser.add_argument('--benchmark_path', type=str,
                        help='Path to benchmark file or directory')
    parser.add_argument('--benchmark_type', type=str,
                        choices=['oneig', 'cvtg', 'longtext', 'generic'],
                        help='Type of benchmark')
    parser.add_argument('--filter_categories', nargs='+',
                        help='Categories to evaluate (e.g., Text_Rendering)')
    
    # Generation settings
    parser.add_argument('--generated_dir', type=str,
                        help='Directory containing generated images')
    
    # Metrics
    parser.add_argument('--metrics', nargs='+',
                        choices=['ocr', 'clip', 'dino', 'vlm'],
                        help='Metrics to compute')
    
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
        'metrics': config.get('metrics', ['ocr', 'clip']),
        'device': config.get('device', {}).get('type', 'auto'),
        'benchmark_type': config.get('benchmark', {}).get('type', 'oneig'),
    }
    
    # Create evaluator
    evaluator = GenerationEvaluator(evaluator_config)
    
    # Load benchmark
    benchmark_path = config['benchmark']['path']
    benchmark_data = evaluator.load_benchmark(benchmark_path)
    
    # Filter by categories if specified
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
    
    # Compute summary
    summary = evaluator.compute_summary(df)
    
    # Save results
    output_file = config.get('output', {}).get('result_file', 'eval_results/generation_results.json')
    evaluator.save_results(df, output_file, summary)
    
    print("\n=== Evaluation Complete ===")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
