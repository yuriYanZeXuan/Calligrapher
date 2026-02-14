#!/usr/bin/env python3
"""
Generation Evaluator for text-to-image generation tasks.

This evaluator handles benchmarks like:
- OneIG-Bench
- CVTG-2K
- LongText-Bench
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
import pandas as pd

from .base_evaluator import BaseEvaluator
from .metrics import OCRMetrics, DINOv2Metrics, CLIPMetrics, VLMMetrics, VQAScoreMetrics, AestheticScoreMetrics


class GenerationEvaluator(BaseEvaluator):
    """
    Evaluator for text rendering generation tasks.
    
    Supports benchmarks:
    - OneIG-Bench: Category-based text rendering prompts
    - CVTG-2K: CVTG and CVTG-Style with different area sizes
    - LongText-Bench: Long text generation
    - UnseenWords: Rare/unseen character rendering
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize generation evaluator.
        
        Args:
            config: Configuration dictionary containing:
                - metrics: List of metrics to compute ['ocr', 'clip', 'dino', 'vlm']
                - device: Device for model inference ('cuda' or 'cpu')
                - benchmark_type: Type of benchmark ('oneig', 'cvtg', 'longtext')
                - mineru_path: Path to local MinerU model
                - vlm_path: Path to local VLM model
        """
        super().__init__(config)
        
        self.metrics_config = config.get('metrics', ['ocr', 'clip'])
        self.device = config.get('device', 'cuda' if self._check_cuda() else 'cpu')
        self.benchmark_type = config.get('benchmark_type', 'oneig')
        self.mineru_path = config.get('mineru_path', 'opendatalab/MinerU2.5-2509-1.2B')
        self.vlm_path = config.get('vlm_path', 'Qwen/Qwen2.5-VL-7B-Instruct')
        
        # Initialize metrics
        self._init_metrics()
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _init_metrics(self):
        """Initialize metric calculators."""
        self.metrics = {}
        
        if 'ocr' in self.metrics_config:
            self.metrics['ocr'] = OCRMetrics(model_path=self.mineru_path)
        
        if 'dino' in self.metrics_config:
            self.metrics['dino'] = DINOv2Metrics(device=self.device)
        
        if 'clip' in self.metrics_config:
            self.metrics['clip'] = CLIPMetrics(device=self.device)
        
        if 'vlm' in self.metrics_config:
            self.metrics['vlm'] = VLMMetrics(model_path=self.vlm_path, device=self.device)
            
        if 'vqa' in self.metrics_config:
            self.metrics['vqa'] = VQAScoreMetrics(device=self.device)
            
        if 'aesthetic' in self.metrics_config:
            self.metrics['aesthetic'] = AestheticScoreMetrics(device=self.device)
    
    def load_benchmark(self, benchmark_path: str) -> List[Dict]:
        """
        Load benchmark data based on benchmark type.
        
        Args:
            benchmark_path: Path to benchmark file or directory
            
        Returns:
            List of benchmark samples
        """
        benchmark_path = Path(benchmark_path)
        
        if self.benchmark_type == 'oneig':
            return self._load_oneig_benchmark(benchmark_path)
        elif self.benchmark_type == 'cvtg':
            return self._load_cvtg_benchmark(benchmark_path)
        elif self.benchmark_type == 'longtext':
            return self._load_longtext_benchmark(benchmark_path)
        elif self.benchmark_type == 'unseen':
            return self._load_unseen_benchmark(benchmark_path)
        else:
            # Generic JSON loading
            return self._load_generic_benchmark(benchmark_path)
    
    def _load_oneig_benchmark(self, benchmark_path: Path) -> List[Dict]:
        """Load OneIG-Bench format data."""
        samples = []
        
        # Support both single file and directory
        if benchmark_path.is_file():
            json_files = [benchmark_path]
        else:
            json_files = list(benchmark_path.glob('*.json'))
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                # Support both English and Chinese prompts
                prompt = item.get('prompt_en') or item.get('prompt_cn') or item.get('prompt', '')
                samples.append({
                    'id': item.get('id', ''),
                    'prompt': prompt,
                    'category': item.get('category', ''),
                    'class': item.get('class', ''),
                    'source': json_file.stem  # Track source file
                })
        
        self.logger.info(f"Loaded {len(samples)} samples from OneIG-Bench")
        return samples
    
    def _load_cvtg_benchmark(self, benchmark_path: Path) -> List[Dict]:
        """Load CVTG-2K format data."""
        samples = []
        
        # CVTG-2K has structure: CVTG/2.json, CVTG/3.json, CVTG-Style/2.json, etc.
        for subdir in ['CVTG', 'CVTG-Style']:
            subdir_path = benchmark_path / subdir
            if not subdir_path.exists():
                continue
            
            for json_file in subdir_path.glob('*.json'):
                # Skip combined files
                if 'combined' in json_file.name:
                    continue
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                area = json_file.stem  # e.g., '2', '3', '4', '5'
                for item in data.get('data_list', []):
                    samples.append({
                        'id': f"{subdir}_{area}_{item.get('index', 0)}",
                        'prompt': item.get('prompt', ''),
                        'area': area,
                        'benchmark_type': subdir,
                        'carrier_list': item.get('carrier_list', []),
                        'sentence_list': item.get('sentence_list', [])
                    })
        
        self.logger.info(f"Loaded {len(samples)} samples from CVTG-2K")
        return samples
    
    def _load_longtext_benchmark(self, benchmark_path: Path) -> List[Dict]:
        """Load LongText-Bench format data."""
        samples = []
        
        # Support both .json and .jsonl files
        json_files = []
        if benchmark_path.is_file():
            json_files = [benchmark_path]
        else:
            json_files = list(benchmark_path.glob('*.json')) + list(benchmark_path.glob('*.jsonl'))
        
        for json_file in json_files:
            # Detect language prefix from filename
            lang_prefix = 'zh' if 'zh' in json_file.name else 'en'
            
            with open(json_file, 'r', encoding='utf-8') as f:
                if json_file.suffix == '.jsonl':
                    # JSONL format: one JSON object per line
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        prompt_id = item.get('prompt_id', len(samples))
                        sample_id = item.get('id') or f"longtext_{lang_prefix}_{prompt_id}"
                        samples.append({
                            'id': sample_id,
                            'prompt': item.get('prompt', ''),
                            'text': item.get('text', []),
                            'text_length': item.get('text_length', 0),
                            'category': item.get('category', ''),
                            'length': item.get('length', '')
                        })
                else:
                    # JSON format: array of objects
                    data = json.load(f)
                    for item in data:
                        prompt_id = item.get('prompt_id', len(samples))
                        sample_id = item.get('id') or f"longtext_{lang_prefix}_{prompt_id}"
                        samples.append({
                            'id': sample_id,
                            'prompt': item.get('prompt', ''),
                            'text': item.get('text', []),
                            'text_length': item.get('text_length', 0),
                            'category': item.get('category', ''),
                            'length': item.get('length', '')
                        })
        
        self.logger.info(f"Loaded {len(samples)} samples from LongText-Bench")
        return samples
    
    def _load_unseen_benchmark(self, benchmark_path: Path) -> List[Dict]:
        """Load UnseenWords benchmark format.
        
        Directory of JSONL files, each line: {category, length, prompt, text, text_length, prompt_id}
        Sample id follows run_parallel_benchmark convention: {file_prefix}_{prompt_id}
        Generated images named: result_{file_prefix}_{prompt_id}.png
        """
        samples = []
        
        if benchmark_path.is_file():
            jsonl_files = [benchmark_path]
        else:
            jsonl_files = sorted(benchmark_path.glob('*.jsonl'))
        
        for jsonl_file in jsonl_files:
            file_prefix = jsonl_file.stem  # e.g. "unseen_en"
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    prompt_id = item.get('prompt_id', len(samples))
                    text_list = item.get('text', [])
                    samples.append({
                        'id': f"{file_prefix}_{prompt_id}",
                        'prompt': item.get('prompt', ''),
                        'text': text_list,
                        'text_length': item.get('text_length', 0),
                        'category': item.get('category', ''),
                        'length': item.get('length', ''),
                        'source': file_prefix,
                    })
        
        self.logger.info(f"Loaded {len(samples)} samples from UnseenWords ({len(jsonl_files)} files)")
        return samples
    
    def _load_generic_benchmark(self, benchmark_path: Path) -> List[Dict]:
        """Load generic JSON benchmark format."""
        samples = []
        
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            for i, item in enumerate(data):
                samples.append({
                    'id': item.get('id', str(i)),
                    'prompt': item.get('prompt', item.get('prompt_en', item.get('prompt_cn', ''))),
                    **{k: v for k, v in item.items() if k not in ['id', 'prompt', 'prompt_en', 'prompt_cn']}
                })
        elif isinstance(data, dict) and 'data_list' in data:
            for item in data['data_list']:
                samples.append({
                    'id': str(item.get('index', 0)),
                    'prompt': item.get('prompt', ''),
                    **{k: v for k, v in item.items() if k not in ['index', 'prompt']}
                })
        
        self.logger.info(f"Loaded {len(samples)} samples from generic benchmark")
        return samples
    
    def _find_generated_image(self, sample_id: str, generated_dir: str) -> Optional[str]:
        """
        Find generated image for a sample.
        
        Supports various naming conventions:
        - {id}.png
        - result_{id}_*.png
        - {id}_*.png
        """
        generated_dir = Path(generated_dir)
        
        # Try exact match first
        for ext in ['.png', '.jpg', '.jpeg']:
            exact_path = generated_dir / f"{sample_id}{ext}"
            if exact_path.exists():
                return str(exact_path)
        
        # Try result_{id}.ext (run_parallel_benchmark convention)
        for ext in ['.png', '.jpg', '.jpeg']:
            result_path = generated_dir / f"result_{sample_id}{ext}"
            if result_path.exists():
                return str(result_path)
        
        # Try pattern matching
        for pattern in [
            f"result_*_{sample_id}_*.png",
            f"result_{sample_id}_*.png",
            f"{sample_id}_*.png",
            f"*{sample_id}*.png"
        ]:
            matches = list(generated_dir.glob(pattern))
            if matches:
                return str(matches[0])
        
        return None
    
    def evaluate_sample(self, sample: Dict, generated_dir: str) -> Optional[Dict]:
        """
        Evaluate a single generation sample.
        
        Args:
            sample: Benchmark sample with 'id' and 'prompt'
            generated_dir: Directory containing generated images
            
        Returns:
            Dictionary with evaluation results or None if evaluation fails
        """
        sample_id = sample.get('id', '')
        prompt = sample.get('prompt', '')
        
        # Find generated image
        image_path = self._find_generated_image(sample_id, generated_dir)
        if not image_path:
            self.logger.warning(f"Generated image not found for sample {sample_id}")
            return None
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
        
        # Build result dictionary
        result = {
            'id': sample_id,
            'prompt': prompt,
            'image_path': image_path,
            'category': sample.get('category', ''),
            'benchmark_type': sample.get('benchmark_type', self.benchmark_type),
            'length': sample.get('length', ''),
            'text_length': sample.get('text_length', 0),
            'source': sample.get('source', ''),
        }
        
        # Compute metrics
        if 'ocr' in self.metrics and self.metrics['ocr'].available:
            # Use 'text' field as ground truth for LongText-Bench, otherwise use prompt
            gt_text = sample.get('text')
            if gt_text and isinstance(gt_text, list):
                gt_text = ' '.join(gt_text)
            gt_text = gt_text or prompt
            ocr_metrics = self.metrics['ocr'].compute_accuracy(image, gt_text)
            result['ocr_acc'] = ocr_metrics['ocr_acc']
            result['ocr_ned'] = ocr_metrics['ocr_ned']
            result['ground_truth'] = gt_text
        
        if 'clip' in self.metrics and self.metrics['clip'].available:
            clip_score = self.metrics['clip'].compute_clip_score(image_path, prompt)
            result['clip_score'] = clip_score
            
        if 'vqa' in self.metrics and self.metrics['vqa'].available:
            vqa_score = self.metrics['vqa'].compute_score(image_path, prompt)
            result['vqa_score'] = vqa_score
            
        if 'aesthetic' in self.metrics and self.metrics['aesthetic'].available:
            aes_score = self.metrics['aesthetic'].compute_score(image_path)
            result['aesthetic_score'] = aes_score
        
        # DINO and VLM require reference images (not available in generation benchmarks)
        # These are skipped for pure generation tasks
        
        return result
    
    def evaluate_batch(self, benchmark_data: List[Dict], generated_dir: str) -> pd.DataFrame:
        """
        Evaluate all samples in batch.
        
        Args:
            benchmark_data: List of benchmark samples
            generated_dir: Directory containing generated images
            
        Returns:
            DataFrame with all evaluation results
        """
        df = super().evaluate_batch(benchmark_data, generated_dir)
        
        if len(df) == 0:
            return df
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Category-wise summary
        if 'category' in df.columns:
            non_empty = df[df['category'].notna() & (df['category'] != '')]
            if len(non_empty) > 0:
                self.logger.info("\n=== Category-wise Summary ===")
                for category in sorted(non_empty['category'].unique()):
                    cat_df = non_empty[non_empty['category'] == category]
                    self.logger.info(f"\nCategory: {category} ({len(cat_df)} samples)")
                    for col in numeric_cols:
                        mean_val = cat_df[col].mean()
                        self.logger.info(f"  {col}: {mean_val:.4f}")
        
        # Source-wise summary (useful for UnseenWords with multiple JSONL files)
        if 'source' in df.columns:
            non_empty = df[df['source'].notna() & (df['source'] != '')]
            if len(non_empty) > 0:
                self.logger.info("\n=== Source-wise Summary ===")
                for source in sorted(non_empty['source'].unique()):
                    src_df = non_empty[non_empty['source'] == source]
                    self.logger.info(f"\nSource: {source} ({len(src_df)} samples)")
                    for col in numeric_cols:
                        mean_val = src_df[col].mean()
                        self.logger.info(f"  {col}: {mean_val:.4f}")
        
        # Length-wise summary (useful for benchmarks with length categories)
        if 'length' in df.columns:
            non_empty = df[df['length'].notna() & (df['length'] != '')]
            if len(non_empty) > 0:
                self.logger.info("\n=== Length-wise Summary ===")
                for length in sorted(non_empty['length'].unique()):
                    len_df = non_empty[non_empty['length'] == length]
                    self.logger.info(f"\nLength: {length} ({len(len_df)} samples)")
                    for col in numeric_cols:
                        mean_val = len_df[col].mean()
                        self.logger.info(f"  {col}: {mean_val:.4f}")
        
        return df
