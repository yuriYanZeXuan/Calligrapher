#!/usr/bin/env python3
"""
Editing Evaluator for text rendering editing tasks.

This evaluator handles benchmarks like:
- Calligrapher_bench_testing (with source, mask, and reference images)
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

from .base_evaluator import BaseEvaluator
from .metrics import OCRMetrics, DINOv2Metrics, CLIPMetrics, FIDMetrics, VLMMetrics


class EditingEvaluator(BaseEvaluator):
    """
    Evaluator for text rendering editing tasks that require masks.
    
    This evaluator is designed for benchmarks where:
    - Source image: Original image with text region to be edited
    - Mask image: Binary mask indicating the text region
    - Reference image: Target style/appearance reference
    - Generated image: Model output after editing
    
    Supported benchmarks:
    - Calligrapher_bench_testing
    - Any editing benchmark with source/mask/ref structure
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize editing evaluator.
        
        Args:
            config: Configuration dictionary containing:
                - metrics: List of metrics to compute ['ocr', 'dino', 'fid', 'vlm', 'clip']
                - device: Device for model inference ('cuda' or 'cpu')
                - mask_required: Whether mask is required (default: True)
                - use_masked_metrics: Whether to apply mask in metric computation
                - mineru_path: Path to local MinerU model
                - vlm_path: Path to local VLM model
        """
        super().__init__(config)
        
        self.metrics_config = config.get('metrics', ['ocr', 'dino'])
        self.device = config.get('device', 'cuda' if self._check_cuda() else 'cpu')
        self.mask_required = config.get('mask_required', True)
        self.use_masked_metrics = config.get('use_masked_metrics', True)
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
        
        if 'fid' in self.metrics_config:
            self.metrics['fid'] = FIDMetrics(device=self.device)
        
        if 'vlm' in self.metrics_config:
            self.metrics['vlm'] = VLMMetrics(model_path=self.vlm_path, device=self.device)
    
    def load_benchmark(self, benchmark_path: str) -> List[Dict]:
        """
        Load benchmark data.
        
        Supports:
        - Directory with source/mask/ref images
        - TXT file listing samples
        - JSON file with sample metadata
        
        Args:
            benchmark_path: Path to benchmark directory or file
            
        Returns:
            List of benchmark samples
        """
        benchmark_path = Path(benchmark_path)
        
        if benchmark_path.is_file():
            if benchmark_path.suffix == '.txt':
                return self._load_txt_benchmark(benchmark_path)
            elif benchmark_path.suffix == '.json':
                return self._load_json_benchmark(benchmark_path)
            else:
                raise ValueError(f"Unsupported benchmark file format: {benchmark_path.suffix}")
        else:
            return self._load_directory_benchmark(benchmark_path)
    
    def _load_txt_benchmark(self, benchmark_file: Path) -> List[Dict]:
        """Load benchmark from TXT file (e.g., self_bench.txt)."""
        samples = []
        
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    samples.append({
                        'id': parts[0],
                        'source': parts[1],
                        'ref': parts[2],
                        'prompt': parts[3],
                        'mask': parts[2] if len(parts) < 5 else parts[4]  # Use ref as mask if not specified
                    })
        
        self.logger.info(f"Loaded {len(samples)} samples from TXT file")
        return samples
    
    def _load_json_benchmark(self, benchmark_file: Path) -> List[Dict]:
        """Load benchmark from JSON file."""
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        
        if isinstance(data, list):
            for item in data:
                samples.append({
                    'id': item.get('id', ''),
                    'source': item.get('source', ''),
                    'mask': item.get('mask', item.get('ref', '')),
                    'ref': item.get('ref', ''),
                    'prompt': item.get('prompt', item.get('text', ''))
                })
        elif isinstance(data, dict):
            # Handle nested structure
            for key, value in data.items():
                if isinstance(value, dict):
                    samples.append({
                        'id': key,
                        'source': value.get('source', ''),
                        'mask': value.get('mask', value.get('ref', '')),
                        'ref': value.get('ref', ''),
                        'prompt': value.get('prompt', value.get('text', ''))
                    })
        
        self.logger.info(f"Loaded {len(samples)} samples from JSON file")
        return samples
    
    def _load_directory_benchmark(self, benchmark_dir: Path) -> List[Dict]:
        """
        Load benchmark from directory structure.
        
        Expects files named like:
        - {id}_source.png
        - {id}_mask.png
        - {id}_ref.png
        """
        samples = []
        
        # Find all source files
        source_files = list(benchmark_dir.glob('*_source.png')) + \
                       list(benchmark_dir.glob('*_source.jpg'))
        
        for source_file in source_files:
            # Extract ID from filename
            match = re.match(r'(.+)_source\.(png|jpg)', source_file.name)
            if match:
                sample_id = match.group(1)
                mask_file = benchmark_dir / f"{sample_id}_mask.png"
                if not mask_file.exists():
                    mask_file = benchmark_dir / f"{sample_id}_mask.jpg"
                ref_file = benchmark_dir / f"{sample_id}_ref.png"
                if not ref_file.exists():
                    ref_file = benchmark_dir / f"{sample_id}_ref.jpg"
                
                # Try to find prompt file
                prompt_file = benchmark_dir / f"{sample_id}_prompt.txt"
                prompt = ""
                if prompt_file.exists():
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                
                samples.append({
                    'id': sample_id,
                    'source': str(source_file.relative_to(benchmark_dir)),
                    'mask': str(mask_file.relative_to(benchmark_dir)) if mask_file.exists() else '',
                    'ref': str(ref_file.relative_to(benchmark_dir)) if ref_file.exists() else '',
                    'prompt': prompt
                })
        
        self.logger.info(f"Loaded {len(samples)} samples from directory")
        return samples
    
    def parse_generated_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse generated filename to extract metadata.
        
        Supports patterns like:
        - result_{id}_{prompt}_{seed}.png
        - {id}_result.png
        - result_{id}.png
        """
        basename = os.path.basename(filename)
        
        # Pattern: result_{index}_{id}_{prompt}_{seed}.png
        patterns = [
            r'result_(\d+)_(test\d+)_(.*?)_(\d+)\.png',
            r'result_(test\d+)_(.*?)_(\d+)\.png',
            r'result_(\d+)_(.*?)_(\d+)\.png',
            r'(test\d+)_(\d+)\.png',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, basename)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return {
                        'ref_id': groups[-2] if len(groups) >= 3 else groups[0],
                        'prompt_safe': groups[-2] if len(groups) >= 3 else groups[-1],
                        'seed': groups[-1] if len(groups) >= 3 else '0'
                    }
        
        # Simple pattern: just extract ID from filename
        if 'result_' in basename:
            parts = basename.replace('.png', '').split('_')
            if len(parts) >= 2:
                return {
                    'ref_id': parts[1] if len(parts) > 1 else parts[0],
                    'prompt_safe': '',
                    'seed': '0'
                }
        
        return None
    
    def load_images_for_evaluation(self, generated_path: str, benchmark_dir: str) -> Tuple:
        """
        Load all necessary images for evaluation.
        
        Args:
            generated_path: Path to generated image
            benchmark_dir: Path to benchmark directory
            
        Returns:
            Tuple of (generated_img, source_img, mask_img, ref_img, metadata)
        """
        metadata = self.parse_generated_filename(generated_path)
        if not metadata:
            self.logger.warning(f"Could not parse filename: {generated_path}")
            return None, None, None, None, None
        
        ref_id = metadata['ref_id']
        benchmark_path = Path(benchmark_dir)
        
        # Try to find source, mask, ref files
        possible_patterns = [
            (f"{ref_id}_source.png", f"{ref_id}_mask.png", f"{ref_id}_ref.png"),
            (f"{ref_id}_source.jpg", f"{ref_id}_mask.jpg", f"{ref_id}_ref.jpg"),
            (f"source_{ref_id}.png", f"mask_{ref_id}.png", f"ref_{ref_id}.png"),
        ]
        
        source_path = mask_path = ref_path = None
        
        for src_pat, mask_pat, ref_pat in possible_patterns:
            if not source_path and (benchmark_path / src_pat).exists():
                source_path = benchmark_path / src_pat
            if not mask_path and (benchmark_path / mask_pat).exists():
                mask_path = benchmark_path / mask_pat
            if not ref_path and (benchmark_path / ref_pat).exists():
                ref_path = benchmark_path / ref_pat
        
        # Load images
        try:
            generated_img = Image.open(generated_path).convert("RGB")
            source_img = Image.open(source_path).convert("RGB") if source_path else None
            ref_img = Image.open(ref_path).convert("RGB") if ref_path else None
            
            if mask_path:
                mask_img = Image.open(mask_path).convert("L")
                # Ensure mask is binary
                mask_np = np.array(mask_img)
                mask_np[mask_np > 0] = 255
                mask_img = Image.fromarray(mask_np)
            else:
                mask_img = None
            
            return generated_img, source_img, mask_img, ref_img, metadata
        
        except Exception as e:
            self.logger.error(f"Error loading images for {ref_id}: {e}")
            return None, None, None, None, None
    
    def _find_generated_images(self, generated_dir: str) -> List[str]:
        """Find all generated images in directory."""
        generated_dir = Path(generated_dir)
        
        # Look for result_*.png files
        image_files = list(generated_dir.glob('result_*.png')) + \
                      list(generated_dir.glob('result_*.jpg'))
        
        return [str(f) for f in image_files]
    
    def evaluate_sample(self, sample: Dict, generated_dir: str) -> Optional[Dict]:
        """
        Evaluate a single editing sample.
        
        Note: For editing evaluation, we need the generated image path,
        so this method finds the appropriate generated image for the sample.
        """
        sample_id = sample.get('id', '')
        prompt = sample.get('prompt', '')
        benchmark_dir = sample.get('benchmark_dir', generated_dir)
        
        # Find generated image for this sample
        generated_images = self._find_generated_images(generated_dir)
        
        # Find matching image
        matching_image = None
        for img_path in generated_images:
            metadata = self.parse_generated_filename(img_path)
            if metadata and metadata['ref_id'] == sample_id:
                matching_image = img_path
                break
        
        if not matching_image:
            self.logger.warning(f"Generated image not found for sample {sample_id}")
            return None
        
        # Load images
        gen_img, source_img, mask_img, ref_img, _ = self.load_images_for_evaluation(
            matching_image, benchmark_dir
        )
        
        if gen_img is None:
            return None
        
        # Build result
        result = {
            'id': sample_id,
            'prompt': prompt,
            'image_path': matching_image,
        }
        
        # Compute metrics
        # OCR Accuracy (with mask if available)
        if 'ocr' in self.metrics and self.metrics['ocr'].available:
            ocr_mask = mask_img if self.use_masked_metrics else None
            ocr_metrics = self.metrics['ocr'].compute_accuracy(gen_img, prompt, mask=ocr_mask)
            result['ocr_acc'] = ocr_metrics['ocr_acc']
            result['ocr_ned'] = ocr_metrics['ocr_ned']
        
        # DINO Similarity (with mask if available)
        if 'dino' in self.metrics and self.metrics['dino'].available and ref_img:
            dino_mask = mask_img if self.use_masked_metrics else None
            dino_sim = self.metrics['dino'].compute_similarity(gen_img, ref_img, dino_mask)
            result['dino_similarity'] = dino_sim
        
        # CLIP Score
        if 'clip' in self.metrics and self.metrics['clip'].available:
            clip_score = self.metrics['clip'].compute_clip_score(matching_image, prompt)
            result['clip_score'] = clip_score
        
        # VLM Scores
        if 'vlm' in self.metrics and self.metrics['vlm'].available:
            if ref_img:
                aesthetic = self.metrics['vlm'].evaluate_aesthetic(gen_img, ref_img)
                result['vlm_aesthetic'] = aesthetic
            text_match = self.metrics['vlm'].evaluate_text_match(gen_img, prompt)
            result['vlm_text_match'] = text_match
        
        return result
    
    def evaluate_batch(self, benchmark_data: List[Dict], generated_dir: str) -> pd.DataFrame:
        """
        Evaluate all samples in batch.
        
        Also computes FID if requested and if benchmark_dir contains reference images.
        """
        # Add benchmark_dir to samples if not present
        benchmark_dir = self.config.get('benchmark_dir', generated_dir)
        for sample in benchmark_data:
            if 'benchmark_dir' not in sample:
                sample['benchmark_dir'] = benchmark_dir
        
        df = super().evaluate_batch(benchmark_data, generated_dir)
        
        # Compute FID if requested
        if 'fid' in self.metrics and self.metrics['fid'].available:
            self.logger.info("Computing FID score...")
            # FID computation requires feature extraction from all images
            # This is a placeholder for the actual FID implementation
            self.logger.warning("FID computation not yet implemented in batch mode")
        
        return df
