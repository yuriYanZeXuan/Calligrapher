#!/usr/bin/env python3
"""
Base Evaluator class for unified evaluation framework.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary containing evaluation settings
        """
        self.config = config
        self.logger = self._setup_logger()
        self.results = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the evaluator."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    @abstractmethod
    def load_benchmark(self, benchmark_path: str) -> List[Dict]:
        """
        Load benchmark data from file.
        
        Args:
            benchmark_path: Path to benchmark file or directory
            
        Returns:
            List of benchmark samples
        """
        pass
    
    @abstractmethod
    def evaluate_sample(self, sample: Dict, generated_path: str) -> Dict:
        """
        Evaluate a single sample.
        
        Args:
            sample: Benchmark sample data
            generated_path: Path to generated image
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    def evaluate_batch(self, benchmark_data: List[Dict], generated_dir: str) -> pd.DataFrame:
        """
        Evaluate all samples in batch.
        
        Args:
            benchmark_data: List of benchmark samples
            generated_dir: Directory containing generated images
            
        Returns:
            DataFrame containing all evaluation results
        """
        self.logger.info(f"Starting evaluation for {len(benchmark_data)} samples...")
        
        for sample in tqdm(benchmark_data, desc="Evaluating"):
            try:
                result = self.evaluate_sample(sample, generated_dir)
                if result:
                    self.results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating sample {sample.get('id', 'unknown')}: {e}")
                continue
        
        df = pd.DataFrame(self.results)
        self.logger.info(f"Evaluation completed. Processed {len(df)} samples.")
        return df
    
    def compute_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute summary statistics from results.
        
        Args:
            df: DataFrame containing evaluation results
            
        Returns:
            Dictionary containing summary statistics
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        summary = {}
        for col in numeric_cols:
            summary[f"mean_{col}"] = df[col].mean()
            summary[f"std_{col}"] = df[col].std()
        return summary
    
    def save_results(self, df: pd.DataFrame, output_path: str, summary: Optional[Dict] = None):
        """
        Save evaluation results to file.
        
        Args:
            df: DataFrame containing evaluation results
            output_path: Path to save results
            summary: Optional summary statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            results_dict = {
                'detailed_results': df.to_dict('records'),
                'summary': summary or self.compute_summary(df)
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
        else:
            # Default to CSV
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        self.logger.info(f"Results saved to {output_path}")
        
        # Print summary
        if summary:
            self.logger.info("\n=== Evaluation Summary ===")
            for key, value in summary.items():
                if isinstance(value, float):
                    self.logger.info(f"{key}: {value:.4f}")
                else:
                    self.logger.info(f"{key}: {value}")
