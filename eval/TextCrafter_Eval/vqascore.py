#!/usr/bin/env python3
"""
VQAScore wrapper using t2v_metrics library.
This is a thin wrapper around t2v_metrics.VQAScore for compatibility with existing code.
"""

from typing import List, Union

import torch
import t2v_metrics

# Global cache directory for all models
CACHE_DIR = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight"


class VQAScore:
    """
    VQAScore wrapper using t2v_metrics library.
    
    Usage:
        model = VQAScore(model='clip-flant5-xxl', device='cuda')
        score = model(images=[image_path], texts=[text])
        # or
        score = model.forward(images=[image_path], texts=[text])
    """
    
    def __init__(self, model: str = 'clip-flant5-xxl', device: str = 'cuda'):
        """
        Initialize VQAScore with specified model.
        
        Args:
            model: Model name, e.g. 'clip-flant5-xxl', 'clip-flant5-xl'
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize t2v_metrics VQAScore with global cache dir
        self._model = t2v_metrics.VQAScore(model=model, device=self.device, cache_dir=CACHE_DIR)
    
    def forward(self, 
                images: Union[str, List[str]], 
                texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass of the model to return n scores for n (image, text) pairs.
        
        Args:
            images: Single image path or list of image paths
            texts: Single text or list of texts
            
        Returns:
            torch.Tensor: Scores for each (image, text) pair
        """
        if isinstance(images, str):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        assert len(images) == len(texts), "Number of images and texts must match"
        
        # t2v_metrics returns M x N tensor for M images and N texts
        # We need diagonal (paired scores) when len(images) == len(texts)
        scores = self._model(images=images, texts=texts)
        
        # If scores is 2D (M x N matrix), extract diagonal for paired scores
        if scores.dim() == 2:
            return torch.diag(scores)
        return scores

    def batch_forward(self,
                      images: List[str],
                      texts: List[str],
                      batch_size: int = 8) -> torch.Tensor:
        """
        Batch version of forward for efficiency.
        
        Args:
            images: List of image paths
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            torch.Tensor: Concatenated scores for all pairs
        """
        all_scores = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_scores = self.forward(batch_images, batch_texts)
            all_scores.append(batch_scores)

        if not all_scores:
            return torch.tensor([]).to(self.device)

        return torch.cat(all_scores)

    def __call__(self, images: Union[str, List[str]], texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Make callable like t2v_metrics.VQAScore.
        
        Args:
            images: Single image path or list of image paths
            texts: Single text or list of texts
            
        Returns:
            torch.Tensor: Scores for each (image, text) pair
        """
        return self.forward(images, texts)
