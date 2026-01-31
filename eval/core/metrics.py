#!/usr/bin/env python3
"""
Unified metrics computation for text rendering evaluation.
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

# Optional imports - only needed for specific metrics
try:
    import torch
    from torchvision import transforms
    from torch.nn.functional import cosine_similarity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OCRMetrics:
    """OCR-based metrics for text accuracy evaluation."""
    
    def __init__(self, model_name: str = "opendatalab/MinerU2.5-2509-1.2B"):
        """Initialize OCR metrics with MinerU."""
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            from eval.bak.mineru_ocr import create_mineru_client, blocks_to_text
            self.ocr = create_mineru_client(model_name=model_name)
            self.blocks_to_text = blocks_to_text
            self.available = True
            self.logger.info("MinerU OCR initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MinerU OCR: {e}")
            self.available = False
    
    def compute_accuracy(self, image: Image.Image, ground_truth: str, mask: Optional[Image.Image] = None) -> float:
        """
        Compute character-level OCR accuracy.
        
        Args:
            image: Generated image
            ground_truth: Ground truth text
            mask: Optional mask for region-based evaluation
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not self.available:
            return 0.0
        
        import Levenshtein
        
        # Preprocess ground truth
        gt_processed = ground_truth.replace(" ", "").lower()
        
        # Apply mask if provided
        if mask:
            mask_resized = mask.resize(image.size, Image.NEAREST)
            img_np = np.array(image.convert('RGB'))
            mask_np = np.array(mask_resized.convert('L'))
            img_np[mask_np == 0] = 0
            image_to_ocr = Image.fromarray(img_np)
        else:
            image_to_ocr = image
        
        # Perform OCR
        try:
            blocks = self.ocr.two_step_extract(image_to_ocr)
            recognized_text = self.blocks_to_text(blocks)
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            recognized_text = ""
        
        recognized_processed = recognized_text.replace(" ", "").lower()
        
        # Compute accuracy
        if not gt_processed:
            return 1.0 if not recognized_processed else 0.0
        if not recognized_processed:
            return 0.0
        
        distance = Levenshtein.distance(gt_processed, recognized_processed)
        accuracy = 1 - (distance / len(gt_processed))
        return max(0.0, accuracy)
    
    def compute_word_accuracy(self, image_path: str, gt_words: List[str]) -> Dict[str, Any]:
        """
        Compute word-level accuracy metrics.
        
        Args:
            image_path: Path to generated image
            gt_words: List of ground truth words
            
        Returns:
            Dictionary with word accuracy metrics
        """
        if not self.available:
            return {'total_words': len(gt_words), 'correct_words': 0, 'word_accuracy': 0.0}
        
        try:
            image = Image.open(image_path).convert("RGB")
            blocks = self.ocr.two_step_extract(image)
            recognized_text = self.blocks_to_text(blocks)
            pred_words = recognized_text.lower().split()
            
            if not pred_words:
                pred_words = ['']
            
            correct = sum(1 for word in gt_words if word in pred_words)
            
            return {
                'total_words': len(gt_words),
                'correct_words': correct,
                'word_accuracy': correct / len(gt_words) if gt_words else 0.0
            }
        except Exception as e:
            self.logger.error(f"Word accuracy computation failed: {e}")
            return {'total_words': len(gt_words), 'correct_words': 0, 'word_accuracy': 0.0}


class DINOv2Metrics:
    """DINOv2-based feature similarity metrics."""
    
    def __init__(self, device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'):
        """Initialize DINOv2 metrics."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            self.available = True
            self.logger.info(f"DINOv2 initialized on {device}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize DINOv2: {e}")
            self.available = False
    
    @torch.no_grad()
    def get_embedding(self, image: Image.Image, mask: Optional[Image.Image] = None) -> torch.Tensor:
        """Get DINOv2 embedding for an image."""
        if not self.available:
            return torch.zeros(1, 768).to(self.device)
        
        if mask:
            mask_resized = mask.resize(image.size, Image.NEAREST)
            img_np = np.array(image)
            mask_np = np.array(mask_resized)
            img_np[mask_np == 0] = 0
            image = Image.fromarray(img_np)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(image_tensor)
        return embedding
    
    def compute_similarity(self, gen_image: Image.Image, ref_image: Image.Image, 
                          mask: Optional[Image.Image] = None) -> float:
        """
        Compute cosine similarity between generated and reference images.
        
        Args:
            gen_image: Generated image
            ref_image: Reference image
            mask: Optional mask for region-based evaluation
            
        Returns:
            Similarity score between -1 and 1
        """
        if not self.available:
            return 0.0
        
        gen_embedding = self.get_embedding(gen_image, mask)
        ref_embedding = self.get_embedding(ref_image)
        similarity = cosine_similarity(gen_embedding, ref_embedding).item()
        return similarity


class CLIPMetrics:
    """CLIP-based metrics for text-image alignment."""
    
    def __init__(self, device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'):
        """Initialize CLIP metrics."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.available = False
        
        try:
            import clip
            from sklearn.preprocessing import normalize
            self.model, self.preprocess = clip.load("ViT-L/14", device=device, jit=False)
            self.model.eval()
            self.available = True
            self.logger.info("CLIP initialized successfully")
        except ImportError:
            self.logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    
    def compute_clip_score(self, image_path: str, text: str) -> float:
        """
        Compute CLIP score for image-text alignment.
        
        Args:
            image_path: Path to image
            text: Text prompt
            
        Returns:
            CLIP score
        """
        if not self.available:
            return 0.0
        
        try:
            import clip
            from sklearn.preprocessing import normalize
            
            prefix = "A photo depicts "
            full_text = prefix + text
            
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([full_text], truncate=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).item()
                clip_score = 2.5 * max(similarity, 0)
            
            return clip_score
        except Exception as e:
            self.logger.error(f"CLIP score computation failed: {e}")
            return 0.0


class FIDMetrics:
    """FID (Frechet Inception Distance) metrics for distribution evaluation."""
    
    def __init__(self, device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'):
        """Initialize FID metrics."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.available = True  # FID uses basic torch operations
    
    def compute_statistics(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance statistics for FID.
        
        Args:
            images: List of image arrays
            
        Returns:
            Tuple of (mean, covariance)
        """
        features = np.array(images)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1: np.ndarray, sigma1: np.ndarray, 
                     mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """
        Calculate FID score between two distributions.
        
        Args:
            mu1, sigma1: Mean and covariance of first distribution
            mu2, sigma2: Mean and covariance of second distribution
            
        Returns:
            FID score (lower is better)
        """
        from scipy.linalg import sqrtm
        
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


class VLMMetrics:
    """Vision-Language Model based metrics for aesthetic and quality evaluation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize VLM metrics."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key or os.environ.get("API_KEY")
        self.available = self.api_key is not None
        
        if not self.available:
            self.logger.warning("API_KEY not set. VLM metrics will be skipped.")
    
    def evaluate_aesthetic(self, image: Image.Image, ref_image: Optional[Image.Image] = None) -> float:
        """
        Evaluate aesthetic quality using VLM.
        
        Args:
            image: Generated image
            ref_image: Optional reference image
            
        Returns:
            Aesthetic score
        """
        if not self.available:
            return 0.0
        
        # Placeholder for actual VLM implementation
        # This would call an API like GPT-4V, Qwen-VL, etc.
        self.logger.warning("VLM aesthetic evaluation not fully implemented")
        return 0.0
    
    def evaluate_text_match(self, image: Image.Image, text: str) -> float:
        """
        Evaluate text-image match using VLM.
        
        Args:
            image: Generated image
            text: Text prompt
            
        Returns:
            Text match score
        """
        if not self.available:
            return 0.0
        
        # Placeholder for actual VLM implementation
        self.logger.warning("VLM text match evaluation not fully implemented")
        return 0.0
