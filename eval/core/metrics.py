#!/usr/bin/env python3
"""
Unified metrics computation for text rendering evaluation.
"""

import os
from pathlib import Path
import sys
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

import torch
from torchvision import transforms
from torch.nn.functional import cosine_similarity

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OCRMetrics:
    """OCR-based metrics for text accuracy evaluation."""
    
    def __init__(self, model_path: str = "opendatalab/MinerU2.5-2509-1.2B"):
        """Initialize OCR metrics with MinerU.
        
        Args:
            model_path: Path to MinerU model (local path or HuggingFace model name)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        from eval.bak.mineru_ocr import create_mineru_client, blocks_to_text
        self.ocr = create_mineru_client(model_name=model_path)
        self.blocks_to_text = blocks_to_text
        self.available = True
        self.logger.info(f"MinerU OCR initialized with model: {model_path}")
    
    def compute_accuracy(self, image: Image.Image, ground_truth: str, mask: Optional[Image.Image] = None) -> Dict[str, float]:
        """
        Compute character-level OCR accuracy with two metrics.
        
        Args:
            image: Generated image
            ground_truth: Ground truth text
            mask: Optional mask for region-based evaluation
            
        Returns:
            Dictionary with two metrics:
            - 'ocr_acc': Accuracy normalized by ground truth length (recall-oriented)
            - 'ocr_ned': NED normalized by max length (symmetric similarity)
        """
        if not self.available:
            return {'ocr_acc': 0.0, 'ocr_ned': 0.0}
        
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
        recognized_text=""
        blocks = self.ocr.two_step_extract(image_to_ocr)
        recognized_text = self.blocks_to_text(blocks)
        # Log recognized text to file for debugging
        log_dir = Path("eval_logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "ocr_recognized_text.log"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] GT: {ground_truth} | Recognized: {recognized_text}\n")
        recognized_processed = recognized_text.replace(" ", "").lower()
        
        # Handle edge cases
        if not gt_processed:
            both_empty = not recognized_processed
            return {
                'ocr_acc': 1.0 if both_empty else 0.0,
                'ocr_ned': 1.0 if both_empty else 0.0
            }
        if not recognized_processed:
            return {'ocr_acc': 0.0, 'ocr_ned': 0.0}
        
        # Compute edit distance
        distance = Levenshtein.distance(gt_processed, recognized_processed)
        
        # OCR-Acc: Normalized by ground truth length (recall-oriented)
        # 衡量"应该生成的文本有多少被正确生成了"
        ocr_acc = 1 - (distance / len(gt_processed))
        ocr_acc = max(0.0, ocr_acc)
        
        # OCR-NED: Normalized by max length (symmetric similarity, F1-style)
        # 更对称的相似度度量，考虑了过度生成的惩罚
        max_len = max(len(gt_processed), len(recognized_processed))
        ocr_ned = 1 - (distance / (max_len + 1e-5))
        ocr_ned = max(0.0, ocr_ned)
        
        return {'ocr_acc': ocr_acc, 'ocr_ned': ocr_ned}
    
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

class DINOv2Metrics:
    """DINOv2-based feature similarity metrics."""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize DINOv2 metrics."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
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
    
    def __init__(self, device: str = 'cuda'):
        """Initialize CLIP metrics."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.available = False
        
        import clip
        from sklearn.preprocessing import normalize
        self.model, self.preprocess = clip.load("ViT-L/14", device=device, jit=False)
        self.model.eval()
        self.available = True
        self.logger.info("CLIP initialized successfully")
    
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


class FIDMetrics:
    """FID (Frechet Inception Distance) metrics for distribution evaluation."""
    
    def __init__(self, device: str = 'cuda'):
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
    """Vision-Language Model based metrics for text rendering quality evaluation.
    
    Uses local Qwen2.5-VL model for evaluation.
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        """Initialize VLM metrics with local model.
        
        Args:
            model_path: Path to local VLM model (e.g., /path/to/Qwen2.5-VL-7B)
            device: Device for inference ('auto', 'cuda', 'cpu')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.device = device
        self.available = False
        self.model = None
        self.processor = None
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        self.device = device
        
        self.logger.info(f"Loading VLM model from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device).eval()
        
        self.available = True
        self.logger.info(f"VLM model loaded on {device}")
    
    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for model input."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def evaluate_text_rendering(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Evaluate text rendering quality using VLM.
        
        Args:
            image: Generated image
            prompt: Original text prompt
            
        Returns:
            Dictionary with scores for text accuracy and image quality
        """
        if not self.available:
            return {"text_accuracy": 0.0, "image_quality": 0.0, "overall": 0.0}
        
        image = self._prepare_image(image)
        
        # Prompt for text accuracy evaluation
        text_prompt = f'''You are evaluating an AI-generated image. The prompt was: "{prompt}"

Please evaluate the text rendering in this image:
1. Are all the texts from the prompt correctly rendered?
2. Are there any spelling errors, missing text, or garbled text?
3. Rate the text rendering quality from 0-10, where 10 means perfect text rendering.

Respond with only a number from 0 to 10.'''
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_input], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10, temperature=0.1)
        
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Extract number from response
        import re
        numbers = re.findall(r'\d+', response)
        text_score = float(numbers[0]) / 10.0 if numbers else 0.5
        
        # Prompt for overall image quality
        quality_prompt = '''Evaluate the overall quality of this image considering:
1. Image clarity and sharpness
2. Visual coherence and aesthetics
3. Proper rendering of all elements

Rate from 0-10, respond with only a number.'''
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": quality_prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_input], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10, temperature=0.1)
        
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        import re
        numbers = re.findall(r'\d+', response)
        quality_score = float(numbers[0]) / 10.0 if numbers else 0.5
        
        return {
            "text_accuracy": text_score,
            "image_quality": quality_score,
            "overall": (text_score + quality_score) / 2
        }
    
    def evaluate_aesthetic(self, image: Image.Image) -> float:
        """Evaluate aesthetic quality using VLM.
        
        Args:
            image: Generated image
            
        Returns:
            Aesthetic score (0-1)
        """
        result = self.evaluate_text_rendering(image, "")
        return result.get("image_quality", 0.0)
    
    def evaluate_text_match(self, image: Image.Image, text: str) -> float:
        """Evaluate text-image match using VLM.
        
        Args:
            image: Generated image
            text: Text prompt
            
        Returns:
            Text match score (0-1)
        """
        result = self.evaluate_text_rendering(image, text)
        return result.get("text_accuracy", 0.0)


class VQAScoreMetrics:
    """VQA Score metrics using local VQAScore implementation."""
    
    def __init__(self, model: str = 'clip-flant5-xxl', device: str = 'cuda', cache_dir: Optional[str] = None):
        """Initialize VQA Score metrics.
        
        Args:
            model: VQA model name (default: 'clip-flant5-xxl')
            device: Device for inference
            cache_dir: Optional cache directory for model weights (not used, for compatibility)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.available = False
        
        # Import local VQAScore from TextCrafter_Eval
        from eval.TextCrafter_Eval.vqascore import VQAScore
        
        self.vqa_model = VQAScore(model=model, device=device)
        
        self.available = True
        self.logger.info(f"VQAScore initialized with model: {model}")
    
    def compute_score(self, image_path: str, text: str) -> float:
        """Compute VQA Score for image-text pair.
        
        Args:
            image_path: Path to image file
            text: Text prompt
            
        Returns:
            VQA score (typically 0-1)
        """
        if not self.available:
            return 0.0
        
        score = self.vqa_model(images=[image_path], texts=[text])
        return float(score.cpu().numpy().mean())
    
    def compute_batch(self, image_paths: List[str], texts: List[str]) -> List[float]:
        """Compute VQA Score for multiple image-text pairs.
        
        Args:
            image_paths: List of image file paths
            texts: List of text prompts
            
        Returns:
            List of VQA scores
        """
        if not self.available:
            return [0.0] * len(image_paths)
        
        scores = self.vqa_model(images=image_paths, texts=texts)
        return scores.cpu().numpy().tolist()


class AestheticScoreMetrics:
    """Aesthetic Score metrics using OpenCLIP + linear predictor."""
    
    def __init__(self, device: str = 'cuda', cache_dir: Optional[str] = None):
        """Initialize Aesthetic Score metrics.
        
        Args:
            device: Device for inference
            cache_dir: Optional cache directory for model weights
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.available = False
        
        import open_clip
        import torch.nn as nn
        
        # Get cache directory from environment or parameter
        if cache_dir is None:
            cache_dir = os.environ.get('HF_HOME', None)
        
        # Load OpenCLIP model
        if cache_dir:
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='openai', cache_dir=cache_dir)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='openai')
        
        model.to(self.device)
        model.eval()
        self.openclip_model = model
        self.preprocess = preprocess
        
        # Load aesthetic predictor
        aesthetic_model = self._load_aesthetic_predictor()
        if aesthetic_model:
            self.aesthetic_predictor = aesthetic_model
            self.available = True
            self.logger.info("Aesthetic Score initialized with OpenCLIP ViT-L-14")
        else:
            self.logger.warning("Aesthetic predictor model not found")
                
    
    def _load_aesthetic_predictor(self) -> Optional[torch.nn.Module]:
        """Load aesthetic predictor linear model.
        
        Returns:
            Loaded model or None if not found
        """
        import torch.nn as nn
        
        # Try to find model in multiple locations
        possible_paths = [
            # In TextCrafter_Eval directory
            Path(__file__).parent.parent / "TextCrafter_Eval" / "sa_0_4_vit_l_14_linear.pth",
            # In eval directory
            Path(__file__).parent.parent / "sa_0_4_vit_l_14_linear.pth",
            # In current directory
            Path("sa_0_4_vit_l_14_linear.pth"),
        ]
        
        for model_path in possible_paths:
            if model_path.exists():
                m = nn.Linear(768, 1)
                s = torch.load(model_path, map_location=self.device)
                m.load_state_dict(s)
                m.eval()
                m.to(self.device)
                self.logger.info(f"Loaded aesthetic predictor from: {model_path}")
                return m
        
        self.logger.warning("Aesthetic predictor model file not found in any expected location")
        return None
            
    
    def compute_score(self, image_path: str) -> float:
        """Compute aesthetic score for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Aesthetic score (typically 0-10)
        """
        if not self.available:
            return 0.0
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.openclip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_predictor(image_features)
        
        return float(prediction.cpu().numpy().item())
        
    
    def compute_batch(self, image_paths: List[str]) -> List[float]:
        """Compute aesthetic scores for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of aesthetic scores
        """
        if not self.available:
            return [0.0] * len(image_paths)
        
        scores = []
        for image_path in image_paths:
            score = self.compute_score(image_path)
            scores.append(score)
        
        return scores
