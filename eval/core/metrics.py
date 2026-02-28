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

def extract_text_from_prompt(prompt: str) -> str:
        """Extract text content from prompt (text within quotes)."""
        import re
        # Extract text within quotes (English and Chinese quotes)
        # Supports: "text", 'text', “text”, ‘text’, 「text」, 『text』
        
        # Combined pattern for all quote types
        patterns = [
            r'"([^"]+)"',          # English double quotes
            r"'([^']+)'",          # English single quotes
            r'“([^”]+)”',          # Chinese double quotes
            r'‘([^’]+)’',          # Chinese single quotes
            r'「([^」]+)」',        # Corner brackets
            r'『([^』]+)』'         # Double corner brackets
        ]
        
        pattern = '|'.join(patterns)
        
        matches = re.findall(pattern, prompt)
        # findall returns tuples for groups, flatten and filter empty
        # Each match is a tuple where only one element is non-empty corresponding to the matched group
        quoted_texts = []
        for match in matches:
            for group in match:
                if group:
                    quoted_texts.append(group)
                    
        # print(f"Extracted quoted texts: {quoted_texts}")
        # print("============")
        if quoted_texts:
            return ' '.join(quoted_texts)
        # If no quoted text, return the full prompt
        return prompt

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

    Supports two backends:
      - model_path="ApiCall": 通过 OpenAI 兼容 API 调用远程 VLM（不占 GPU）
      - 其他路径: 加载本地 Qwen2.5-VL 模型
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.device = device
        self.available = False
        self.model = None
        self.processor = None
        self._api_client = None
        self._api_model = None

        if model_path == "ApiCall":
            self._init_api_backend()
        else:
            self._init_local_backend(model_path, device)

    # ---- backend initialisation ----

    def _init_api_backend(self):
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

        api_key = os.getenv("QST_API_KEY")
        base_url = os.getenv("QST_BASE_URL")
        if not api_key or not base_url:
            self.logger.error("QST_API_KEY / QST_BASE_URL not set – VLM ApiCall unavailable")
            return

        self._api_client = OpenAI(api_key=api_key, base_url=base_url)
        self._api_model = "qwen3-vl-235b-a22b-instruct"
        self.available = True
        self.logger.info(f"VLM ApiCall backend ready (model={self._api_model})")

    def _init_local_backend(self, model_path: str, device: str):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        self.logger.info(f"Loading VLM model from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device).eval()
        self.available = True
        self.logger.info(f"VLM model loaded on {device}")

    # ---- unified VLM call ----

    def _image_to_base64(self, image: Image.Image) -> str:
        import base64, io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_vlm(self, image: Image.Image, text_prompt: str, max_tokens: int = 512) -> str:
        """Unified VLM call: dispatches to API or local model."""
        if self._api_client is not None:
            return self._call_vlm_api(image, text_prompt, max_tokens)
        return self._call_vlm_local(image, text_prompt, max_tokens)

    def _call_vlm_api(self, image: Image.Image, text_prompt: str, max_tokens: int) -> str:
        import time
        b64 = self._image_to_base64(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": text_prompt},
            ],
        }]
        while True:
            try:
                resp = self._api_client.chat.completions.create(
                    model=self._api_model,
                    messages=messages,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                self.logger.warning(f"VLM API call failed: {e}, retrying in 30s...")
                time.sleep(30)

    def _call_vlm_local(self, image: Image.Image, text_prompt: str, max_tokens: int) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt},
        ]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_input], images=[image], return_tensors="pt").to(self.device)
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        generated_ids = outputs[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # ---- helper ----

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    @staticmethod
    def _compute_text_accuracy(ground_truth: str, recognized: str) -> Dict[str, float]:
        """Compute text accuracy using Levenshtein distance."""
        import Levenshtein

        gt_processed = ' '.join(ground_truth.lower().split())
        recognized_processed = ' '.join(recognized.lower().split())

        if not gt_processed:
            both_empty = not recognized_processed
            return {'text_accuracy': 1.0 if both_empty else 0.0, 'text_ned': 1.0 if both_empty else 0.0}

        distance = Levenshtein.distance(gt_processed, recognized_processed)

        text_acc = max(0.0, 1 - distance / len(gt_processed))
        max_len = max(len(gt_processed), len(recognized_processed))
        text_ned = max(0.0, 1 - distance / (max_len + 1e-5))

        return {'text_accuracy': text_acc, 'text_ned': text_ned}

    # ---- evaluation methods ----

    def evaluate_text_rendering(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Evaluate text rendering quality: text accuracy, image quality, faithfulness.

        Returns dict with keys:
            text_accuracy, text_ned, image_quality, faithfulness, overall,
            recognized_text, ground_truth
        """
        if not self.available:
            return {"text_accuracy": 0.0, "text_ned": 0.0, "image_quality": 0.0,
                    "faithfulness": 0.0, "overall": 0.0}

        image = self._prepare_image(image)
        ground_truth = extract_text_from_prompt(prompt)

        # --- 1. Text recognition ---
        text_prompt = (
            "Please read and output ALL the text content visible in this image.\n"
            "Only output the text you can see, nothing else. If there are multiple text elements, "
            "separate them with spaces.\n"
            "Do not add any explanations or descriptions, just the raw text content."
        )
        recognized_text = self._call_vlm(image, text_prompt, max_tokens=512)

        if (recognized_text.startswith('"') and recognized_text.endswith('"')) or \
           (recognized_text.startswith("'") and recognized_text.endswith("'")):
            recognized_text = recognized_text[1:-1]

        accuracy_result = self._compute_text_accuracy(ground_truth, recognized_text)
        text_score = max(0.0, min(1.0, float(accuracy_result['text_accuracy'])))
        text_ned = accuracy_result['text_ned']

        # --- 2. Image quality ---
        quality_prompt = (
            "Evaluate the overall quality of this image considering:\n"
            "1. Image clarity and sharpness\n"
            "2. Visual coherence and aesthetics\n"
            "3. Proper rendering of all elements\n\n"
            "Rate from 0-10, respond with only a number."
        )
        quality_response = self._call_vlm(image, quality_prompt, max_tokens=10)

        import re
        numbers = re.findall(r'\d+\.?\d*', quality_response)
        if numbers:
            raw_score = min(10.0, max(0.0, float(numbers[0])))
            quality_score = raw_score / 10.0
        else:
            quality_score = 0.0

        # --- 3. Faithfulness (prompt adherence) ---
        faithfulness_score = self._evaluate_faithfulness(image, prompt)

        overall = (text_score + quality_score + faithfulness_score) / 3

        return {
            "text_accuracy": text_score,
            "text_ned": text_ned,
            "image_quality": quality_score,
            "faithfulness": faithfulness_score,
            "overall": overall,
            "recognized_text": recognized_text,
            "ground_truth": ground_truth,
        }

    def _evaluate_faithfulness(self, image: Image.Image, prompt: str) -> float:
        """Evaluate how faithfully the image matches the prompt description.

        Asks the VLM to score scene composition, objects, style, and text placement
        adherence on a 0-10 scale, then normalises to [0, 1].
        """
        faithfulness_prompt = (
            "You are evaluating how faithfully this generated image matches its text prompt.\n\n"
            f"Prompt: \"{prompt}\"\n\n"
            "Consider the following aspects:\n"
            "1. Scene & background: Does the scene match the description?\n"
            "2. Objects & elements: Are all described objects/elements present?\n"
            "3. Style & color: Does the visual style match the prompt's intent?\n"
            "4. Text content & placement: Is the text rendered in the correct location with correct content?\n\n"
            "Rate the overall faithfulness from 0-10, respond with only a number."
        )
        response = self._call_vlm(image, faithfulness_prompt, max_tokens=10)

        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            raw = min(10.0, max(0.0, float(numbers[0])))
            return raw / 10.0
        return 0.0

    def evaluate_aesthetic(self, image: Image.Image) -> float:
        result = self.evaluate_text_rendering(image, "")
        return result.get("image_quality", 0.0)

    def evaluate_text_match(self, image: Image.Image, text: str) -> float:
        result = self.evaluate_text_rendering(image, text)
        return result.get("text_accuracy", 0.0), result.get("text_ned", 0.0)

    def evaluate_vlm_quality(self, image: Image.Image) -> Dict[str, float]:
        """Evaluate three VLM-based text rendering quality dimensions.

        Returns:
            Dict with keys: VLM_printed_like, VLM_sharpness, VLM_OCR_friendly
            Each score normalised to [0, 1].
        """
        if not self.available:
            return {"VLM_printed_like": 0.0, "VLM_sharpness": 0.0, "VLM_OCR_friendly": 0.0}

        image = self._prepare_image(image)

        prompt = (
            "You are evaluating the quality of text rendered inside a generated image.\n"
            "Rate the following three aspects independently on a 0-10 scale.\n\n"
            "1. Printed-like: Does the text look like professionally printed/typeset text "
            "(as opposed to hand-drawn, blurry, or distorted text)?\n"
            "2. Sharpness: Are the text edges crisp, clear, and free from artifacts, "
            "blur, or aliasing?\n"
            "3. OCR-friendly: Could an OCR engine reliably read the text? Consider "
            "character spacing, contrast against background, and absence of overlapping elements.\n\n"
            "Respond with EXACTLY three numbers separated by commas, nothing else. "
            "Example: 8, 7, 9"
        )

        response = self._call_vlm(image, prompt, max_tokens=30)

        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        scores = [0.0, 0.0, 0.0]
        for i, n in enumerate(numbers[:3]):
            scores[i] = min(10.0, max(0.0, float(n))) / 10.0

        return {
            "VLM_printed_like": scores[0],
            "VLM_sharpness": scores[1],
            "VLM_OCR_friendly": scores[2],
        }


class VQAScoreMetrics:
    """VQA Score metrics using local VQAScore implementation."""
    
    def __init__(self, model: str = 'clip-flant5-xxl', device: str = 'cuda'):
        """Initialize VQA Score metrics.
        
        Args:
            model: VQA model name (default: 'clip-flant5-xxl')
            device: Device for inference
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

class HPSv3Metrics:
    """HPSv3 (Human Preference Score v3) based on Qwen2-VL."""

    HPSV3_ROOT = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/HPSv3 "

    def __init__(self, device: str = "cuda",
                 config_path: str | None = None,
                 checkpoint_path: str | None = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device

        import sys
        if self.HPSV3_ROOT not in sys.path:
            sys.path.insert(0, self.HPSV3_ROOT)
        from hpsv3 import HPSv3RewardInferencer

        self.inferencer = HPSv3RewardInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.logger.info("HPSv3 initialized")

    def compute_score(self, image_path: str, prompt: str) -> float:
        """Return the HPSv3 mu score for one image-prompt pair."""
        rewards = self.inferencer.reward([prompt], [image_path])
        return float(rewards[0][0].item())

    def compute_batch(self, image_paths: List[str], prompts: List[str]) -> List[float]:
        rewards = self.inferencer.reward(prompts, image_paths)
        return [float(r[0].item()) for r in rewards]


def main():
    txt="阳光明媚的广场上挤满了热闹的户外集市，充满活力的购物人群在色彩斑斓的摊位间穿梭。在这个充满动感的市场场景中央，一个醒目的大型木质招牌悬挂在一个热门摊位上方，温暖的笔触清晰展示着“新鲜农场 当地土特产”的字样。在主标题下方，优雅简洁的标语鼓励性地写着“品尝最自然的农产品”，并附上诱人的优惠信息“特价：今日有机农产品九折！”。摊位周围艺术地散落着小型手写风格的黑板牌，清晰展示着吸引人的附加信息，如“提供显现的苹果，草莓，有机蔬菜”等，进一步吸引好奇的游客驻足。购物者们常驻足细读这些生动呈现的招牌文字，在热闹的集市氛围中增添了温暖与真实感。"
    img_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/samples/result_longtext_zh_12.png"
    # Test VLM metrics
    vlm = VLMMetrics(model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B", device="cuda")
    image = Image.open(img_path).convert('RGB')
    result = vlm.evaluate_text_rendering(image, txt)
    print("VLM Evaluation Result:")
    print(f"  Text Accuracy: {result.get('text_accuracy', 0.0):.4f}")
    print(f"  Image Quality: {result.get('image_quality', 0.0):.4f}")
    print(f"  Overall Score: {result.get('overall', 0.0):.4f}")
    print(f"  Text NED: {result.get('text_ned', 0.0):.4f}")

if __name__ == "__main__":
    main()