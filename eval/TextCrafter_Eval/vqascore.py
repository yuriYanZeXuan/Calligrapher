#!/usr/bin/env python3
"""
Minimal VQAScore implementation using clip-flant5-xxl.
Compatible with t2v_metrics.VQAScore interface.
"""

import torch
from PIL import Image
from typing import List, Union, Optional
import os


class VQAScore:
    """Minimal VQAScore wrapper for clip-flant5-xxl."""
    
    def __init__(self, model: str = 'clip-flant5-xxl', device: str = 'cuda', cache_dir: Optional[str] = None):
        """
        Initialize VQAScore with clip-flant5-xxl model.
        
        Args:
            model: Model name, must be 'clip-flant5-xxl' or 'clip-flant5-xl'
            device: Device to run on ('cuda' or 'cpu')
            cache_dir: HuggingFace cache directory (not used, for compatibility)
        """
        self.model_name = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Import here to avoid dependency issues if not used
        from transformers import T5Tokenizer, T5ForConditionalGeneration, CLIPProcessor, CLIPModel
        
        # Model configurations
        model_configs = {
            'clip-flant5-xxl': {
                't5_path': 'google/flan-t5-xxl',
                'clip_path': 'openai/clip-vit-large-patch14',
            },
            'clip-flant5-xl': {
                't5_path': 'google/flan-t5-xl', 
                'clip_path': 'openai/clip-vit-large-patch14',
            }
        }
        
        if model not in model_configs:
            raise ValueError(f"Model {model} not supported. Use 'clip-flant5-xxl' or 'clip-flant5-xl'")
        
        config = model_configs[model]
        
        # Load CLIP for image encoding
        self.clip_processor = CLIPProcessor.from_pretrained(config['clip_path'])
        self.clip_model = CLIPModel.from_pretrained(config['clip_path']).to(self.device).eval()
        
        # Load T5 for text generation
        self.tokenizer = T5Tokenizer.from_pretrained(config['t5_path'], model_max_length=2048)
        self.model = T5ForConditionalGeneration.from_pretrained(
            config['t5_path'],
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
        ).to(self.device).eval()
        
        # Templates
        self.question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self.answer_template = "Yes"
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
    
    def _format_question(self, text: str) -> str:
        """Format question with system message."""
        question = self.question_template.format(text)
        # t5_chat format
        formatted = f"USER: <image>\n{question} ASSISTANT: "
        return formatted
    
    @torch.no_grad()
    def forward(self, 
                images: Union[str, List[str]], 
                texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Compute VQA scores for image-text pairs.
        
        Args:
            images: Single image path or list of image paths
            texts: Single text or list of texts
            
        Returns:
            Tensor of scores (higher is better)
        """
        # Convert to lists
        if isinstance(images, str):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        assert len(images) == len(texts), "Number of images and texts must match"
        
        scores = torch.zeros(len(images))
        
        for i, (image_path, text) in enumerate(zip(images, texts)):
            # Load image features using CLIP
            pixel_values = self._load_image(image_path)
            clip_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            image_features = clip_outputs.last_hidden_state
            
            # Format question and answer
            question = self._format_question(text)
            answer = self.answer_template
            
            # Tokenize
            input_ids = self.tokenizer(
                question, 
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).input_ids.to(self.device)
            
            labels = self.tokenizer(
                answer,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).input_ids.to(self.device)
            
            # Compute log probability of answer given image+question
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                decoder_input_ids=self.model._shift_right(labels)
            )
            
            # Negative loss as score (higher is better)
            scores[i] = -outputs.loss.item()
        
        return scores.to(self.device)
    
    def batch_forward(self,
                      images: List[str],
                      texts: List[str],
                      batch_size: int = 8) -> torch.Tensor:
        """
        Batch version of forward for efficiency.
        
        Args:
            images: List of image paths
            texts: List of texts (must match length of images)
            batch_size: Batch size for processing
            
        Returns:
            Tensor of scores
        """
        all_scores = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_scores = self.forward(batch_images, batch_texts)
            all_scores.append(batch_scores)
        
        return torch.cat(all_scores)
    
    def __call__(self, images, texts):
        """Make callable like t2v_metrics.VQAScore."""
        return self.forward(images, texts)
