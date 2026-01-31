#!/usr/bin/env python3
"""
VQAScore implementation using clip-flant5-xxl.
Aligned with t2v_metrics.VQAScore implementation.
"""

from typing import List, Union, Optional
import os

import torch
from PIL import Image
from transformers import AutoTokenizer

# Import from local clip_t5 module (same structure as t2v_metrics)
from .clip_t5.model import CLIPT5ForConditionalGeneration, ModelArguments
from .clip_t5.model.multimodal_encoder.clip_encoder import CLIPVisionTower

# ============================================================================
# Constants (aligned with t2v_metrics/constants.py)
# ============================================================================
HF_CACHE_DIR = None  # Use default HF cache
CONTEXT_LEN = 2048
SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

# ============================================================================
# Utility Functions
# ============================================================================
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def t5_tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def load_pretrained_model(model_cls,
                          model_args,
                          model_path=None,
                          tokenizer_path=None,
                          model_max_length=None,
                          padding_side=None,
                          image_aspect_ratio='pad',
                          mmprojector_repo=None,
                          mmprojector_name=None,
                          device='cuda',
                          cache_dir=HF_CACHE_DIR):
    """Load pretrained model (aligned with t2v_metrics/mm_utils.py)"""
    tokenizer_dict = {}
    if model_max_length:
        tokenizer_dict['model_max_length'] = model_max_length
    if padding_side:
        tokenizer_dict['padding_side'] = padding_side
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, **tokenizer_dict)

    # Load model
    model = model_cls.from_pretrained(model_path, cache_dir=cache_dir)
    
    if mmprojector_repo:
        from huggingface_hub import hf_hub_download
        model_base_name = mmprojector_repo.split('/')[-1]
        
        if cache_dir is not None:
            local_dir = os.path.join(cache_dir, model_base_name)
        elif os.environ.get('HF_HOME') is not None:
            local_dir = os.path.join(os.environ.get('HF_HOME'), model_base_name)
        else:
            local_dir = os.path.join(os.path.expanduser("~"), model_base_name)
        print(f"Downloading projector weights to {local_dir}")
        hf_hub_download(
            repo_id=mmprojector_repo,
            filename=mmprojector_name,
            local_dir=local_dir,
        )
        pretrain_mm_mlp_adapter = os.path.join(local_dir, mmprojector_name)
        model_args.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter
        
        model.get_model().initialize_vision_modules(model_args)
    else:
        model.resize_token_embeddings(len(tokenizer))

    # Load vision tower weights if not loaded yet
    if not model.get_vision_tower().is_loaded:
        model.get_vision_tower().load_model()
    
    model.to(device=device, dtype=torch.bfloat16)
    image_processor = model.get_vision_tower().image_processor

    model.requires_grad_(False)
    
    model.config.image_aspect_ratio = image_aspect_ratio
    model.config.use_cache = False
    model.config.image_grid_pinpoints = None
    model.config.freeze_mm_mlp_adapter = True

    model = model.eval()
    return tokenizer, model, image_processor


# ============================================================================
# Formatting Functions
# ============================================================================
default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"


def format_question(question, conversation_style='plain'):
    if conversation_style == 't5_plain':
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 't5_chat':
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system':
        question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 't5_chat_no_system_no_user':
        question = "" + DEFAULT_IMAGE_TOKEN + "\n" + question + " : "
    else:
        raise NotImplementedError()
    return question


def format_answer(answer, conversation_style='plain'):
    return answer


# ============================================================================
# Model Configurations (aligned with t2v_metrics/clip_t5_model.py)
# ============================================================================
CLIP_T5_MODELS = {
    'clip-flant5-xxl': {
        'tokenizer' : {
            'path': 'google/flan-t5-xxl',
            'model_max_length': CONTEXT_LEN,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xxl',
            'conversation': 't5_chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'clip-flant5-xl': {
        'tokenizer' : {
            'path': 'google/flan-t5-xl',
            'model_max_length': CONTEXT_LEN,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xl',
            'conversation': 't5_chat',
            'image_aspect_ratio': 'pad',
        },
    },
}


# ============================================================================
# VQAScore Class (aligned with t2v_metrics CLIPT5Model)
# ============================================================================
class VQAScore:
    """VQAScore wrapper for clip-flant5-xxl, aligned with t2v_metrics."""
    
    def __init__(self, model: str = 'clip-flant5-xxl', device: str = 'cuda', cache_dir: Optional[str] = None):
        """
        Initialize VQAScore with clip-flant5-xxl model.
        
        Args:
            model: Model name, must be 'clip-flant5-xxl' or 'clip-flant5-xl'
            device: Device to run on ('cuda' or 'cpu')
            cache_dir: HuggingFace cache directory
        """
        assert model in CLIP_T5_MODELS, f"Model {model} not supported. Use one of {list(CLIP_T5_MODELS.keys())}"
        
        self.model_name = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.cache_dir = cache_dir or HF_CACHE_DIR
        
        # Load model configuration
        model_config = CLIP_T5_MODELS[model]
        
        model_args = ModelArguments()
        model_max_length = model_config['tokenizer'].get('model_max_length', None)
        padding_side = model_config['tokenizer'].get('padding_side', None)
        mmprojector_repo = model_config['model'].get('mmprojector_repo', None)
        mmprojector_name = model_config['model'].get('mmprojector_name', None)
        
        self.image_aspect_ratio = model_config['model'].get('image_aspect_ratio', 'pad')
        self.conversational_style = model_config['model']['conversation']
        self.context_len = CONTEXT_LEN
        
        # Load model
        self.tokenizer, self.model, self.image_processor = load_pretrained_model(
            CLIPT5ForConditionalGeneration,
            model_args,
            model_path=model_config['model']['path'],
            tokenizer_path=model_config['tokenizer']['path'],
            model_max_length=model_max_length,
            padding_side=padding_side,
            image_aspect_ratio=self.image_aspect_ratio,
            mmprojector_repo=mmprojector_repo,
            mmprojector_name=mmprojector_name,
            device=self.device,
            cache_dir=self.cache_dir
        )

    def load_images(self, image_paths: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device"""
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            images.append(image)

        if self.image_aspect_ratio == 'pad':
            images = [expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in images]

        images = [self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in images]
        assert all(x.shape == images[0].shape for x in images)
        images = torch.stack(images, dim=0).to(self.device)
        return images

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, 
                images: Union[str, List[str]], 
                texts: Union[str, List[str]],
                question_template: str = default_question_template,
                answer_template: str = default_answer_template) -> torch.Tensor:
        """
        Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        if isinstance(images, str):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        assert len(images) == len(texts), "Number of images and texts must match"
        
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        # Formatting for CLIP-FlanT5 desired input including system message and image tokens
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]

        images_tensor = self.load_images(images)

        input_ids = [t5_tokenizer_image_token(qs, self.tokenizer, return_tensors='pt') for qs in questions]
        labels = [t5_tokenizer_image_token(ans, self.tokenizer, return_tensors='pt') for ans in answers]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        decoder_attention_mask = labels.ne(IGNORE_INDEX)

        input_ids, attention_mask, decoder_attention_mask, labels = input_ids.to(self.device), \
            attention_mask.to(self.device), decoder_attention_mask.to(self.device), labels.to(self.device)

        model_input_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'images': images_tensor,
            'past_key_values': None,
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': True,
        }

        outputs = self.model(**model_input_kwargs)

        logits = outputs.logits
        lm_prob = torch.zeros(logits.shape[0])
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp()

        return lm_prob

    def batch_forward(self,
                      images: List[str],
                      texts: List[str],
                      batch_size: int = 8) -> torch.Tensor:
        """
        Batch version of forward for efficiency.
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

    def __call__(self, images, texts):
        """Make callable like t2v_metrics.VQAScore."""
        return self.forward(images, texts)
