import torch
import torch.nn as nn
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    SiglipVisionModel,
    PretrainedConfig,
)
from diffusers import AutoencoderKL
# Conditional imports based on model_type will be handled in functions
# from flux_ip.transformer_flux_inpainting import FluxTransformer2DModel
# from flux_ip.attention_processor import IPAFluxAttnProcessor2_0

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    elif model_class == "CLIPTextModel":
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_text_encoders_and_tokenizers(args):
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )
    
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    
    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two

def load_vae_and_transformer(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    
    if args.model_type == 'flux':
        # Import the custom transformer model that supports IP-Adapter (image_emb).
        # from diffusers.models import FluxTransformer2DModel as TransformerModel
        from train.flux_ip.transformer_flux_inpainting import FluxTransformer2DModel as TransformerModel
    elif args.model_type == 'qwen':
        from qwen_ip.transformer import QwenTransformer2DModel as TransformerModel
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    transformer = TransformerModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    return vae, transformer

def load_image_encoder(args):
    image_encoder = SiglipVisionModel.from_pretrained(args.siglip_path)
    return image_encoder

class MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

def setup_ip_adapter(transformer, accelerator, weight_dtype, args):
    if args.model_type == 'flux':
        from flux_ip.attention_processor import IPAFluxAttnProcessor2_0 as IPAttentionProcessor
        num_tokens = 128
    elif args.model_type == 'qwen':
        from qwen_ip.attention_processor import IPAQwenAttnProcessor as IPAttentionProcessor
        # NOTE: Placeholder, adjust as needed for Qwen
        num_tokens = 16 
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    image_proj_model = MLPProjModel(
        cross_attention_dim=transformer.config.joint_attention_dim,
        id_embeddings_dim=1152, 
        num_tokens=num_tokens,
    )
    
    ip_attn_procs = {}
    # NOTE: This attention processor naming convention is based on Flux.
    # It might need to be adapted for the Qwen model's layer names.
    for name in transformer.attn_processors.keys():
        if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
            ip_attn_procs[name] = IPAttentionProcessor(
                hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=num_tokens,
            )
    
    if ip_attn_procs:
        transformer.set_attn_processor(ip_attn_procs)
    
    return image_proj_model
