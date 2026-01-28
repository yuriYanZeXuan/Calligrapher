import torch
import types
import textcrafter_transformer_flux

def init_forwards(self, root_module: torch.nn.Module):
    # Bind custom methods from textcrafter_transformer_flux
    CustomFluxTransformer2DModel = textcrafter_transformer_flux.FluxTransformer2DModel
    
    # Bind forward_hidden_states_list if not present
    if not hasattr(root_module, "forward_hidden_states_list"):
        root_module.forward_hidden_states_list = types.MethodType(
            CustomFluxTransformer2DModel.forward_hidden_states_list, root_module
        )
        
    # Bind insulation_replace_hidden_states if not present
    if not hasattr(root_module, "insulation_replace_hidden_states"):
        root_module.insulation_replace_hidden_states = types.MethodType(
            CustomFluxTransformer2DModel.insulation_replace_hidden_states, root_module
        )
        
    # Always override forward to support insulation arguments
    root_module.forward = types.MethodType(CustomFluxTransformer2DModel.forward, root_module)
        
    for name, module in root_module.named_modules():
        if (
            "attn" in name
            and "transformer_blocks" in name
            and "single_transformer_blocks" not in name
            and module.__class__.__name__ == "FluxAttention"
        ):
            module.forward = FluxTransformerBlock_init_forward(self, module)           
        elif (
            "attn" in name
            and "single_transformer_blocks" in name
            and module.__class__.__name__ == "FluxAttention"
        ):
            module.forward = FluxSingleTransformerBlock_init_forward(self, module) 

def FluxSingleTransformerBlock_init_forward(self, module):
    def forward(
        hidden_states=None,
        encoder_hidden_states=None,
        image_rotary_emb=None
    ):
        return module.processor(
            module,
            hidden_states=hidden_states,
            image_rotary_emb=image_rotary_emb
        )
    return forward

def FluxTransformerBlock_init_forward(self, module):
    def forward(
        hidden_states=None,
        encoder_hidden_states=None,
        image_rotary_emb=None
    ):
        return module.processor(
            module,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb
        )
    return forward