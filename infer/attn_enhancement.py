"""
Prompt-Latent Attention Enhancement

在去噪过程中增强 prompt 中待渲染文本 token 与字型 mask 对应 image patch 之间的注意力。
通过在 attention logits 添加 log(scale) 偏置实现，兼容 F.scaled_dot_product_attention。

用法:
    enh = AttentionEnhancement.create(config, tokenizer, prompt, mask_latent, ...)
    enh.install(transformer)
    for step in steps:
        enh.set_step(step, total)
        ...  # 正常去噪
    enh.uninstall(transformer)
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

SEQ_MULTI_OF = 32


# ============ Token / Patch 索引提取 ============


def find_quoted_token_indices(
    tokenizer, prompt: str, max_seq_length: int = 512
) -> List[int]:
    """从 prompt 中提取双引号内文本对应的 non-padding token 索引。

    与 pipeline._encode_prompt 使用完全相同的 chat-template + tokenizer 流程，
    保证索引对齐。

    Returns:
        在 caption embedding 序列中的 token 索引列表（0-based）。
    """
    matches = [m.group(1) for m in re.finditer(r'"([^"]*)"', prompt)]
    if not matches:
        return []

    # 与 pipeline._encode_prompt 一致的预处理
    messages = [{"role": "user", "content": prompt}]
    processed = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )

    encoding = tokenizer(
        processed, padding="max_length", max_length=max_seq_length, truncation=True,
    )
    full_ids = encoding.input_ids
    attn_mask = encoding.attention_mask
    num_real = sum(attn_mask)
    real_ids = full_ids[:num_real]

    result: list[int] = []
    for text in matches:
        # 优先尝试裸文本编码
        sub_ids = tokenizer.encode(text, add_special_tokens=False)
        found = _find_subseq(real_ids, sub_ids)
        if found is None:
            # 退化：带引号一起编码，去掉引号 token
            sub_ids_q = tokenizer.encode(f'"{text}"', add_special_tokens=False)
            found = _find_subseq(real_ids, sub_ids_q)
        if found is not None:
            result.extend(found)

    return sorted(set(result))


def _find_subseq(seq: list, subseq: list) -> Optional[List[int]]:
    """在 seq 中查找 subseq 的首次出现，返回索引列表。"""
    n, m = len(seq), len(subseq)
    if m == 0:
        return []
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return list(range(i, i + m))
    return None


def compute_glyph_patch_indices(
    mask_latent: torch.Tensor, patch_size: int = 2
) -> List[int]:
    """将 latent-space glyph mask 映射到 image patch 索引（raster order）。

    Args:
        mask_latent: (1, 1, H, W) binary mask in latent space
        patch_size: transformer patch size (default 2)

    Returns:
        有字型覆盖的 patch 索引列表。
    """
    mask = mask_latent[0, 0]  # (H, W)
    H, W = mask.shape
    Hp, Wp = H // patch_size, W // patch_size

    # reshape → (Hp, p, Wp, p)，对 patch 内取 max
    mask_grid = mask[: Hp * patch_size, : Wp * patch_size].reshape(
        Hp, patch_size, Wp, patch_size
    )
    has_glyph = mask_grid.amax(dim=(1, 3)) > 0  # (Hp, Wp)

    return torch.where(has_glyph.flatten())[0].tolist()


# ============ Enhancement State ============


class _EnhancementState:
    """在 transformer layers 间共享的增强状态。"""

    def __init__(
        self,
        config,  # InjectionConfig
        text_indices: List[int],
        image_indices: List[int],
        x_seq_len: int,
        cap_seq_len: int,
        num_layers: int,
    ):
        self.config = config
        self.text_indices = text_indices
        self.image_indices = image_indices
        self.x_seq_len = x_seq_len
        self.cap_seq_len = cap_seq_len
        self.num_layers = num_layers

        self.current_step = 0
        self.total_steps = 1
        self._bias_cache: dict = {}

    def should_enhance(self, layer_idx: int) -> bool:
        if not self.text_indices or not self.image_indices:
            return False
        step_ratio = self.current_step / max(self.total_steps, 1)
        if step_ratio >= self.config.attn_enhance_timestep_ratio:
            return False
        layer_ratio = layer_idx / max(self.num_layers, 1)
        if layer_ratio >= self.config.attn_enhance_layer_ratio:
            return False
        return True

    def get_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """构造 (1, 1, L, L) 的 additive attention bias。

        在 text-token ↔ glyph-patch 交叉位置填入 log(scale)，其余为 0。
        """
        key = (seq_len, device)
        if key in self._bias_cache:
            return self._bias_cache[key].to(dtype)

        bias = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=torch.float32)

        x_len = self.x_seq_len
        log_scale = math.log(max(self.config.attn_enhance_scale, 1e-6))

        # 绝对位置：unified = [image_patches (x_len), caption_tokens (cap_len)]
        text_abs = torch.tensor(
            [x_len + i for i in self.text_indices if i < self.cap_seq_len],
            dtype=torch.long, device=device,
        )
        image_abs = torch.tensor(
            [i for i in self.image_indices if i < x_len],
            dtype=torch.long, device=device,
        )

        if len(text_abs) == 0 or len(image_abs) == 0:
            self._bias_cache[key] = bias
            return bias.to(dtype)

        # 利用 advanced indexing 一次性填充 M×T 个位置
        if self.config.attn_enhance_image_to_text:
            bias[0, 0, image_abs.unsqueeze(1), text_abs.unsqueeze(0)] = log_scale

        if self.config.attn_enhance_text_to_image:
            bias[0, 0, text_abs.unsqueeze(1), image_abs.unsqueeze(0)] = log_scale

        self._bias_cache[key] = bias
        return bias.to(dtype)


# ============ Enhanced Attention Processor ============


class EnhancedAttnProcessor:
    """包装原始 attention processor，在激活时使用 logit-bias 增强注意力。

    非激活时（错误的 layer/timestep）直接 fallback 到原始 processor，零开销。
    """

    def __init__(self, original, layer_idx: int, state: _EnhancementState):
        # 用 object.__setattr__ 避免触发 __getattr__
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_layer_idx", layer_idx)
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
        """透传所有属性到原始 processor（兼容 to_k_ip 等 IP-Adapter 属性）。"""
        return getattr(self._original, name)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
        **kwargs,
    ) -> torch.Tensor:
        state: _EnhancementState = self._state
        if not state.should_enhance(self._layer_idx):
            return self._original(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis, **kwargs,
            )
        return self._forward_with_bias(
            attn, hidden_states, attention_mask, freqs_cis, state, **kwargs,
        )

    @staticmethod
    def _forward_with_bias(
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        state: _EnhancementState,
        **kwargs,
    ) -> torch.Tensor:
        """与 ZSingleStreamAttnProcessor 相同的计算流程，但注入 logit bias。"""
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # RoPE（直接复用原始实现）
        if freqs_cis is not None:
            def _rope(x_in, fc):
                with torch.amp.autocast("cuda", enabled=False):
                    x = torch.view_as_complex(
                        x_in.float().reshape(*x_in.shape[:-1], -1, 2)
                    )
                    fc = fc.unsqueeze(2)
                    return torch.view_as_real(x * fc).flatten(3).type_as(x_in)

            query = _rope(query, freqs_cis)
            key = _rope(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # (B, N, H, D) → (B, H, N, D)
        B, N, H, D = query.shape
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2).to(dtype)

        # 构造 float attention mask: padding → -inf, 其余 → 0
        attn_bias = state.get_bias(N, q.device, q.dtype)  # (1, 1, N, N)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, N)
            pad_bias = torch.zeros_like(attention_mask, dtype=q.dtype)
            pad_bias.masked_fill_(~attention_mask.bool(), float("-inf"))
            attn_bias = attn_bias + pad_bias  # broadcast → (B, 1, N, N)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False,
        )

        hidden_states = out.transpose(1, 2).flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


# ============ Public API ============


class AttentionEnhancement:
    """
    Prompt-Latent Attention Enhancement 管理器。

    用法:
        enh = AttentionEnhancement.create(config, tokenizer, prompt, mask_latent, ...)
        enh.install(transformer)       # monkey-patch layers
        enh.set_step(step, total)      # 每个去噪步调用
        enh.uninstall(transformer)     # 还原
    """

    def __init__(self, state: _EnhancementState):
        self._state = state
        self._installed = False

    @classmethod
    def create(
        cls,
        config,  # InjectionConfig
        tokenizer,
        prompt: str,
        mask_latent: torch.Tensor,
        latent_height: int,
        latent_width: int,
        cap_ori_len: int,
        num_layers: int,
        patch_size: int = 2,
        max_seq_length: int = 512,
    ) -> Optional["AttentionEnhancement"]:
        """工厂方法：提取 token/patch 索引并创建 enhancement。

        Args:
            config: InjectionConfig 实例
            tokenizer: pipeline.tokenizer
            prompt: 原始 prompt 字符串
            mask_latent: (1, 1, H, W) glyph mask in latent space
            latent_height, latent_width: latent 空间尺寸
            cap_ori_len: caption embedding 的原始 token 数（非 padding）
            num_layers: transformer.layers 的数量
            patch_size: transformer patch size
            max_seq_length: tokenizer max length

        Returns:
            AttentionEnhancement 实例；若无可增强的 token/patch 则返回 None。
        """
        if not config.attn_enhance_enabled:
            return None

        text_indices = find_quoted_token_indices(tokenizer, prompt, max_seq_length)
        image_indices = compute_glyph_patch_indices(mask_latent, patch_size)

        if not text_indices or not image_indices:
            print(
                f"[AttnEnhancement] 跳过：text_tokens={len(text_indices)}, "
                f"glyph_patches={len(image_indices)}"
            )
            return None

        # 与 transformer.patchify_and_embed 一致的 SEQ_MULTI_OF 对齐
        Hp, Wp = latent_height // patch_size, latent_width // patch_size
        num_patches = Hp * Wp
        x_seq_len = num_patches + (-num_patches) % SEQ_MULTI_OF
        cap_seq_len = cap_ori_len + (-cap_ori_len) % SEQ_MULTI_OF

        state = _EnhancementState(
            config, text_indices, image_indices,
            x_seq_len, cap_seq_len, num_layers,
        )

        print(
            f"[AttnEnhancement] 激活：text_tokens={len(text_indices)}, "
            f"glyph_patches={len(image_indices)}, "
            f"scale={config.attn_enhance_scale:.1f}, "
            f"timestep_ratio={config.attn_enhance_timestep_ratio:.0%}, "
            f"layer_ratio={config.attn_enhance_layer_ratio:.0%}"
        )
        return cls(state)

    def install(self, transformer) -> None:
        """将前 N 层的 attention processor 替换为增强版。"""
        if self._installed:
            return
        num_layers = len(transformer.layers)
        enhance_count = max(1, int(num_layers * self._state.config.attn_enhance_layer_ratio))

        for idx in range(min(enhance_count, num_layers)):
            layer = transformer.layers[idx]
            original = layer.attention.processor
            layer.attention.processor = EnhancedAttnProcessor(original, idx, self._state)

        self._installed = True

    def uninstall(self, transformer) -> None:
        """还原所有 attention processor。"""
        if not self._installed:
            return
        for layer in transformer.layers:
            proc = layer.attention.processor
            if isinstance(proc, EnhancedAttnProcessor):
                layer.attention.processor = proc._original
        self._installed = False

    def set_step(self, step: int, total_steps: int) -> None:
        """更新当前去噪步（在每个 timestep 开头调用）。"""
        self._state.current_step = step
        self._state.total_steps = total_steps
