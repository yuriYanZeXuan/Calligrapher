"""
Prompt-Latent Attention Enhancement

在去噪过程中增强 prompt 中待渲染文本 token 与字型 mask 对应 image patch 之间的注意力。
通过在 attention logits 添加 log(scale) 偏置实现，兼容 F.scaled_dot_product_attention。

用法:
    enh = AttentionEnhancement.create(config, tokenizer, prompt, mask_latent, ..., logger=logger)
    enh.install(transformer)
    for step in steps:
        enh.set_step(step, total)
        ...  # 正常去噪
    enh.uninstall(transformer)
"""

import math
import re
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SEQ_MULTI_OF = 32


# ============ 可视化工具 ============


def _attn_to_heatmap(attn_map: np.ndarray, size: tuple = None) -> Image.Image:
    """将 2D attention 数组转为 热力图 PIL Image。

    Args:
        attn_map: (H, W) float array, 值域 [0, 1] 或任意范围（会自动归一化）。
        size: 可选的输出尺寸 (w, h)。
    """
    arr = attn_map.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    # Jet-like colormap：蓝(0) → 青 → 绿 → 黄 → 红(1)
    r = np.clip(1.5 - abs(4 * arr - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * arr - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * arr - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    if size is not None:
        img = img.resize(size, Image.NEAREST)
    return img


def _make_patch_grid_image(
    mask: np.ndarray, indices: List[int], Hp: int, Wp: int, cell: int = 8
) -> Image.Image:
    """可视化 glyph patch 选中情况。

    返回一张 (Wp*cell, Hp*cell) 的图：灰色=mask覆盖的patch，黑色=未选中。
    """
    grid = np.zeros((Hp, Wp), dtype=np.uint8)
    for idx in indices:
        r, c = divmod(idx, Wp)
        if r < Hp:
            grid[r, c] = 200
    img = Image.fromarray(grid).resize((Wp * cell, Hp * cell), Image.NEAREST)
    return img


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
        sub_ids = tokenizer.encode(text, add_special_tokens=False)
        found = _find_subseq(real_ids, sub_ids)
        if found is None:
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
        logger=None,
        Hp: int = 0,
        Wp: int = 0,
    ):
        self.config = config
        self.text_indices = text_indices
        self.image_indices = image_indices
        self.x_seq_len = x_seq_len
        self.cap_seq_len = cap_seq_len
        self.num_layers = num_layers
        self.logger = logger
        self.Hp = Hp
        self.Wp = Wp

        self.current_step = 0
        self.total_steps = 1
        self._bias_cache: dict = {}
        # 记录已经可视化过的 (step, layer) 组合，避免重复保存
        self._logged_pairs: set = set()

    # ---- 日志步选择 ----
    # 只在第一个增强 step 和中间 step 各记录一次 layer-0 的 attention map

    def should_log_attn(self, layer_idx: int) -> bool:
        """当前 (step, layer) 是否需要可视化 attention。

        记录条件：每隔 3 个 layer、每隔 3 个 timestep，且未重复记录过。
        """
        if self.logger is None:
            return False
        if layer_idx % 1 != 0:
            return False
        if self.current_step % 3 != 0:
            return False
        pair = (self.current_step, layer_idx)
        if pair in self._logged_pairs:
            return False
        return True

    def mark_logged(self, layer_idx: int):
        self._logged_pairs.add((self.current_step, layer_idx))

    def should_enhance(self, layer_idx: int) -> bool:
        if not self.text_indices or not self.image_indices:
            return False
        step_ratio = self.current_step / max(self.total_steps, 1)
        if step_ratio >= self.config.attn_enhance_timestep_ratio:
            return False
        # layer 过滤：None = 所有层，list = 指定层
        layers = self.config.attn_enhance_layers
        if layers is not None and layer_idx not in layers:
            return False
        return True

    def get_bias(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """构造 (1, 1, L, L) 的 additive attention bias。"""
        key = (seq_len, device)
        if key in self._bias_cache:
            return self._bias_cache[key].to(dtype)

        bias = torch.zeros(
            1, 1, seq_len, seq_len, device=device, dtype=torch.float32
        )

        x_len = self.x_seq_len
        log_scale = math.log(max(self.config.attn_enhance_scale, 1e-6))

        text_abs = torch.tensor(
            [x_len + i for i in self.text_indices if i < self.cap_seq_len],
            dtype=torch.long,
            device=device,
        )
        image_abs = torch.tensor(
            [i for i in self.image_indices if i < x_len],
            dtype=torch.long,
            device=device,
        )

        if len(text_abs) == 0 or len(image_abs) == 0:
            self._bias_cache[key] = bias
            return bias.to(dtype)

        # 正向增强：glyph patch ↔ text token 注意力放大
        if self.config.attn_enhance_image_to_text:
            bias[0, 0, image_abs.unsqueeze(1), text_abs.unsqueeze(0)] = log_scale

        if self.config.attn_enhance_text_to_image:
            bias[0, 0, text_abs.unsqueeze(1), image_abs.unsqueeze(0)] = log_scale

        # 方案 D: 反向抑制 — 非 glyph patch 对 text token 的注意力压制
        suppress = self.config.attn_suppress_scale
        if suppress < 1.0 and suppress > 0:
            log_suppress = math.log(suppress)  # 负值，如 log(0.1) = -2.3
            # 所有 image patch 索引
            all_image = torch.arange(x_len, dtype=torch.long, device=device)
            # 非 glyph patch = 所有 image patch 中去掉 glyph 的
            glyph_set = set(self.image_indices)
            non_glyph = torch.tensor(
                [i for i in range(x_len) if i not in glyph_set],
                dtype=torch.long, device=device,
            )
            if len(non_glyph) > 0:
                # 非 glyph patch → text token 方向抑制
                bias[0, 0, non_glyph.unsqueeze(1), text_abs.unsqueeze(0)] = log_suppress
                # text token → 非 glyph patch 方向抑制
                bias[0, 0, text_abs.unsqueeze(1), non_glyph.unsqueeze(0)] = log_suppress

        self._bias_cache[key] = bias
        return bias.to(dtype)


# ============ Enhanced Attention Processor ============


class EnhancedAttnProcessor:
    """包装原始 attention processor，在激活时使用 logit-bias 增强注意力。"""

    def __init__(self, original, layer_idx: int, state: _EnhancementState):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_layer_idx", layer_idx)
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
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
        enhancing = state.should_enhance(self._layer_idx)

        # 非增强层：仍需检查是否要记录 attention map（每隔 3 层 / 3 步）
        if not enhancing:
            if state.should_log_attn(self._layer_idx):
                # 走 _forward_with_bias 但不注入 bias（仅记录 attention map）
                return self._forward_with_bias(
                    attn, hidden_states, attention_mask, freqs_cis,
                    state, self._layer_idx, inject_bias=False, **kwargs,
                )
            return self._original(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                **kwargs,
            )

        return self._forward_with_bias(
            attn, hidden_states, attention_mask, freqs_cis,
            state, self._layer_idx, inject_bias=True, **kwargs,
        )

    @staticmethod
    def _forward_with_bias(
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        state: _EnhancementState,
        layer_idx: int,
        inject_bias: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """与 ZSingleStreamAttnProcessor 相同的计算流程。

        inject_bias=True 时注入 logit bias（增强层），
        inject_bias=False 时仅走手动 attention 以便记录 attention map（非增强层）。
        """
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

        B, N, H, D = query.shape
        q = query.transpose(1, 2)  # (B, H, N, D)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2).to(dtype)

        # ---- 可视化：增强前后 attention 对比 ----
        if state.should_log_attn(layer_idx):
            state.mark_logged(layer_idx)
            _log_attn_comparison(q, k, attention_mask, state, layer_idx)

        # 构造 attention mask
        if inject_bias:
            attn_bias = state.get_bias(N, q.device, q.dtype)  # enhancement bias
        else:
            attn_bias = torch.zeros(1, 1, 1, 1, device=q.device, dtype=q.dtype)  # no bias

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            pad_bias = torch.zeros_like(attention_mask, dtype=q.dtype)
            pad_bias.masked_fill_(~attention_mask.bool(), float("-inf"))
            attn_bias = attn_bias + pad_bias

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )

        hidden_states = out.transpose(1, 2).flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


# ============ Attention Map 可视化 ============


@torch.no_grad()
def _log_attn_comparison(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    state: _EnhancementState,
    layer_idx: int,
):
    """在 enhance 前后分别计算 attention weights，可视化 text→image 响应图。

    只取 batch=0，对所有 head 取平均，提取 text-token 对 image-patch 的 attention
    并 reshape 为 (Hp, Wp) 空间热力图。
    """
    logger = state.logger
    step = state.current_step
    B, H, N, D = q.shape
    x_len = state.x_seq_len

    # 计算 raw scores（只取 batch 0）
    scores = torch.matmul(q[0], k[0].transpose(-2, -1)) / math.sqrt(D)  # (H, N, N)

    # padding mask
    if attention_mask is not None:
        mask = attention_mask
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        scores = scores.masked_fill(~mask[0].bool(), float("-inf"))

    # ---- 增强前 ----
    weights_before = torch.softmax(scores, dim=-1)  # (H, N, N)

    # ---- 增强后 ----
    enhance_bias = state.get_bias(N, q.device, scores.dtype)  # (1, 1, N, N)
    weights_after = torch.softmax(scores + enhance_bias[0, 0], dim=-1)

    # 提取：text tokens 对 image patches 的 attention（text→image 方向）
    # unified = [image(x_len), caption(cap_len)]
    text_abs = [x_len + i for i in state.text_indices if i < state.cap_seq_len]
    image_abs = [i for i in state.image_indices if i < x_len]

    if not text_abs or not image_abs:
        return

    text_t = torch.tensor(text_abs, device=q.device)
    # image→text: image patch 行对 text token 列的 attention
    # 对 text tokens 取平均，得到每个 image patch 的 "对文字的关注度"

    # (1) image→text 响应图：每个 image patch 对所有 text token 的平均 attention
    _save_response_map(
        weights_before, weights_after, state,
        row_indices=list(range(x_len)),  # 所有 image patches
        col_indices=text_abs,            # text tokens
        tag="img2txt", step=step, layer_idx=layer_idx,
    )

    # (2) text→image 响应图：每个 text token 对所有 image patch 的 attention，reshape 成空间图
    _save_response_map(
        weights_before, weights_after, state,
        row_indices=text_abs,            # text tokens
        col_indices=list(range(x_len)),  # 所有 image patches
        tag="txt2img", step=step, layer_idx=layer_idx,
    )


def _save_response_map(
    w_before: torch.Tensor,
    w_after: torch.Tensor,
    state: _EnhancementState,
    row_indices: List[int],
    col_indices: List[int],
    tag: str,
    step: int,
    layer_idx: int,
):
    """提取 attention 子矩阵并保存热力图。

    row_indices → query 方向, col_indices → key 方向
    对所有 head 取平均 → (len(row), len(col))
    然后根据 tag 决定聚合维度和 reshape 方式。
    """
    logger = state.logger
    H = w_before.shape[0]  # num heads
    x_len = state.x_seq_len
    Hp, Wp = state.Hp, state.Wp

    row_t = torch.tensor(row_indices, device=w_before.device)
    col_t = torch.tensor(col_indices, device=w_before.device)

    # 提取子矩阵：(H, len(row), len(col))
    sub_before = w_before[:, row_t][:, :, col_t].float().mean(dim=0).cpu().numpy()
    sub_after = w_after[:, row_t][:, :, col_t].float().mean(dim=0).cpu().numpy()

    if tag == "img2txt":
        # row=image patches, col=text tokens
        # 对 text token 维度求和 → 每个 image patch 对文字的总关注度 → reshape (Hp, Wp)
        resp_before = sub_before.sum(axis=1)  # (num_image_patches,)
        resp_after = sub_after.sum(axis=1)

        # 映射到完整 patch grid
        grid_before = np.zeros(x_len, dtype=np.float32)
        grid_after = np.zeros(x_len, dtype=np.float32)
        for i, idx in enumerate(row_indices):
            if idx < x_len:
                grid_before[idx] = resp_before[i]
                grid_after[idx] = resp_after[i]
        map_before = grid_before[:Hp * Wp].reshape(Hp, Wp)
        map_after = grid_after[:Hp * Wp].reshape(Hp, Wp)

    elif tag == "txt2img":
        # row=text tokens, col=image patches
        # 对 text token 维度求平均 → 每个 image patch 被文字关注的程度 → reshape (Hp, Wp)
        resp_before = sub_before.mean(axis=0)  # (num_image_patches,)
        resp_after = sub_after.mean(axis=0)

        grid_before = np.zeros(x_len, dtype=np.float32)
        grid_after = np.zeros(x_len, dtype=np.float32)
        for i, idx in enumerate(col_indices):
            if idx < x_len:
                grid_before[idx] = resp_before[i]
                grid_after[idx] = resp_after[i]
        map_before = grid_before[:Hp * Wp].reshape(Hp, Wp)
        map_after = grid_after[:Hp * Wp].reshape(Hp, Wp)
    else:
        return

    # 保存热力图（统一 scale 以便对比）
    vmin = min(map_before.min(), map_after.min())
    vmax = max(map_before.max(), map_after.max())
    if vmax - vmin < 1e-10:
        vmax = vmin + 1

    target_size = (Wp * 8, Hp * 8)

    hm_before = _attn_to_heatmap(map_before, size=target_size)
    hm_after = _attn_to_heatmap(map_after, size=target_size)

    # 拼接 before | after
    combined = Image.new("RGB", (target_size[0] * 2 + 4, target_size[1]), (255, 255, 255))
    combined.paste(hm_before, (0, 0))
    combined.paste(hm_after, (target_size[0] + 4, 0))

    caption = (
        f"step={step} layer={layer_idx} [{tag}]  "
        f"scale={state.config.attn_enhance_scale:.1f}  "
        f"LEFT=before  RIGHT=after"
    )
    logger.save_image(
        combined,
        f"attn_{tag}_step{step}_layer{layer_idx}",
        caption=caption,
        subfolder="attn_enhance",
    )


# ============ Public API ============


class AttentionEnhancement:
    """
    Prompt-Latent Attention Enhancement 管理器。

    用法:
        enh = AttentionEnhancement.create(config, tokenizer, prompt, mask_latent, ..., logger=logger)
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
        logger=None,
    ) -> Optional["AttentionEnhancement"]:
        """工厂方法：提取 token/patch 索引并创建 enhancement。"""
        if not config.attn_enhance_enabled:
            return None

        text_indices = find_quoted_token_indices(tokenizer, prompt, max_seq_length)
        image_indices = compute_glyph_patch_indices(mask_latent, patch_size)

        if not text_indices or not image_indices:
            msg = (
                f"[AttnEnhancement] 跳过：text_tokens={len(text_indices)}, "
                f"glyph_patches={len(image_indices)}"
            )
            print(msg)
            if logger:
                logger.info(msg)
            return None

        # 与 transformer.patchify_and_embed 一致的 SEQ_MULTI_OF 对齐
        Hp, Wp = latent_height // patch_size, latent_width // patch_size
        num_patches = Hp * Wp
        x_seq_len = num_patches + (-num_patches) % SEQ_MULTI_OF
        cap_seq_len = cap_ori_len + (-cap_ori_len) % SEQ_MULTI_OF

        state = _EnhancementState(
            config,
            text_indices,
            image_indices,
            x_seq_len,
            cap_seq_len,
            num_layers,
            logger=logger,
            Hp=Hp,
            Wp=Wp,
        )

        # ---- 日志：token 选择验证 ----
        info_msg = (
            f"[AttnEnhancement] 激活：text_tokens={len(text_indices)}, "
            f"glyph_patches={len(image_indices)}/{num_patches}, "
            f"scale={config.attn_enhance_scale:.1f}, "
            f"timestep_ratio={config.attn_enhance_timestep_ratio:.0%}, "
            f"layers={config.attn_enhance_layers or 'all'}"
        )
        print(info_msg)

        if logger:
            logger.info(info_msg)

            # 解码选中的 token 显示原文
            encoding = tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                ),
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
            )
            real_ids = encoding.input_ids[: sum(encoding.attention_mask)]
            selected_ids = [real_ids[i] for i in text_indices if i < len(real_ids)]
            decoded_tokens = tokenizer.decode(selected_ids)
            logger.info(
                f"[AttnEnhancement] 选中 text token indices: {text_indices}"
            )
            logger.info(
                f"[AttnEnhancement] 解码内容: \"{decoded_tokens}\""
            )
            logger.info(
                f"[AttnEnhancement] 逐 token: "
                + " | ".join(
                    f"[{i}]={tokenizer.decode([real_ids[i]])!r}"
                    for i in text_indices
                    if i < len(real_ids)
                )
            )
            logger.info(
                f"[AttnEnhancement] glyph patch 数: {len(image_indices)}, "
                f"patch grid: ({Hp}, {Wp}), x_seq_len={x_seq_len}, cap_seq_len={cap_seq_len}"
            )

            # 保存 glyph patch 选中可视化
            mask_np = mask_latent[0, 0].cpu().numpy()
            patch_grid_img = _make_patch_grid_image(mask_np, image_indices, Hp, Wp, cell=8)
            logger.save_image(
                patch_grid_img,
                "attn_enhance_patch_grid",
                caption=f"glyph patches: {len(image_indices)}/{num_patches}  grid=({Hp},{Wp})",
                subfolder="attn_enhance",
            )

            # --- bias 可视化 1: text×image 交叉子矩阵 ---
            # 只提取 text_indices 行 × image_indices 列 的紧凑子矩阵
            bias = state.get_bias(
                x_seq_len + cap_seq_len, mask_latent.device, torch.float32
            )
            bias_np = bias[0, 0].cpu().numpy()
            text_abs = [x_seq_len + i for i in text_indices if i < cap_seq_len]
            image_abs = [i for i in image_indices if i < x_seq_len]
            if text_abs and image_abs:
                # image→text 子矩阵: rows=image_patches, cols=text_tokens
                sub = bias_np[np.ix_(image_abs, text_abs)]  # (M, T)
                # 放大到可见尺寸
                h_px = max(sub.shape[0], 4)
                w_px = max(sub.shape[1] * 8, 64)  # text token 较少，横向放大
                sub_img = _attn_to_heatmap(sub, size=(w_px, h_px))
                token_labels = ", ".join(
                    f"{tokenizer.decode([real_ids[i]])!r}" for i in text_indices if i < len(real_ids)
                )
                logger.save_image(
                    sub_img,
                    "attn_enhance_bias_submatrix",
                    caption=(
                        f"bias submatrix (img_patches × text_tokens): "
                        f"{sub.shape[0]}×{sub.shape[1]}  "
                        f"log_scale={math.log(config.attn_enhance_scale):.3f}\n"
                        f"text tokens: {token_labels}"
                    ),
                    subfolder="attn_enhance",
                )

            # --- bias 可视化 2: image patch 空间增强图 ---
            # 每个 patch 被增强的 text token 数 → reshape (Hp, Wp)
            patch_enhance = np.zeros(x_seq_len, dtype=np.float32)
            for idx in image_abs:
                patch_enhance[idx] = 1.0
            spatial = patch_enhance[: Hp * Wp].reshape(Hp, Wp)
            spatial_img = _attn_to_heatmap(spatial, size=(Wp * 8, Hp * 8))
            logger.save_image(
                spatial_img,
                "attn_enhance_spatial_mask",
                caption=(
                    f"enhanced patch spatial map ({Hp}×{Wp})  "
                    f"patches={len(image_abs)}/{num_patches}  "
                    f"i2t={config.attn_enhance_image_to_text} t2i={config.attn_enhance_text_to_image}"
                ),
                subfolder="attn_enhance",
            )

        return cls(state)

    def install(self, transformer) -> None:
        """将所有层的 attention processor 替换为增强版。

        增强逻辑（logit bias）只在 attn_enhance_layers 指定的层中生效，
        但 attention map 日志记录在所有层上按 layer_idx % 3 == 0 触发。
        """
        if self._installed:
            return
        num_layers = len(transformer.layers)

        for idx in range(num_layers):
            layer = transformer.layers[idx]
            original = layer.attention.processor
            layer.attention.processor = EnhancedAttnProcessor(
                original, idx, self._state
            )

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
