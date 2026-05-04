#!/usr/bin/env python3
"""FreeText-style baseline on Qwen-Image.

This implementation follows the two main modules described in FreeText:

1. Attention-guided endogenous text-region localization:
   run a probe denoising pass, read image-to-text attention maps from Qwen
   DiT blocks when available, aggregate confident maps, and convert them into
   a latent-space writing mask with topology-aware post-processing.
2. Spectral-Modulated Glyph Injection (SGMI):
   render the target text into the localized region, encode it with Qwen's VAE,
   apply a Log-Gabor band-pass modulation to the noise-aligned glyph latent,
   and inject it during a mid-early denoising window.

The attention APIs of Qwen-Image variants differ across diffusers versions. To
keep this usable as a benchmark baseline, the localizer falls back to a
deterministic text-region prior if no compatible attention maps are exposed.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from infer.glyph_injector import InjectionConfig  # noqa: E402
from infer.attn_enhancement import find_quoted_token_indices  # noqa: E402
from qwen_inference import QwenImageInference, _calculate_shift, _retrieve_timesteps  # noqa: E402


@dataclass
class FreeTextConfig:
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: int = 42
    probe_steps: int = 16
    top_k_attention_maps: int = 12
    fallback_min_area: float = 0.08
    sgmi_strength: float = 0.28
    sgmi_window: tuple[float, float] = (0.20, 0.42)
    log_gabor_center: float = 0.36
    log_gabor_sigma: float = 0.55
    stroke_mask_dilate: int = 3
    stroke_mask_blur: int = 7
    max_font_size_ratio: float = 0.58
    max_region_width: float = 0.55
    max_region_height: float = 0.16


def extract_target_text(prompt: str, text: Optional[list[str] | str] = None) -> list[str]:
    if isinstance(text, list) and text:
        return [str(t) for t in text if str(t).strip()]
    if isinstance(text, str) and text.strip():
        return [text]
    quoted = [m.group(1) for m in re.finditer(r'"([^"]+)"', prompt)]
    if quoted:
        return quoted
    quoted = [m.group(1) for m in re.finditer(r"'([^']+)'", prompt)]
    return quoted


class AttentionMapCollector:
    """Best-effort image-to-text attention collector.

    The wrapper never changes model outputs: it computes attention maps from the
    same q/k projections when possible, stores them, then delegates to the
    original attention processor.
    """

    def __init__(self, hp: int, wp: int, text_indices: Optional[list[int]] = None, max_maps: int = 256):
        self.hp = hp
        self.wp = wp
        self.text_indices = text_indices or []
        self.max_maps = max_maps
        self.current_step = 0
        self.current_layer = 0
        self.maps: list[tuple[float, torch.Tensor]] = []
        self._installed: dict[object, object] = {}

    def install(self, transformer) -> None:
        idx = 0
        for module in transformer.modules():
            if not hasattr(module, "processor") or not hasattr(module, "to_q") or not hasattr(module, "to_k"):
                continue
            original = module.processor
            module.processor = _CaptureProcessor(original, self, idx)
            self._installed[module] = original
            idx += 1

    def uninstall(self) -> None:
        for module, original in self._installed.items():
            module.processor = original
        self._installed.clear()

    def maybe_record(self, attn, hidden_states, encoder_hidden_states) -> None:
        if encoder_hidden_states is None:
            return
        if len(self.maps) >= self.max_maps:
            return
        try:
            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            scale = getattr(attn, "scale", None) or (q.shape[-1] ** -0.5)
            logits = torch.bmm(q.float() * scale, k.float().transpose(1, 2))
            probs = logits.softmax(dim=-1)
            if self.text_indices:
                valid = [i for i in self.text_indices if 0 <= i < probs.shape[-1]]
                if valid:
                    probs = probs[:, :, valid]
            # Average over heads and target text tokens to recover image-token
            # attribution for the requested text span.
            spatial = probs.mean(dim=(0, 2))
            n = self.hp * self.wp
            if spatial.numel() < n:
                return
            spatial = spatial[:n].reshape(self.hp, self.wp)
            spatial = spatial - spatial.min()
            denom = spatial.max().clamp_min(1e-6)
            spatial = spatial / denom
            score = float((spatial.max() - spatial.mean()).detach().cpu())
            self.maps.append((score, spatial.detach().cpu()))
        except Exception:
            return

    def build_mask(self, height: int, width: int, top_k: int) -> Optional[np.ndarray]:
        if not self.maps:
            return None
        selected = sorted(self.maps, key=lambda x: x[0], reverse=True)[:top_k]
        heat = torch.stack([m for _, m in selected], dim=0).mean(dim=0).numpy()
        return topology_refine_attention_map(heat, width=width, height=height)


class _CaptureProcessor:
    def __init__(self, original, collector: AttentionMapCollector, layer_idx: int):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_collector", collector)
        object.__setattr__(self, "_layer_idx", layer_idx)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        image_rotary_emb=None,
        encoder_hidden_states_mask=None,
        image_emb=None,
        *args,
        **kwargs,
    ):
        self._collector.current_layer = self._layer_idx
        self._collector.maybe_record(attn, hidden_states, encoder_hidden_states)
        candidate_kwargs = dict(kwargs)
        candidate_kwargs["encoder_hidden_states"] = encoder_hidden_states
        if attention_mask is not None:
            candidate_kwargs["attention_mask"] = attention_mask
        if temb is not None:
            candidate_kwargs["temb"] = temb
        if image_rotary_emb is not None:
            candidate_kwargs["image_rotary_emb"] = image_rotary_emb
        if encoder_hidden_states_mask is not None:
            candidate_kwargs["encoder_hidden_states_mask"] = encoder_hidden_states_mask
        if image_emb is not None:
            candidate_kwargs["image_emb"] = image_emb

        # Diffusers ships several Qwen attention processors with slightly
        # different call signatures. Forward only what the wrapped processor can
        # accept, otherwise Python raises on harmless None/extra kwargs.
        try:
            sig = inspect.signature(self._original.__call__)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not has_var_kw:
                candidate_kwargs = {k: v for k, v in candidate_kwargs.items() if k in params}
        except (TypeError, ValueError):
            pass

        return self._original(attn, hidden_states, *args, **candidate_kwargs)


def topology_refine_attention_map(heat: np.ndarray, width: int, height: int) -> np.ndarray:
    heat = heat.astype(np.float32)
    heat = cv2.GaussianBlur(heat, (3, 3), 0)
    heat = heat - heat.min()
    if heat.max() > 1e-6:
        heat = heat / heat.max()
    heat_u8 = (heat * 255).astype(np.uint8)
    _, binary = cv2.threshold(heat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros((height, width), dtype=np.uint8)

    best_label = 1
    best_score = -1.0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 2:
            continue
        component = labels == label
        score = float(heat[component].mean()) * math.sqrt(float(area))
        if score > best_score:
            best_score = score
            best_label = label

    mask_small = (labels == best_label).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
    return cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)


def fallback_text_mask(width: int, height: int, text_count: int, min_area: float) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    box_w = int(width * 0.46)
    box_h = int(height * max(min_area, 0.10))
    x1 = (width - box_w) // 2
    y1 = int(height * 0.36)
    if text_count > 1:
        box_h = min(int(height * 0.12 * text_count), int(height * 0.45))
        y1 = (height - box_h) // 2
    cv2.rectangle(mask, (x1, y1), (x1 + box_w, y1 + box_h), 255, -1)
    return mask


def mask_to_plan(
    mask: np.ndarray,
    texts: list[str],
    max_font_size_ratio: float = 0.58,
    max_region_width: float = 0.55,
    max_region_height: float = 0.16,
) -> dict:
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    h, w = mask.shape
    if not rows.any() or not cols.any():
        x1, y1, x2, y2 = 0.15, 0.42, 0.85, 0.58
    else:
        yy = np.where(rows)[0]
        xx = np.where(cols)[0]
        pad_x = int(0.03 * w)
        pad_y = int(0.02 * h)
        x1 = max(int(xx[0]) - pad_x, 0) / w
        x2 = min(int(xx[-1]) + pad_x, w - 1) / w
        y1 = max(int(yy[0]) - pad_y, 0) / h
        y2 = min(int(yy[-1]) + pad_y, h - 1) / h

    # Attention maps can cover the carrier object rather than the text itself.
    # Keep the glyph reference a plausible text span instead of filling a large
    # object region, otherwise the encoded glyph prior becomes a giant template.
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw = min(x2 - x1, max_region_width)
    bh = min(y2 - y1, max_region_height * max(1, min(len(texts), 3)))
    x1, x2 = max(cx - bw * 0.5, 0.02), min(cx + bw * 0.5, 0.98)
    y1, y2 = max(cy - bh * 0.5, 0.02), min(cy + bh * 0.5, 0.98)

    texts = texts or [""]
    regions = []
    span = max(y2 - y1, 1e-3)
    for i, text in enumerate(texts):
        sub_y1 = y1 + span * i / len(texts)
        sub_y2 = y1 + span * (i + 1) / len(texts)
        regions.append({
            "content": text,
            "bbox": [float(x1), float(sub_y1), float(x2), float(sub_y2)],
            "font": "auto",
            "font_weight": "regular",
            "color": "white",
            "font_size_ratio": max_font_size_ratio,
            "is_latex": any(ch in text for ch in "\\_^{}=$"),
            "alignment": "center",
            "rotation": 0,
        })
    return {"image_analysis": {}, "text_regions": regions}


def log_gabor_filter_like(x: torch.Tensor, center: float, sigma: float) -> torch.Tensor:
    """Apply a 2D isotropic Log-Gabor band-pass filter per latent channel."""
    orig_dtype = x.dtype
    x_float = x.float()
    h, w = x_float.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device).view(h, 1)
    fx = torch.fft.fftfreq(w, device=x.device).view(1, w)
    rho = torch.sqrt(fx * fx + fy * fy).clamp_min(1e-6)
    center = max(center, 1e-3)
    sigma = max(sigma, 1e-3)
    kernel = torch.exp(-(torch.log(rho / center) ** 2) / (2 * (math.log(sigma) ** 2)))
    kernel = kernel.to(x_float.dtype)
    kernel[0, 0] = 0.0
    freq = torch.fft.fft2(x_float, dim=(-2, -1))
    filtered = torch.fft.ifft2(freq * kernel, dim=(-2, -1)).real
    return filtered.to(orig_dtype)


def make_soft_stroke_mask(mask_latent: torch.Tensor, dilate: int = 3, blur: int = 5) -> torch.Tensor:
    """Turn the rendered glyph stroke mask into a soft local injection mask.

    The FreeText paper injects SGMI inside the localized writing region. In this
    benchmark implementation the localization may fall back to a coarse box, so
    using the full region as replacement support creates visible rectangular
    patches. A softened stroke support keeps SGMI local to glyph structures and
    avoids injecting the template background.
    """
    mask = mask_latent.detach().float().clamp(0, 1)
    if dilate > 1:
        pad = dilate // 2
        mask = F.max_pool2d(mask, kernel_size=dilate, stride=1, padding=pad)
    if blur > 1:
        pad = blur // 2
        mask = F.avg_pool2d(F.pad(mask, [pad] * 4, mode="reflect"), kernel_size=blur, stride=1)
    return mask.clamp(0, 1).to(mask_latent.device)


def make_sgmi_residual(
    reference_latent: torch.Tensor,
    current_latent: torch.Tensor,
    center: float,
    sigma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a zero-mean SGMI structural residual.

    The paper's SGMI suppresses low-frequency background and irrelevant noise.
    Directly replacing the denoising latent with the rendered template latent can
    still impose the template's fixed white/gray appearance. This residual form
    keeps only normalized band-pass glyph structure and lets the current latent
    retain color/style statistics.
    """
    sgmi = log_gabor_filter_like(reference_latent, center, sigma).float()
    dims = (-2, -1)
    sgmi = sgmi - sgmi.mean(dim=dims, keepdim=True)
    sgmi_std = sgmi.std(dim=dims, keepdim=True).clamp_min(eps)
    cur_std = current_latent.float().std(dim=dims, keepdim=True).clamp_min(eps)
    return (sgmi / sgmi_std * cur_std).to(current_latent.dtype)


class FreeTextQwenGenerator:
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.qwen = QwenImageInference(model_path=model_path, device=device, dtype=dtype)

    @property
    def pipe(self):
        return self.qwen.pipeline

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        output_path: str,
        text: Optional[list[str] | str] = None,
        config: Optional[FreeTextConfig] = None,
    ) -> Image.Image:
        config = config or FreeTextConfig()
        texts = extract_target_text(prompt, text)
        if not texts:
            image = self.pipe(
                prompt=prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                true_cfg_scale=config.true_cfg_scale,
                generator=torch.Generator(device="cpu").manual_seed(config.seed),
            ).images[0]
            image.save(output_path)
            return image

        mask = self._localize_text_region(prompt, config, len(texts))
        plan = mask_to_plan(
            mask,
            texts,
            max_font_size_ratio=config.max_font_size_ratio,
            max_region_width=config.max_region_width,
            max_region_height=config.max_region_height,
        )
        image = self._generate_with_sgmi(prompt, plan, mask, config)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return image

    def _prepare_noise_and_timesteps(self, config: FreeTextConfig, steps: int):
        pipe = self.pipe
        latent_h = 2 * (config.height // (pipe.vae_scale_factor * 2))
        latent_w = 2 * (config.width // (pipe.vae_scale_factor * 2))
        num_channels = pipe.transformer.config.in_channels // 4
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        noise = torch.randn((1, 1, num_channels, latent_h, latent_w), generator=generator, dtype=torch.float32)
        packed = pipe._pack_latents(noise, 1, num_channels, latent_h, latent_w)
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = _calculate_shift(
            packed.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = _retrieve_timesteps(pipe.scheduler, steps, "cpu", sigmas=sigmas, mu=mu)
        return noise, timesteps

    def _localize_text_region(self, prompt: str, config: FreeTextConfig, text_count: int) -> np.ndarray:
        pipe = self.pipe
        noise, _ = self._prepare_noise_and_timesteps(config, config.probe_steps)
        device = pipe._execution_device
        dtype = pipe.transformer.dtype
        latent_h, latent_w = noise.shape[-2:]
        hp, wp = latent_h // 2, latent_w // 2

        try:
            text_indices = find_quoted_token_indices(pipe.tokenizer, prompt, max_seq_length=512)
        except Exception:
            text_indices = []
        collector = AttentionMapCollector(hp=hp, wp=wp, text_indices=text_indices)
        try:
            collector.install(pipe.transformer)
            self._run_denoising(
                prompt=prompt,
                noise=noise,
                steps=config.probe_steps,
                height=config.height,
                width=config.width,
                true_cfg_scale=config.true_cfg_scale,
                device=device,
                dtype=dtype,
                collector=collector,
                decode=False,
            )
        finally:
            collector.uninstall()

        mask = collector.build_mask(config.height, config.width, config.top_k_attention_maps)
        if mask is None or mask.max() == 0:
            mask = fallback_text_mask(config.width, config.height, text_count, config.fallback_min_area)
        return mask

    def _generate_with_sgmi(self, prompt: str, plan: dict, region_mask: np.ndarray, config: FreeTextConfig) -> Image.Image:
        pipe = self.pipe
        noise, timesteps = self._prepare_noise_and_timesteps(config, config.num_inference_steps)
        self.qwen.glyph_injector.sample_tag = "freetext"
        injection_data = self.qwen.glyph_injector.prepare_injection_from_plan(
            plan, (config.width, config.height), noise.squeeze(1), timesteps,
        )

        # Keep the attention/fallback region only for placing the glyph template.
        # For latent replacement, use a softened glyph-stroke support instead of
        # the full region; otherwise a coarse fallback box injects template
        # background and appears as a hard rectangular patch.
        injection_data["mask_latent"] = make_soft_stroke_mask(
            injection_data["mask_latent"].to(self.qwen.primary_device),
            dilate=config.stroke_mask_dilate,
            blur=config.stroke_mask_blur,
        )

        injection_data["sgmi_residual"] = True
        injection_data["log_gabor_center"] = config.log_gabor_center
        injection_data["log_gabor_sigma"] = config.log_gabor_sigma

        inject_cfg = InjectionConfig(
            mask_strength=config.sgmi_strength,
            timestep_ratio=config.sgmi_window,
            freq_decompose=False,
            strength_schedule="cosine",
        )
        return self._run_denoising(
            prompt=prompt,
            noise=noise,
            steps=config.num_inference_steps,
            height=config.height,
            width=config.width,
            true_cfg_scale=config.true_cfg_scale,
            device=pipe._execution_device,
            dtype=pipe.transformer.dtype,
            injection_data=injection_data,
            injection_config=inject_cfg,
            decode=True,
        )

    @torch.no_grad()
    def _run_denoising(
        self,
        prompt: str,
        noise: torch.Tensor,
        steps: int,
        height: int,
        width: int,
        true_cfg_scale: float,
        device,
        dtype,
        injection_data: Optional[dict] = None,
        injection_config: Optional[InjectionConfig] = None,
        collector: Optional[AttentionMapCollector] = None,
        decode: bool = True,
    ):
        pipe = self.pipe
        latent_h, latent_w = noise.shape[-2:]
        num_channels = noise.shape[2]
        prompt_embeds, _ = pipe.encode_prompt(prompt, device=device, max_sequence_length=512)
        neg_embeds, _ = pipe.encode_prompt("", device=device, max_sequence_length=512)
        latent = pipe._pack_latents(noise.to(device=device, dtype=dtype), 1, num_channels, latent_h, latent_w)
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = _calculate_shift(
            latent.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = _retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]]
        txt_seq_lens = [prompt_embeds.shape[1]]
        neg_txt_seq_lens = [neg_embeds.shape[1]]
        guidance = None

        pipe.scheduler.set_begin_index(0)
        for step_idx, t in enumerate(timesteps):
            if collector is not None:
                collector.current_step = step_idx
            timestep = t.expand(latent.shape[0]).to(latent.dtype)
            kwargs = dict(
                hidden_states=latent.to(dtype),
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_seq_lens=txt_seq_lens,
                img_shapes=img_shapes,
                return_dict=False,
            )
            # Some Qwen pipeline versions require masks, some do not.
            sig = inspect.signature(pipe.transformer.forward)
            if "encoder_hidden_states_mask" in sig.parameters:
                kwargs["encoder_hidden_states_mask"] = torch.ones(
                    prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device
                )
            noise_pred = pipe.transformer(**kwargs)[0]
            if true_cfg_scale > 1:
                neg_kwargs = dict(kwargs)
                neg_kwargs["encoder_hidden_states"] = neg_embeds
                neg_kwargs["txt_seq_lens"] = neg_txt_seq_lens
                if "encoder_hidden_states_mask" in neg_kwargs:
                    neg_kwargs["encoder_hidden_states_mask"] = torch.ones(
                        neg_embeds.shape[:2], dtype=torch.long, device=neg_embeds.device
                    )
                neg_pred = pipe.transformer(**neg_kwargs)[0]
                comb = neg_pred + true_cfg_scale * (noise_pred - neg_pred)
                noise_pred = comb * (torch.norm(noise_pred, dim=-1, keepdim=True) / torch.norm(comb, dim=-1, keepdim=True))

            latent = pipe.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
            if injection_data is not None and injection_config is not None:
                spatial = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
                spatial_4d = spatial.squeeze(2)
                if injection_data.get("sgmi_residual"):
                    spatial_4d = self._inject_sgmi_residual(
                        spatial_4d, injection_data, step_idx + 1, injection_config,
                    )
                else:
                    spatial_4d = self.qwen.glyph_injector.inject_latent(
                        spatial_4d, injection_data, step_idx + 1, config=injection_config,
                    )
                latent = pipe._pack_latents(spatial_4d.unsqueeze(2), 1, num_channels, latent_h, latent_w)

        if not decode:
            return None
        return self.qwen._decode_latent(latent, height, width)

    def _inject_sgmi_residual(
        self,
        current_latent: torch.Tensor,
        injection_data: dict,
        step_idx: int,
        config: InjectionConfig,
    ) -> torch.Tensor:
        total_steps = injection_data["total_steps"]
        if not config.should_inject(step_idx, total_steps):
            return current_latent
        latent_list = injection_data.get("latent_list")
        if not latent_list:
            return current_latent

        idx = min(step_idx, len(latent_list) - 1)
        reference = latent_list[idx].to(device=current_latent.device, dtype=current_latent.dtype)
        mask = injection_data["mask_latent"].to(device=current_latent.device, dtype=current_latent.dtype)
        mask = mask.expand_as(current_latent)
        strength = config.get_strength(step_idx, total_steps)

        residual = make_sgmi_residual(
            reference,
            current_latent,
            center=float(injection_data.get("log_gabor_center", 0.36)),
            sigma=float(injection_data.get("log_gabor_sigma", 0.55)),
        )
        return current_latent + mask * strength * residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen-image-2512")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--output_path", type=str, default="output/freetext_qwen.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--probe_steps", type=int, default=16)
    args = parser.parse_args()

    gen = FreeTextQwenGenerator(args.model_path)
    cfg = FreeTextConfig(seed=args.seed, num_inference_steps=args.steps, probe_steps=args.probe_steps)
    text = [args.text] if args.text.strip() else None
    gen.generate(args.prompt, args.output_path, text=text, config=cfg)


if __name__ == "__main__":
    main()
