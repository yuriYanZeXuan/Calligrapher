"""
Glyph-Only Ablation: Pass1 参考图 → VLM 布局 → Clean 背景 + 像素粘贴

不使用: latent 注入、频率分解、Pass 3 风格化。
纯像素空间操作——渲染字形模板后直接粘贴到 clean 背景上。
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from infer.VLM_agent import VLMAgent
from infer.formula_helper import render_formula, resolve_font_name
from infer.glyph_injector import GlyphInjector


def _render_and_composite(plan: dict, background: Image.Image) -> Image.Image:
    """渲染字形模板并像素粘贴到背景上。"""
    w, h = background.size
    canvas = Image.new("RGB", (w, h), "black")

    for r in plan.get("text_regions", []):
        bbox = r["bbox"]
        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
        rw, rh = max(x2 - x1, 1), max(y2 - y1, 1)

        color = GlyphInjector._resolve_color(r.get("color", "white"))
        font_path = r.get("font_path") or resolve_font_name(r.get("font"))

        img = render_formula(
            r["content"], rw, rh, text_color=color,
            force_latex=r.get("is_latex", False),
            font_weight=r.get("font_weight", "regular"),
            font_path=font_path,
            rotation=r.get("rotation", 0),
        )
        if img.size != (rw, rh):
            img = img.resize((rw, rh), Image.LANCZOS)
        canvas.paste(img, (x1, y1))

    gray = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bg_arr = np.array(background)
    tpl_arr = np.array(canvas)
    result = bg_arr.copy()
    result[mask > 127] = tpl_arr[mask > 127]
    return Image.fromarray(result)


class GlyphOnlyInference:
    """Glyph-Only 简化管线。

    支持 base_model: "zimage" | "qwen" | "klein"
    """

    def __init__(self, base_model: str, model_path: str, device: str = "cuda"):
        self.base_model = base_model
        self.device = device
        self._vlm = None
        self._load_pipeline(model_path)

    def _load_pipeline(self, model_path: str):
        if self.base_model == "zimage":
            from train.zimage_ip.pipeline_z_image import ZImagePipeline
            self._pipe = ZImagePipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
            ).to(self.device)
        elif self.base_model == "qwen":
            from diffusers import DiffusionPipeline
            self._pipe = DiffusionPipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
            ).to(self.device)
        elif self.base_model == "klein":
            sys.path.insert(0, os.path.join(
                os.path.dirname(__file__), "..", "..", "baselines", "fluxklein"))
            from inference_fluxklein import FluxKleinGenerator
            self._pipe = FluxKleinGenerator(
                model_path=model_path, device=self.device,
                enable_cpu_offload=False,
            )

    @property
    def vlm(self) -> VLMAgent:
        if self._vlm is None:
            self._vlm = VLMAgent()
        return self._vlm

    def _gen_image(self, prompt: str, seed: int) -> Image.Image:
        if self.base_model == "zimage":
            gen = torch.Generator(device=self.device).manual_seed(seed)
            return self._pipe(
                prompt=prompt, height=1024, width=1024,
                num_inference_steps=20, guidance_scale=0.0, generator=gen,
            ).images[0]
        elif self.base_model == "qwen":
            gen = torch.Generator(device=self.device).manual_seed(seed)
            return self._pipe(
                prompt=prompt, height=1024, width=1024,
                num_inference_steps=50, true_cfg_scale=4.0, generator=gen,
            ).images[0]
        elif self.base_model == "klein":
            tmp = os.path.join(tempfile.gettempdir(), f"_glyph_only_{os.getpid()}.png")
            return self._pipe.generate(
                prompt=prompt, image=None, seed=seed,
                num_inference_steps=50, guidance_scale=4.0,
                height=1024, width=1024, output_path=tmp,
            )

    def generate(
        self,
        prompt: str,
        text_contents: list[str] | None = None,
        seed: int = 42,
        run_name: str | None = None,
    ) -> Image.Image:
        if not text_contents:
            return self._gen_image(prompt, seed)

        # Pass 1: 用完整 prompt 生成参考图
        print("=== [GlyphOnly] Pass 1: reference ===")
        ref = self._gen_image(prompt, seed)

        # VLM 排版规划
        print("=== [GlyphOnly] VLM layout ===")
        plan = self.vlm.analyze_typography(ref, prompt, text_contents)

        # Clean prompt → 背景
        print("=== [GlyphOnly] Clean background ===")
        clean_prompt = self.vlm.generate_clean_prompt(prompt, plan)
        bg = self._gen_image(clean_prompt, seed)

        # 字形渲染 + 像素粘贴
        print("=== [GlyphOnly] Glyph paste ===")
        return _render_and_composite(plan, bg)
