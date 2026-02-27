"""
Z-Image 推理主入口

整合核心接口:
- VLMAgent: 统一的 VLM 调用中心（排版分析、prompt 改写、评分）
- GlyphInjector: 文字渲染和 latent 注入

三阶段推理架构:
  Pass 1: 用完整 prompt 生成参考图 → VLM 自主规划排版
  Pass 2: 从相同噪声用 clean prompt + 字形注入生成文字图
  Pass 3: FluxKlein img2img + 二值 mask → 将文字风格化为白色粉笔字效果
"""

import os
import sys
import json
from typing import Optional, Union
from dataclasses import dataclass, field

# 禁用 torch.compile 避免错误
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.zimage_ip.pipeline_z_image import ZImagePipeline
from infer.VLM_agent import VLMAgent, _add_grid_overlay
from infer.glyph_injector import GlyphInjector, TextRegion, InjectionConfig, create_glyph_injector
from infer.mylogger import TTSLogger

# 默认模型路径
DEFAULT_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"


@dataclass
class GenerationConfig:
    """生成配置"""
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 0.0
    seed: Optional[int] = None

    # Prompt Refiner
    use_prompt_refiner: bool = True
    refiner_temperature: float = 0.7

    # Glyph Injection
    use_glyph_injection: bool = True
    injection_config: InjectionConfig = field(default_factory=InjectionConfig)

    # Pass 3: 风格化 harmonizer
    use_harmonization: bool = True             # 启用 Pass 3 风格化
    harmonizer_type: str = "klein"             # "klein" | "qwenedit"

    # FluxKlein 参数
    klein_model_path: str = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein"
    klein_steps: int = 10
    klein_guidance_scale: float = 4.0
    klein_seed: Optional[int] = None

    # QwenEdit 参数
    qwenedit_model_path: str = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_edit_2511"
    qwenedit_steps: int = 50
    qwenedit_seed: Optional[int] = None


class ZImageInference:
    """
    Z-Image 两阶段推理

    Pass 1: 用完整 prompt（含文字描述）标准去噪 → 参考图
    VLM:   分析参考图，自主规划文本区域的 bbox / 字体 / 颜色 / 大小
    Pass 2: 从相同噪声，用 clean prompt（不含文字）去噪 + 字形注入
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Union[str, list[str]] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        logger: TTSLogger = None,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.logger = logger or TTSLogger(run_name="zimage_inference")

        if isinstance(device, str):
            self.devices = [device]
        else:
            self.devices = device
        self.primary_device = self.devices[0]

        # 延迟加载
        self._pipeline = None
        self._vlm_agent = None
        self._glyph_injector = None
        self._klein_generator = None
        self._qwenedit_generator = None
        self._output_counter = 0
        self._current_tag = "000"

    # ---- 延迟加载属性 ----

    @property
    def pipeline(self) -> ZImagePipeline:
        if self._pipeline is None:
            print(f"正在加载 Z-Image 模型到 {self.primary_device}...")
            self._pipeline = ZImagePipeline.from_pretrained(
                self.model_path, torch_dtype=self.dtype, low_cpu_mem_usage=False,
            )
            self._pipeline.to(self.primary_device)
            print("模型加载完成")
        return self._pipeline

    @property
    def vlm_agent(self) -> VLMAgent:
        if self._vlm_agent is None:
            self._vlm_agent = VLMAgent()
        return self._vlm_agent

    @property
    def glyph_injector(self) -> GlyphInjector:
        if self._glyph_injector is None:
            self._glyph_injector = create_glyph_injector(
                self.pipeline, device=self.primary_device, logger=self.logger,
            )
        return self._glyph_injector

    # ---- 显存管理 ----

    def _offload_pipeline_to_cpu(self):
        """将主管道移至 CPU 并释放 GPU 缓存，为 harmonizer 腾出显存。"""
        if self._pipeline is not None:
            print("  [MEM] 将 Z-Image pipeline 卸载到 CPU ...")
            self._pipeline.to("cpu")
            torch.cuda.empty_cache()

    def _reload_pipeline_to_gpu(self):
        """将主管道从 CPU 移回 GPU（如果当前在 CPU 上）。"""
        if self._pipeline is not None:
            if "cuda" not in str(self._pipeline.device):
                print(f"  [MEM] 将 Z-Image pipeline 移回 {self.primary_device} ...")
                self._pipeline.to(self.primary_device)

    def _offload_harmonizer_to_cpu(self):
        """将非 cpu_offload 模式的 harmonizer 移至 CPU 并释放 GPU 缓存。

        QwenEdit 使用 enable_model_cpu_offload，由 diffusers hook 自动管理，
        不能手动调用 .to()，否则会破坏 hook。
        """
        if self._klein_generator is not None:
            print("  [MEM] 将 FluxKlein 卸载到 CPU ...")
            self._klein_generator.pipe.to("cpu")
        torch.cuda.empty_cache()

    # ---- 延迟加载 harmonizer ----

    def get_klein_generator(self, config: "GenerationConfig"):
        """延迟加载 FluxKlein 生成器（按需创建，避免浪费显存）。"""
        if self._klein_generator is None:
            klein_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "fluxklein")
            if klein_dir not in sys.path:
                sys.path.insert(0, klein_dir)
            from inference_fluxklein import FluxKleinGenerator

            print(f"正在加载 FluxKlein 模型: {config.klein_model_path}")
            self._klein_generator = FluxKleinGenerator(
                model_path=config.klein_model_path,
                device=self.primary_device,
            )
            print("FluxKlein 模型加载完成")
        else:
            self._klein_generator.pipe.to(self.primary_device)
        return self._klein_generator

    def get_qwenedit_generator(self, config: "GenerationConfig"):
        """延迟加载 QwenEdit 生成器（使用 CPU offload 节省显存）。

        enable_model_cpu_offload 模式下：子模块常驻 CPU，推理时按需搬运到 GPU，
        用完自动回 CPU。峰值显存仅为单个子模块 + activation。
        """
        if self._qwenedit_generator is None:
            qwenedit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "qwenedit")
            if qwenedit_dir not in sys.path:
                sys.path.insert(0, qwenedit_dir)
            from inference_qwenedit import QwenEditGenerator

            print(f"正在加载 QwenEdit 模型: {config.qwenedit_model_path}")
            self._qwenedit_generator = QwenEditGenerator(
                model_path=config.qwenedit_model_path,
                device=self.primary_device,
                enable_cpu_offload=True,
            )
            print("QwenEdit 模型加载完成 (CPU offload)")
        return self._qwenedit_generator

    # ---- 主入口 ----

    def generate(
        self,
        prompt: str,
        text_contents: Optional[list[str]] = None,
        text_regions: Optional[list[dict]] = None,
        config: Optional[GenerationConfig] = None,
        run_name: Optional[str] = None,
        **kwargs
    ) -> Image.Image:
        """生成图像。

        Args:
            prompt: 生成 prompt（包含文字描述）
            text_contents: 待渲染的文本/公式内容列表（不含 bbox，由 VLM 自主布局）
            text_regions: **调试旁路**：含 bbox 的完整区域定义，传入时跳过 VLM 规划
            config: 生成配置
            run_name: 本次生成的标签名（用于 log 文件命名）
        """
        if config is None:
            config = GenerationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # 统一递增计数器，设置本次 sample 的 tag（所有中间产物共用此 tag）
        self._output_counter += 1
        self._current_tag = f"{self._output_counter:03d}_{run_name}" if run_name else f"{self._output_counter:03d}"

        # 文字内容汇总（用于 prompt refiner）
        text_content_str = None
        if text_contents:
            text_content_str = " ".join(text_contents)
        elif text_regions:
            text_content_str = " ".join(r.get("content", "") for r in text_regions)

        # 1. Prompt 优化（确定性后缀，不经 VLM 改写，保证原始文本不变）
        working_prompt = prompt
        if config.use_prompt_refiner:
            working_prompt = prompt + " High quality, with clearly legible and well-positioned text."
            print(f"Refined prompt: {working_prompt[:120]}...")

        # 2. 普通生成（可能带 Glyph Injection 三阶段推理）
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.primary_device).manual_seed(config.seed)

        has_text = (text_contents and len(text_contents) > 0) or (text_regions and len(text_regions) > 0)

        if config.use_glyph_injection and has_text:
            image = self._generate_with_injection(
                prompt=working_prompt,
                text_contents=text_contents,
                config=config,
                generator=generator,
                text_regions_override=text_regions,
            )
        else:
            image = self.pipeline(
                prompt=working_prompt,
                height=config.height, width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            ).images[0]

        # 强制 RGB，避免 RGBA 输出（harmonizer/VAE 可能返回 RGBA）
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 保存最终图到 logs
        if self.logger is not None:
            self.logger.save_image(
                image, f"final_{self._current_tag}",
                caption=f"[{self._current_tag}] {working_prompt[:150]}",
                subfolder="output",
            )

        return image

    # ---- 两阶段推理核心 ----

    def _generate_with_injection(
        self,
        prompt: str,
        text_contents: Optional[list[str]],
        config: GenerationConfig,
        generator: Optional[torch.Generator] = None,
        text_regions_override: Optional[list[dict]] = None,
    ) -> Image.Image:
        """三阶段推理：Pass 1 参考图 + VLM 规划 + Pass 2 clean 注入 + Pass 3 FluxKlein refine。

        将各阶段结果保存到 candidates 字典（仅用于可视化对比，不参与选优）。
        最终输出始终是管线最后一级（pass3 > pass2），pass1 仅作为排版参考。
        """
        candidates: dict[str, Image.Image] = {}
        tag = self._current_tag

        # 同步 tag 到 glyph_injector（用于其内部 save_image）
        if self._glyph_injector is not None:
            self._glyph_injector.sample_tag = tag

        # 准备共享噪声
        noise = self._prepare_noise(config, generator)

        # 设置 scheduler
        self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.primary_device)
        timesteps = self.pipeline.scheduler.timesteps

        if text_regions_override:
            typography_plan = self._text_regions_to_plan(text_regions_override)
            print("[调试旁路] 使用手动指定的 text_regions，跳过 VLM 规划")
        else:
            # === Pass 1: 参考图生成 ===
            print("=== Pass 1: 生成参考图 ===")
            reference_image = self._run_pass1_reference(prompt, noise.clone(), timesteps, config)
            candidates["pass1_reference"] = reference_image

            if self.logger is not None:
                self.logger.save_image(
                    reference_image, f"{tag}_pass1_reference",
                    caption=f"[Pass1] {prompt[:150]}",
                    subfolder="two_pass",
                )

            # === VLM 自主规划排版 ===
            print("=== VLM 排版规划 ===")
            reference_with_grid = _add_grid_overlay(reference_image)
            if self.logger is not None:
                self.logger.save_image(
                    reference_with_grid, f"{tag}_pass1_with_grid",
                    caption=f"[Grid] {prompt[:150]}",
                    subfolder="two_pass",
                )

            typography_plan = self.vlm_agent.analyze_typography(
                reference_image, prompt, text_contents,
            )

        self._save_typography_plan(typography_plan)

        # === 重置 scheduler ===
        self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.primary_device)
        timesteps = self.pipeline.scheduler.timesteps

        # === Pass 2: Clean 背景生成 + 像素空间字形合成 ===
        print("=== Pass 2: Clean 推理 + 像素空间字形合成 ===")
        clean_prompt = self.vlm_agent.generate_clean_prompt(prompt, typography_plan)
        print(f"Clean prompt: {clean_prompt}...")

        image_size = (config.width, config.height)
        self.glyph_injector.sample_tag = tag
        injection_data = self.glyph_injector.prepare_injection_from_plan(
            typography_plan, image_size, noise, timesteps,
        )

        # === Mask 全黑检测：模板渲染失败 → 回退 pass1 ===
        if injection_data["full_mask"].max() == 0:
            print("  [WARN] glyph mask 全黑（模板渲染失败），回退到 pass1")
            self._save_candidates_concat(candidates)
            return candidates.get("pass1_reference", list(candidates.values())[-1])

        background = self._run_pass2_injection_with_data(
            clean_prompt, noise, timesteps, injection_data, config,
        )

        pass2_image = self._pixel_composite_text(background, injection_data)
        candidates["pass2_injection"] = pass2_image

        # === Pass 3: 风格化 → 多变体候选 ===
        pass3_variants: list[tuple[str, Image.Image]] = []
        if config.use_harmonization:
            print(f"=== Pass 3: {config.harmonizer_type} 风格化 ===")
            pass3_variants = self._run_pass3_harmonize(
                pass2_image, injection_data, typography_plan, config,
            )
            for name, img in pass3_variants:
                candidates[f"pass3_{name}"] = img

        # === 拼接所有阶段结果保存到 CAT_IMG ===
        self._save_candidates_concat(candidates)

        # === VLM 基于文本准确度从 pass2 + 所有 pass3 变体中选优 ===
        all_text = " ".join(
            r["content"] for r in typography_plan.get("text_regions", []))
        selection_pool = [("pass2_injection", pass2_image)]
        selection_pool.extend(pass3_variants)

        if len(selection_pool) > 1:
            names, images = zip(*selection_pool)
            best_idx = self.vlm_agent.select_best_text_match(list(images), all_text)
            print(f"  VLM 文本准确度选优: {names[best_idx]}")
            return images[best_idx]
        return pass2_image

    def _prepare_noise(
        self, config: GenerationConfig, generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """准备共享噪声。"""
        latent_height = 2 * (config.height // (self.pipeline.vae_scale_factor * 2))
        latent_width = 2 * (config.width // (self.pipeline.vae_scale_factor * 2))
        num_channels = self.pipeline.transformer.in_channels
        return torch.randn(
            (1, num_channels, latent_height, latent_width),
            generator=generator,
            device=self.primary_device,
            dtype=torch.float32,
        )

    def _run_pass1_reference(
        self,
        prompt: str,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        config: GenerationConfig,
    ) -> Image.Image:
        """Pass 1: 用完整 prompt 标准去噪，生成参考图（无注入）。"""
        dtype = self.pipeline.transformer.dtype
        latent = noise.clone()

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt, device=self.primary_device, do_classifier_free_guidance=False,
        )

        for t in timesteps:
            timestep = t.expand(1)
            timestep_norm = (1000 - timestep) / 1000

            latent_input = latent.to(dtype).unsqueeze(2)
            latent_list = list(latent_input.unbind(dim=0))

            with torch.no_grad():
                model_out = self.pipeline.transformer(
                    latent_list, timestep_norm, prompt_embeds, return_dict=False,
                )[0]

            noise_pred = -torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
            latent = self.pipeline.scheduler.step(
                noise_pred.to(torch.float32), t, latent, return_dict=False,
            )[0]

        # 解码
        return self._decode_latent(latent)

    def _run_pass2_injection(
        self,
        clean_prompt: str,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        typography_plan: dict,
        config: GenerationConfig,
    ) -> Image.Image:
        """Pass 2: 用 clean prompt 去噪 + 按 plan 注入字形。"""
        image_size = (config.width, config.height)
        injection_data = self.glyph_injector.prepare_injection_from_plan(
            typography_plan, image_size, noise, timesteps,
        )
        return self._run_pass2_injection_with_data(
            clean_prompt, noise, timesteps, injection_data, config,
        )

    def _run_pass2_injection_with_data(
        self,
        clean_prompt: str,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        injection_data: dict,
        config: GenerationConfig,
    ) -> Image.Image:
        """Pass 2 核心：用 clean prompt 去噪 + 注入字形（使用预先准备好的 injection_data）。"""
        # 编码 clean prompt
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=clean_prompt, device=self.primary_device, do_classifier_free_guidance=False,
        )

        # 获取 latent 尺寸信息（用于 attention enhancement）
        latent_height = noise.shape[2]
        latent_width = noise.shape[3]

        # Attention Enhancement
        attn_enh = None
        icfg = config.injection_config
        if icfg.attn_enhance_enabled:
            from infer.attn_enhancement import AttentionEnhancement
            attn_enh = AttentionEnhancement.create(
                config=icfg,
                tokenizer=self.pipeline.tokenizer,
                prompt=clean_prompt,
                mask_latent=injection_data["mask_latent"],
                latent_height=latent_height,
                latent_width=latent_width,
                cap_ori_len=len(prompt_embeds[0]),
                num_layers=len(self.pipeline.transformer.layers),
                logger=self.logger,
            )
            if attn_enh is not None:
                attn_enh.install(self.pipeline.transformer)

        # 去噪 + 注入
        latent = self._denoise_template_inject(
            noise, timesteps, prompt_embeds, injection_data, config,
            attn_enh=attn_enh,
        )

        # 卸载 attention enhancement
        if attn_enh is not None:
            attn_enh.uninstall(self.pipeline.transformer)

        return self._decode_latent(latent)

    def _denoise_template_inject(
        self,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: list,
        injection_data: dict,
        config: GenerationConfig,
        attn_enh=None,
    ) -> torch.Tensor:
        """模板注入去噪：attention enhancement + 可选 latent 注入。"""
        dtype = self.pipeline.transformer.dtype
        latent = noise.clone()
        total_steps = len(timesteps)

        for step_idx, t in enumerate(timesteps):
            if attn_enh is not None:
                attn_enh.set_step(step_idx, total_steps)
            timestep = t.expand(1)
            timestep_norm = (1000 - timestep) / 1000

            latent_input = latent.to(dtype).unsqueeze(2)
            latent_list = list(latent_input.unbind(dim=0))

            with torch.no_grad():
                model_out = self.pipeline.transformer(
                    latent_list, timestep_norm, prompt_embeds, return_dict=False,
                )[0]
            noise_pred = -torch.stack([o.float() for o in model_out], dim=0).squeeze(2)

            latent = self.pipeline.scheduler.step(
                noise_pred.to(torch.float32), t, latent, return_dict=False,
            )[0]

            if config.use_glyph_injection:
                latent = self.glyph_injector.inject_latent(
                    latent, injection_data, step_idx + 1,
                    config=config.injection_config,
                )

        return latent

    # ---- 工具方法 ----

    def _decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """解码 latent 为 PIL Image。"""
        latent = latent.to(self.pipeline.vae.dtype)
        latent = (latent / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        with torch.no_grad():
            image = self.pipeline.vae.decode(latent, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

    # ---- 像素空间合成 ----

    def _pixel_composite_text(
        self,
        background: Image.Image,
        injection_data: dict,
    ) -> Image.Image:
        """像素空间合成：用 full_mask 只贴文字笔画像素到背景上。

        与 latent 注入不同，这里在全分辨率操作，mask 精确到笔画，
        不会引入模板背景色块。
        """
        template = injection_data["combined_template"]
        mask = injection_data["full_mask"]

        bg_arr = np.array(background)
        tpl_arr = np.array(template)

        # 确保尺寸匹配
        if tpl_arr.shape[:2] != bg_arr.shape[:2]:
            template = template.resize(background.size, Image.LANCZOS)
            tpl_arr = np.array(template)
        if mask.shape[:2] != bg_arr.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize(background.size, Image.NEAREST))

        # 只在文字笔画处替换像素
        stroke_mask = mask > 127
        result = bg_arr.copy()
        result[stroke_mask] = tpl_arr[stroke_mask]

        composite = Image.fromarray(result)

        if self.logger is not None:
            tag = self._current_tag
            comp_vis = Image.new("RGB", (background.width * 3, background.height))
            comp_vis.paste(background, (0, 0))
            comp_vis.paste(Image.fromarray(mask).convert("RGB"), (background.width, 0))
            comp_vis.paste(composite, (background.width * 2, 0))
            self.logger.save_image(
                comp_vis, f"{tag}_pixel_composite",
                caption="Background | Mask | Composite",
                subfolder="two_pass",
            )

        return composite

    # ---- Pass 3: 风格化 (harmonizer) ----

    def _run_pass3_harmonize(
        self,
        pass2_image: Image.Image,
        injection_data: dict,
        typography_plan: dict,
        config: GenerationConfig,
    ) -> list[tuple[str, Image.Image]]:
        """Pass 3 统一入口，返回 [(name, image), ...] 候选列表。

        自动管理显存：进入前卸载主管道到 CPU，完成后恢复到 GPU。
        """
        self._offload_pipeline_to_cpu()

        try:
            if config.harmonizer_type == "qwenedit":
                result = self._run_pass3_qwenedit(pass2_image, typography_plan, config)
                variants = [("qwenedit", result)]
            else:
                variants = self._run_pass3_klein(pass2_image, injection_data, typography_plan, config)
        finally:
            self._offload_harmonizer_to_cpu()
            self._reload_pipeline_to_gpu()

        return variants

    def _run_pass3_klein(
        self,
        pass2_image: Image.Image,
        injection_data: dict,
        typography_plan: dict,
        config: GenerationConfig,
    ) -> list[tuple[str, Image.Image]]:
        """Pass 3 (Klein): 生成三种变体供 VLM 选优。

        返回 [(name, image), ...] 列表：
          1. klein_single  — 单图条件（仅 pass2）+ mask 混合
          2. klein_nomask  — 单图条件（仅 pass2），无 mask
          3. klein_dual    — 双图条件（pass2 + glyph 模板），无 mask
        """
        image_analysis = typography_plan.get("image_analysis", {})
        klein_prompt = self.vlm_agent.generate_style_prompt(image_analysis)
        print(f"  VLM 生成提示词: {klein_prompt[:80]}...")

        klein = self.get_klein_generator(config)
        klein_seed = config.klein_seed if config.klein_seed is not None else (config.seed or 42)
        gen_kwargs = dict(
            prompt=klein_prompt,
            num_inference_steps=config.klein_steps,
            guidance_scale=config.klein_guidance_scale,
        )

        template = injection_data["combined_template"]
        if template.size != pass2_image.size:
            template = template.resize(pass2_image.size, Image.LANCZOS)

        def _make_generator():
            return torch.Generator(device=klein.device).manual_seed(klein_seed)

        def _ensure_rgb(img):
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != pass2_image.size:
                img = img.resize(pass2_image.size, Image.LANCZOS)
            return img

        variants: list[tuple[str, Image.Image]] = []

        # --- 变体 1 & 2: 单图条件 → mask 混合 + 无 mask ---
        try:
            print("    [1/3] klein_single + [2/3] klein_nomask")
            raw_single = klein.pipe(
                **gen_kwargs, image=[pass2_image], generator=_make_generator(),
            ).images[0]
            raw_single = _ensure_rgb(raw_single)

            binary_mask = injection_data["full_mask"]
            mask = (binary_mask > 127).astype(np.float32)
            if mask.shape[:2] != (pass2_image.height, pass2_image.width):
                mask = np.array(Image.fromarray(
                    (mask * 255).astype(np.uint8)).resize(
                    pass2_image.size, Image.NEAREST)).astype(np.float32) / 255.0
            mask_3ch = mask[:, :, np.newaxis]
            klein_single = Image.fromarray((
                mask_3ch * np.array(raw_single).astype(np.float32)
                + (1 - mask_3ch) * np.array(pass2_image).astype(np.float32)
            ).clip(0, 255).astype(np.uint8))

            variants.append(("klein_single", klein_single))
            variants.append(("klein_nomask", raw_single))
        except Exception as e:
            print(f"    [WARN] Klein 单图推理失败: {e}")

        # --- 变体 3: 双图条件（pass2 + glyph 模板），无 mask ---
        try:
            print("    [3/3] klein_dual (双图条件)")
            klein_dual = klein.pipe(
                **gen_kwargs, image=[pass2_image, template], generator=_make_generator(),
            ).images[0]
            klein_dual = _ensure_rgb(klein_dual)
            variants.append(("klein_dual", klein_dual))
        except Exception as e:
            print(f"    [WARN] Klein 双图推理失败: {e}")

        if self.logger is not None:
            tag = self._current_tag
            w, h = pass2_image.size
            comp = Image.new("RGB", (w * len(variants), h))
            for i, (name, img) in enumerate(variants):
                comp.paste(img, (w * i, 0))
            self.logger.save_image(comp, f"{tag}_pass3_variants",
                caption=" | ".join(n for n, _ in variants), subfolder="harmonize")

        return variants

    def _run_pass3_qwenedit(
        self,
        pass2_image: Image.Image,
        typography_plan: dict,
        config: GenerationConfig,
    ) -> Image.Image:
        """Pass 3 (QwenEdit): 指令式全图编辑，无 mask 混合。"""
        image_analysis = typography_plan.get("image_analysis", {})
        # style_prompt = self.vlm_agent.generate_style_prompt(image_analysis)
        # print(f"  VLM 生成提示词: {style_prompt[:80]}...")
        style_prompt="keep background unedited, make foreground text and formulas be harmonize with whole image."
        qwenedit = self.get_qwenedit_generator(config)
        qe_seed = config.qwenedit_seed if config.qwenedit_seed is not None else (config.seed or 42)

        edited = qwenedit.generate(
            prompt=style_prompt,
            image=pass2_image,
            seed=qe_seed,
            num_inference_steps=config.qwenedit_steps,
        )

        # 确保 RGB 且尺寸与 pass2 一致（QwenEdit 可能返回不同分辨率）
        if edited.mode != "RGB":
            edited = edited.convert("RGB")
        if edited.size != pass2_image.size:
            print(f"  [WARN] QwenEdit 输出 {edited.size} ≠ Pass2 {pass2_image.size}，resize 对齐")
            edited = edited.resize(pass2_image.size, Image.LANCZOS)

        if self.logger is not None:
            tag = self._current_tag
            comp = Image.new("RGB", (pass2_image.width * 2, pass2_image.height))
            comp.paste(pass2_image, (0, 0))
            comp.paste(edited, (pass2_image.width, 0))
            self.logger.save_image(comp, f"{tag}_pass3_result",
                caption="Pass2 | QwenEdit", subfolder="harmonize")

        return edited

    @staticmethod
    def _text_regions_to_plan(text_regions: list[dict]) -> dict:
        """将调试用的 text_regions dict 列表转为 typography_plan 格式。"""
        plan_regions = []
        for r in text_regions:
            plan_regions.append({
                "content": r.get("content", ""),
                "bbox": r.get("bbox", [0, 0, 1, 1]),
                "font_weight": r.get("font_weight", "regular"),
                "font_size_ratio": r.get("font_size_ratio", 0.7),
                "color": r.get("color", "#FFFFFF"),
                "is_latex": r.get("is_latex", False),
                "alignment": r.get("alignment", "center"),
                "rotation": r.get("rotation", 0),
            })
        return {
            "image_analysis": {
                "background_style": "debug",
                "dominant_colors": ["#000000"],
                "text_style_hint": "debug mode",
            },
            "text_regions": plan_regions,
        }

    _CAT_IMG_DIR = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/logs/CAT_IMG"

    def _save_candidates_concat(self, candidates: dict[str, Image.Image]) -> None:
        """将所有候选图水平拼接，保存到 CAT_IMG 目录。"""
        if not candidates:
            return
        imgs = list(candidates.values())
        max_h = max(img.height for img in imgs)
        total_w = sum(img.width for img in imgs)
        concat = Image.new("RGB", (total_w, max_h))
        x = 0
        for img in imgs:
            concat.paste(img, (x, 0))
            x += img.width
        os.makedirs(self._CAT_IMG_DIR, exist_ok=True)
        save_path = os.path.join(self._CAT_IMG_DIR, f"{self._current_tag}.jpg")
        concat.save(save_path, format="JPEG", quality=90)
        print(f"  候选拼接图已保存: {save_path}")

    def _save_typography_plan(self, plan: dict) -> None:
        """保存 typography_plan JSON 到 logs 目录。"""
        if self.logger is None:
            return
        plan_dir = self.logger.run_dir / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"typography_plan_{self._current_tag}.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        self.logger.info(f"排版规划已保存: {plan_path.relative_to(self.logger.run_dir)}")

    def __call__(
        self,
        prompt: str,
        text_contents: Optional[list[str]] = None,
        text_regions: Optional[list[dict]] = None,
        **kwargs,
    ) -> Image.Image:
        return self.generate(prompt, text_contents, text_regions, **kwargs)


# ============ 多 GPU 并行支持 ============


def _worker_init(rank: int, model_path: str, device: str, dtype: torch.dtype):
    global _worker_inference
    _worker_inference = ZImageInference(model_path, device=device, dtype=dtype)
    print(f"Worker {rank} 初始化完成，设备: {device}")


def _worker_generate(args: tuple) -> tuple:
    global _worker_inference
    prompt, text_contents, text_regions, config_dict, worker_id = args
    config = GenerationConfig(**config_dict)
    if config.seed is not None:
        config.seed = config.seed + worker_id
    image = _worker_inference.generate(prompt, text_contents, text_regions, config)
    return worker_id, image


class ParallelZImageInference:
    """多 GPU 并行推理"""

    def __init__(
        self,
        model_path: str,
        devices: list[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.dtype = dtype
        if devices is None:
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
        self.devices = devices
        self.num_workers = len(devices)
        print(f"并行推理初始化，使用 {self.num_workers} 张 GPU: {devices}")

    def generate_batch(
        self,
        prompts: list[str],
        text_contents_list: Optional[list[list[str]]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> list[Image.Image]:
        if config is None:
            config = GenerationConfig()

        config_dict = {
            "height": config.height, "width": config.width,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "seed": config.seed,
            "use_prompt_refiner": config.use_prompt_refiner,
            "use_glyph_injection": config.use_glyph_injection,
        }

        if text_contents_list is None:
            text_contents_list = [None] * len(prompts)

        results = []
        inference = ZImageInference(self.model_path, device=self.devices[0], dtype=self.dtype)

        for i in range(len(prompts)):
            cfg = GenerationConfig(**config_dict)
            if cfg.seed is not None:
                cfg.seed = cfg.seed + i
            image = inference.generate(prompts[i], text_contents_list[i], config=cfg)
            results.append(image)
            print(f"完成 {i + 1}/{len(prompts)}")

        return results


def create_inference(
    model_path: str,
    device: Union[str, list[str]] = "cuda",
    parallel: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> Union[ZImageInference, ParallelZImageInference]:
    if parallel:
        if isinstance(device, str):
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
        else:
            devices = device
        return ParallelZImageInference(model_path, devices, dtype)
    else:
        return ZImageInference(model_path, device, dtype)


if __name__ == "__main__":
    print("Z-Image Inference 模块加载成功")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
