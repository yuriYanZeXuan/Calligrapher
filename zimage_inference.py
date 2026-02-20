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

        # 文字内容汇总（用于 prompt refiner）
        text_content_str = None
        if text_contents:
            text_content_str = " ".join(text_contents)
        elif text_regions:
            text_content_str = " ".join(r.get("content", "") for r in text_regions)

        # 1. Prompt 优化
        working_prompt = prompt
        if config.use_prompt_refiner and text_content_str:
            refined_prompts = self.vlm_agent.refine_prompt(
                prompt, text_content=text_content_str, num_variants=1,
            )
            working_prompt = refined_prompts[0]
            print(f"优化后的 prompt: {working_prompt[:100]}...")

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
            self._output_counter += 1
            tag = f"{self._output_counter:03d}_{run_name}" if run_name else f"{self._output_counter:03d}"
            self.logger.save_image(
                image, f"final_{tag}",
                caption=f"[{tag}] {working_prompt[:150]}",
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
        """三阶段推理：Pass 1 参考图 + VLM 规划 + Pass 2 clean 注入 + Pass 3 FluxKlein refine。"""

        # 准备共享噪声
        noise = self._prepare_noise(config, generator)

        # 设置 scheduler
        self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.primary_device)
        timesteps = self.pipeline.scheduler.timesteps

        if text_regions_override:
            # 调试旁路：跳过 VLM，直接用手动 text_regions
            typography_plan = self._text_regions_to_plan(text_regions_override)
            print("[调试旁路] 使用手动指定的 text_regions，跳过 VLM 规划")
        else:
            # === Pass 1: 参考图生成 ===
            print("=== Pass 1: 生成参考图 ===")
            reference_image = self._run_pass1_reference(prompt, noise.clone(), timesteps, config)

            # 保存参考图到 logs
            if self.logger is not None:
                self.logger.save_image(
                    reference_image, "pass1_reference",
                    caption=f"[Pass1] {prompt[:150]}",
                    subfolder="two_pass",
                )

            # === VLM 自主规划排版 ===
            print("=== VLM 排版规划 ===")
            
            # 生成并保存带网格的参考图（用于 debug）
            # 使用默认的 11×11 网格（10×10 区域，步长0.1）
            reference_with_grid = _add_grid_overlay(reference_image)
            if self.logger is not None:
                self.logger.save_image(
                    reference_with_grid, "pass1_with_grid",
                    caption=f"[Grid] {prompt[:150]}",
                    subfolder="two_pass",
                )
            
            typography_plan = self.vlm_agent.analyze_typography(
                reference_image, prompt, text_contents,
            )

        # 保存 typography_plan JSON
        self._save_typography_plan(typography_plan)

        # === 重置 scheduler：Pass 1 和 Pass 2 使用独立的 scheduler 状态 ===
        # FlowMatchEulerDiscreteScheduler 内部维护 _step_index，连续使用会导致越界
        self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.primary_device)
        timesteps = self.pipeline.scheduler.timesteps

        # === Pass 2: Clean 背景 + 字形注入 ===
        print("=== Pass 2: Clean 推理 + 字形注入 ===")
        clean_prompt = self.vlm_agent.generate_clean_prompt(prompt, typography_plan)
        print(f"Clean prompt: {clean_prompt[:100]}...")

        # 需要 injection_data 传递给 Pass 3（含 full_mask）
        image_size = (config.width, config.height)
        injection_data = self.glyph_injector.prepare_injection_from_plan(
            typography_plan, image_size, noise, timesteps,
        )

        image = self._run_pass2_injection_with_data(
            clean_prompt, noise, timesteps, injection_data, config,
        )

        # === Pass 3: 风格化 ===
        if config.use_harmonization:
            print(f"=== Pass 3: {config.harmonizer_type} 风格化 ===")
            image = self._run_pass3_harmonize(
                image, injection_data, typography_plan, config,
            )

        return image

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
        """模板注入去噪：频率分解注入 + 递减强度 + 注意力增强/反向抑制。"""
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

    # ---- Pass 3: 风格化 (harmonizer) ----

    def _run_pass3_harmonize(
        self,
        pass2_image: Image.Image,
        injection_data: dict,
        typography_plan: dict,
        config: GenerationConfig,
    ) -> Image.Image:
        """Pass 3 统一入口：根据 harmonizer_type 分派到 Klein 或 QwenEdit。

        自动管理显存：进入前卸载主管道到 CPU，完成后恢复到 GPU。
        """
        self._offload_pipeline_to_cpu()

        try:
            if config.harmonizer_type == "qwenedit":
                result = self._run_pass3_qwenedit(pass2_image, typography_plan, config)
            else:
                result = self._run_pass3_klein(pass2_image, injection_data, typography_plan, config)
        finally:
            self._offload_harmonizer_to_cpu()
            self._reload_pipeline_to_gpu()

        return result

    def _run_pass3_klein(
        self,
        pass2_image: Image.Image,
        injection_data: dict,
        typography_plan: dict,
        config: GenerationConfig,
    ) -> Image.Image:
        """Pass 3 (Klein): FluxKlein img2img + 二值 mask 混合。"""
        image_analysis = typography_plan.get("image_analysis", {})
        klein_prompt = self.vlm_agent.generate_style_prompt(image_analysis)
        print(f"  VLM 生成提示词: {klein_prompt[:80]}...")

        klein = self.get_klein_generator(config)
        klein_seed = config.klein_seed if config.klein_seed is not None else (config.seed or 42)

        edited = klein.generate(
            prompt=klein_prompt,
            image=pass2_image,
            seed=klein_seed,
            num_inference_steps=config.klein_steps,
            guidance_scale=config.klein_guidance_scale,
        )

        # 确保 RGB 且尺寸一致
        if edited.mode != "RGB":
            edited = edited.convert("RGB")
        if edited.size != pass2_image.size:
            edited = edited.resize(pass2_image.size, Image.LANCZOS)

        # 二值 mask 混合：文字区域用编辑结果，背景保持原图
        binary_mask = injection_data["full_mask"]
        mask = (binary_mask > 127).astype(np.float32)
        if mask.shape[:2] != (pass2_image.height, pass2_image.width):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(pass2_image.size, Image.NEAREST)
            mask = (np.array(mask_pil).astype(np.float32) / 255.0) > 0.5

        mask_3ch = mask[:, :, np.newaxis]
        edit_arr = np.array(edited).astype(np.float32)
        bg_arr = np.array(pass2_image).astype(np.float32)

        result_arr = mask_3ch * edit_arr + (1 - mask_3ch) * bg_arr
        result = Image.fromarray(result_arr.clip(0, 255).astype(np.uint8))

        if self.logger is not None:
            self.logger.save_image(
                Image.fromarray(binary_mask).convert("RGB"), "pass3_mask",
                caption=f"mask coverage={mask.mean():.2%}", subfolder="harmonize",
            )
            comp = Image.new("RGB", (pass2_image.width * 3, pass2_image.height))
            comp.paste(pass2_image, (0, 0))
            comp.paste(edited, (pass2_image.width, 0))
            comp.paste(result, (pass2_image.width * 2, 0))
            self.logger.save_image(comp, "pass3_result",
                caption="Pass2 | Klein | Final", subfolder="harmonize")

        return result

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
            comp = Image.new("RGB", (pass2_image.width * 2, pass2_image.height))
            comp.paste(pass2_image, (0, 0))
            comp.paste(edited, (pass2_image.width, 0))
            self.logger.save_image(comp, "pass3_result",
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
                "background_color": r.get("background_color", "#000000"),
                "is_latex": r.get("is_latex", False),
                "alignment": r.get("alignment", "center"),
            })
        return {
            "image_analysis": {
                "background_style": "debug",
                "dominant_colors": ["#000000"],
                "text_style_hint": "debug mode",
            },
            "text_regions": plan_regions,
        }

    def _save_typography_plan(self, plan: dict) -> None:
        """保存 typography_plan JSON 到 logs 目录。"""
        if self.logger is None:
            return
        plan_dir = self.logger.run_dir / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        self._output_counter += 1
        plan_path = plan_dir / f"typography_plan_{self._output_counter:03d}.json"
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
