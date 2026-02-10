"""
Z-Image 推理主入口

整合核心接口:
- VLMAgent: 统一的 VLM 调用中心（排版分析、prompt 改写、评分）
- GlyphInjector: 文字渲染和 latent 注入
- TestTimeScaling: Beam search 策略的测试时缩放

三阶段推理架构:
  Pass 1: 用完整 prompt 生成参考图 → VLM 自主规划排版
  Pass 2: 从相同噪声用 clean prompt + 字形注入生成文字图
  Pass 3: FluxKlein img2img + soft mask 背景融合循环 → VLM 评判直到风格融合达标
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
from infer.VLM_agent import VLMAgent
from infer.glyph_injector import GlyphInjector, TextRegion, InjectionConfig, create_glyph_injector
from infer.test_time_scaling import TestTimeScaling, create_test_time_scaling
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

    # Test Time Scaling
    use_tts: bool = False
    beam_size: int = 8
    early_stop_step: int = 3
    keep_ratio: float = 0.25

    # Pass 3: FluxKlein img2img refine（将粘贴的字体与背景风格融合）
    use_harmonization: bool = True             # 启用 Pass 3 FluxKlein 背景融合
    klein_model_path: str = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein"
    klein_prompt: str = (
        "The text on the surface is rewritten in beautiful style, "
        "with natural texture, soft edges, and subtle imperfections. "
        "The lettering harmonizes perfectly with the background aesthetic. "
        "No other changes to the image."
    )
    klein_steps: int = 50                      # FluxKlein 推理步数
    klein_guidance_scale: float = 4.0          # FluxKlein guidance scale
    klein_seed: Optional[int] = None           # FluxKlein seed（None=跟随主 seed）
    klein_max_iters: int = 3                   # FluxKlein refine 最大循环次数
    klein_target_score: float = 9.5            # VLM 评分阈值
    klein_blur_radius: int = 8                 # soft mask 高斯模糊半径
    klein_enable_cpu_offload: bool = False      # FluxKlein CPU offload


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
        self._tts = None
        self._klein_generator = None
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

    @property
    def tts(self) -> TestTimeScaling:
        if self._tts is None:
            self._tts = create_test_time_scaling(
                self.pipeline,
                vlm_agent=self.vlm_agent,
                device=self.primary_device,
                logger=self.logger,
            )
        return self._tts

    def get_klein_generator(self, config: "GenerationConfig"):
        """延迟加载 FluxKlein 生成器（按需创建，避免浪费显存）。"""
        if self._klein_generator is None:
            # 动态导入，避免不使用时也依赖 FluxKlein
            klein_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "fluxklein")
            if klein_dir not in sys.path:
                sys.path.insert(0, klein_dir)
            from inference_fluxklein import FluxKleinGenerator

            print(f"正在加载 FluxKlein 模型: {config.klein_model_path}")
            self._klein_generator = FluxKleinGenerator(
                model_path=config.klein_model_path,
                device=self.primary_device,
                enable_cpu_offload=config.klein_enable_cpu_offload,
            )
            print("FluxKlein 模型加载完成")
        return self._klein_generator

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

        # 2. TTS 或普通生成
        if config.use_tts:
            image, score = self.tts.generate_with_beam_search(
                prompt=working_prompt,
                text_content=text_content_str,
                height=config.height, width=config.width,
                num_inference_steps=config.num_inference_steps,
                beam_size=config.beam_size,
                early_stop_step=config.early_stop_step,
                keep_ratio=config.keep_ratio,
                seed=config.seed,
            )
            print(f"TTS 最终得分: {score:.2f}")
            return image

        # 3. 普通生成（可能带 Glyph Injection 两阶段推理）
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
        clean_prompt = self.vlm_agent.generate_clean_prompt(prompt)
        print(f"Clean prompt: {clean_prompt[:100]}...")

        # 需要 injection_data 传递给 Pass 3（含 full_mask）
        image_size = (config.width, config.height)
        injection_data = self.glyph_injector.prepare_injection_from_plan(
            typography_plan, image_size, noise, timesteps,
        )

        image = self._run_pass2_injection_with_data(
            clean_prompt, noise, timesteps, injection_data, config,
        )

        # === Pass 3: FluxKlein img2img + Soft Mask 背景融合 ===
        text_content_str = None
        if text_contents:
            text_content_str = " ".join(text_contents)
        elif typography_plan.get("text_regions"):
            text_content_str = " ".join(
                r.get("content", "") for r in typography_plan["text_regions"]
            )

        if config.use_harmonization:
            print("=== Pass 3: FluxKlein img2img refine ===")
            image = self._run_pass3_klein_refine(
                image, injection_data, clean_prompt, config,
                text_content_str=text_content_str,
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

    # ---- Pass 3: FluxKlein img2img + soft mask 背景融合 ----

    @staticmethod
    def _compute_soft_mask(
        binary_mask: np.ndarray, blur_radius: int = 8,
    ) -> np.ndarray:
        """从二值化 mask 生成高斯模糊 soft mask。

        Args:
            binary_mask: uint8 (H, W)，文字区域 255
            blur_radius: 高斯模糊半径

        Returns:
            float32 (H, W)，值域 [0, 1]，1.0=文字区域
        """
        from PIL import ImageFilter

        mask_arr = binary_mask.astype(np.float32) / 255.0
        if blur_radius > 0:
            mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask_arr = np.array(mask_pil).astype(np.float32) / 255.0
        return mask_arr

    def _run_pass3_klein_refine(
        self,
        pass2_image: Image.Image,
        injection_data: dict,
        clean_prompt: str,
        config: GenerationConfig,
        text_content_str: Optional[str] = None,
    ) -> Image.Image:
        """Pass 3: 使用 FluxKlein img2img 将粘贴字体与背景风格融合。

        每次迭代：
          1. 将当前图像送入 FluxKlein 做 img2img 编辑
          2. 用 soft mask 混合：文字区域用 FluxKlein 输出，背景保持原图
          3. VLM 评分；满足阈值或达到最大次数后停止

        Args:
            pass2_image: Pass 2 生成的图像（含粘贴的字体）
            injection_data: Pass 2 的注入数据（含 full_mask）
            clean_prompt: Pass 2 使用的 clean prompt
            config: 生成配置
            text_content_str: 文本内容（用于 VLM 评分）

        Returns:
            融合后的最终图像
        """
        max_iters = config.klein_max_iters
        target = config.klein_target_score
        blur_radius = config.klein_blur_radius
        klein_seed = config.klein_seed if config.klein_seed is not None else (config.seed or 42)

        # 获取 FluxKlein 生成器
        klein = self.get_klein_generator(config)

        # 生成 soft mask（使用高斯模糊平滑边缘）
        binary_mask = injection_data["full_mask"]  # uint8 (H, W)
        soft_mask = self._compute_soft_mask(binary_mask, blur_radius)
        soft_mask_3ch = soft_mask[:, :, np.newaxis]  # (H, W, 1)

        # 保存 soft mask 可视化
        if self.logger is not None:
            sm_vis = (soft_mask * 255).astype(np.uint8)
            self.logger.save_image(
                Image.fromarray(sm_vis).convert("RGB"),
                "pass3_soft_mask",
                caption=f"soft mask  blur_radius={blur_radius}  coverage={soft_mask.mean():.3f}",
                subfolder="harmonize",
            )

        current_image = pass2_image
        best_image = pass2_image
        best_score = 0.0

        # 保存 Pass 2 原图作为背景参考（mask 外区域始终来自此图）
        background_image = pass2_image

        print(f"  FluxKlein refine prompt: {config.klein_prompt[:100]}...")
        print(f"  FluxKlein 参数: steps={config.klein_steps}, guidance={config.klein_guidance_scale}, "
              f"seed={klein_seed}, max_iters={max_iters}")

        for it in range(max_iters):
            print(f"  Pass 3 Klein [{it + 1}/{max_iters}]", end="")

            # 1. FluxKlein img2img 编辑
            klein_output_path = None
            if self.logger is not None:
                klein_dir = self.logger.run_dir / "harmonize"
                klein_dir.mkdir(parents=True, exist_ok=True)
                klein_output_path = str(klein_dir / f"klein_raw_iter{it + 1}.png")
            else:
                klein_output_path = f"/tmp/klein_raw_iter{it + 1}.png"

            edited = klein.generate(
                prompt=config.klein_prompt,
                image=current_image,
                seed=klein_seed + it,  # 每次迭代用不同 seed
                num_inference_steps=config.klein_steps,
                guidance_scale=config.klein_guidance_scale,
                output_path=klein_output_path,
            )

            # 2. Soft mask 混合：文字区域用 FluxKlein 编辑结果，背景保持 Pass 2 原图
            #    确保尺寸匹配
            if edited.size != background_image.size:
                edited = edited.resize(background_image.size, Image.LANCZOS)

            edit_arr = np.array(edited).astype(np.float32)
            bg_arr = np.array(background_image).astype(np.float32)

            # 如果 mask 尺寸不匹配，resize
            mask_for_blend = soft_mask_3ch
            if soft_mask.shape[0] != edited.height or soft_mask.shape[1] != edited.width:
                mask_pil = Image.fromarray((soft_mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(edited.size, Image.LANCZOS)
                mask_resized = np.array(mask_pil).astype(np.float32) / 255.0
                mask_for_blend = mask_resized[:, :, np.newaxis]

            blended_arr = mask_for_blend * edit_arr + (1 - mask_for_blend) * bg_arr
            result_image = Image.fromarray(blended_arr.clip(0, 255).astype(np.uint8))

            # 3. VLM 评分
            score = self.vlm_agent.score_image(
                result_image, clean_prompt, text_content_str,
            )
            print(f"  score={score:.2f}/10")

            if score > best_score:
                best_score = score
                best_image = result_image

            # 日志
            if self.logger is not None:
                self.logger.save_image(
                    result_image, f"pass3_iter{it + 1}",
                    caption=f"iter={it + 1}  score={score:.2f}  "
                            f"steps={config.klein_steps}  guidance={config.klein_guidance_scale}",
                    subfolder="harmonize",
                )
                # 保存对比图：原图 | FluxKlein raw | masked blending
                comparison = Image.new("RGB", (current_image.width * 3, current_image.height))
                comparison.paste(current_image, (0, 0))
                comparison.paste(edited.resize(current_image.size, Image.LANCZOS), (current_image.width, 0))
                comparison.paste(result_image.resize(current_image.size, Image.LANCZOS), (current_image.width * 2, 0))
                self.logger.save_image(
                    comparison, f"pass3_comparison_iter{it + 1}",
                    caption=f"input | Klein raw | masked blend  score={score:.2f}",
                    subfolder="harmonize",
                )

            if score >= target:
                print(f"  Pass 3 达到目标分数 {score:.2f} >= {target}")
                break

            # 下一轮以混合后的结果作为输入
            current_image = result_image

        print(f"  Pass 3 完成，最佳分数 {best_score:.2f}/10")
        return best_image

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
            "use_tts": config.use_tts,
            "beam_size": config.beam_size,
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
