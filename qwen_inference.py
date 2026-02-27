"""
QwenImage 推理主入口（对齐 zimage_inference 的 3-pass 架构）

三阶段推理:
  Pass 1: 用完整 prompt 生成参考图 → VLM 自主规划排版
  Pass 2: 从相同噪声用 clean prompt + 字形注入生成文字图
  Pass 3: FluxKlein img2img 风格化

与 zimage_inference 的区别:
  - Pipeline: QwenImagePipeline（packed latents, CFG with true_cfg_scale）
  - 使用 enable_model_cpu_offload 节省显存
  - VAE 需要 latents_mean/latents_std 归一化
"""

import inspect
import os
import sys
import json
import math
from typing import Optional, Union, List
from dataclasses import dataclass, field

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import numpy as np
import torch
from PIL import Image

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer.VLM_agent import VLMAgent, _add_grid_overlay
from infer.glyph_injector import GlyphInjector, InjectionConfig, create_glyph_injector
from infer.mylogger import TTSLogger

DEFAULT_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen-image-2512"


def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps=None, device=None,
                        timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(f"{scheduler.__class__}'s set_timesteps does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    if sigmas is not None:
        if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(f"{scheduler.__class__}'s set_timesteps does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


@dataclass
class QwenGenerationConfig:
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None

    use_prompt_refiner: bool = True
    use_glyph_injection: bool = True
    injection_config: InjectionConfig = field(default_factory=InjectionConfig)

    use_harmonization: bool = True
    harmonizer_type: str = "klein"
    klein_model_path: str = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein"
    klein_steps: int = 10
    klein_guidance_scale: float = 4.0
    klein_seed: Optional[int] = None


class QwenImageInference:
    """QwenImage 三阶段推理（对齐 ZImageInference 接口）"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        logger: TTSLogger = None,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.primary_device = device
        self.logger = logger or TTSLogger(run_name="qwen_inference")
        self._pipeline = None
        self._vlm_agent = None
        self._glyph_injector = None
        self._klein_generator = None
        self._output_counter = 0
        self._current_tag = "000"

    # ---- 延迟加载 ----

    @staticmethod
    def _register_qwen_classes():
        """将 QwenImage 相关类注册到 diffusers 模块（0.29 兼容）。"""
        import transformers.utils as _tu
        if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
            _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

        import diffusers
        if hasattr(diffusers, "QwenImagePipeline"):
            return
        from train.qwen_ip.pipeline_qwenimage import QwenImagePipeline
        from train.qwen_ip.transformer import QwenTransformer2DModel
        from train.qwen_ip.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
        diffusers.QwenImagePipeline = QwenImagePipeline
        diffusers.QwenImageTransformer2DModel = QwenTransformer2DModel
        diffusers.AutoencoderKLQwenImage = AutoencoderKLQwenImage

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._register_qwen_classes()
            import diffusers
            print(f"正在加载 QwenImage 模型到 {self.primary_device} (cpu_offload)...")
            self._pipeline = diffusers.DiffusionPipeline.from_pretrained(
                self.model_path, torch_dtype=self.dtype,
            )
            self._pipeline.enable_model_cpu_offload(
                gpu_id=int(self.primary_device.split(":")[-1]) if ":" in self.primary_device else 0
            )
            print("QwenImage 模型加载完成 (cpu_offload)")
        return self._pipeline

    @property
    def vlm_agent(self) -> VLMAgent:
        if self._vlm_agent is None:
            self._vlm_agent = VLMAgent()
        return self._vlm_agent

    @property
    def glyph_injector(self) -> GlyphInjector:
        if self._glyph_injector is None:
            self._glyph_injector = GlyphInjector(
                vae=self.pipeline.vae,
                scheduler=self.pipeline.scheduler,
                device=self.primary_device,
                dtype=self.dtype,
                logger=self.logger,
            )
        return self._glyph_injector

    def get_klein_generator(self, config):
        if self._klein_generator is None:
            klein_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "fluxklein")
            if klein_dir not in sys.path:
                sys.path.insert(0, klein_dir)
            from inference_fluxklein import FluxKleinGenerator
            print(f"正在加载 FluxKlein: {config.klein_model_path}")
            self._klein_generator = FluxKleinGenerator(
                model_path=config.klein_model_path,
                device=self.primary_device,
                enable_cpu_offload=True,
            )
        return self._klein_generator

    # ---- 主入口 ----

    def generate(
        self,
        prompt: str,
        text_contents: Optional[list[str]] = None,
        text_regions: Optional[list[dict]] = None,
        config: Optional[QwenGenerationConfig] = None,
        run_name: Optional[str] = None,
        **kwargs,
    ) -> Image.Image:
        if config is None:
            config = QwenGenerationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self._output_counter += 1
        self._current_tag = f"{self._output_counter:03d}_{run_name}" if run_name else f"{self._output_counter:03d}"

        working_prompt = prompt
        if config.use_prompt_refiner:
            working_prompt = prompt + ",horizontal text layout."

        generator = None
        if config.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(config.seed)

        has_text = (text_contents and len(text_contents) > 0) or (text_regions and len(text_regions) > 0)

        if config.use_glyph_injection and has_text:
            image = self._generate_with_injection(
                working_prompt, text_contents, config, generator, text_regions,
            )
        else:
            image = self.pipeline(
                prompt=working_prompt,
                height=config.height, width=config.width,
                num_inference_steps=config.num_inference_steps,
                true_cfg_scale=config.true_cfg_scale,
                guidance_scale=config.guidance_scale,
                generator=generator,
            ).images[0]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.logger is not None:
            self.logger.save_image(
                image, f"final_{self._current_tag}",
                caption=f"[{self._current_tag}] {working_prompt[:150]}",
                subfolder="output",
            )
        return image

    # ---- 三阶段推理 ----

    def _generate_with_injection(self, prompt, text_contents, config, generator, text_regions_override):
        candidates: dict[str, Image.Image] = {}
        tag = self._current_tag

        if self._glyph_injector is not None:
            self._glyph_injector.sample_tag = tag

        # Pass 1: 参考图（用 pipeline 直接生成）
        if text_regions_override:
            from zimage_inference import ZImageInference
            typography_plan = ZImageInference._text_regions_to_plan(text_regions_override)
        else:
            print("=== Pass 1: 生成参考图 ===")
            reference_image = self.pipeline(
                prompt=prompt,
                height=config.height, width=config.width,
                num_inference_steps=config.num_inference_steps,
                true_cfg_scale=config.true_cfg_scale,
                guidance_scale=config.guidance_scale,
                generator=generator,
            ).images[0]
            candidates["pass1_reference"] = reference_image

            if self.logger is not None:
                self.logger.save_image(reference_image, f"{tag}_pass1_reference",
                    caption=f"[Pass1] {prompt[:150]}", subfolder="two_pass")

            print("=== VLM 排版规划 ===")
            reference_with_grid = _add_grid_overlay(reference_image)
            if self.logger is not None:
                self.logger.save_image(reference_with_grid, f"{tag}_pass1_with_grid",
                    caption=f"[Grid] {prompt[:150]}", subfolder="two_pass")

            typography_plan = self.vlm_agent.analyze_typography(
                reference_image, prompt, text_contents,
            )

        self._save_typography_plan(typography_plan)

        # Pass 2: Clean 推理 + 字形注入
        print("=== Pass 2: Clean 推理 + 字形合成 ===")
        clean_prompt = self.vlm_agent.generate_clean_prompt(prompt, typography_plan)
        print(f"Clean prompt: {clean_prompt}...")

        # 准备噪声和注入数据
        pipe = self.pipeline
        image_size = (config.width, config.height)
        latent_h = 2 * (config.height // (pipe.vae_scale_factor * 2))
        latent_w = 2 * (config.width // (pipe.vae_scale_factor * 2))
        num_channels = pipe.transformer.config.in_channels // 4

        noise = torch.randn(
            (1, 1, num_channels, latent_h, latent_w),
            generator=generator, device="cpu", dtype=torch.float32,
        )

        # scheduler setup（复制 pipeline 的 mu-shift 逻辑）
        packed_noise = pipe._pack_latents(noise, 1, num_channels, latent_h, latent_w)
        sigmas = np.linspace(1.0, 1 / config.num_inference_steps, config.num_inference_steps)
        image_seq_len = packed_noise.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = _retrieve_timesteps(pipe.scheduler, config.num_inference_steps, "cpu", sigmas=sigmas, mu=mu)

        self.glyph_injector.sample_tag = tag
        injection_data = self.glyph_injector.prepare_injection_from_plan(
            typography_plan, image_size, noise.squeeze(1), timesteps,
        )

        # Mask 全黑检测
        if injection_data["full_mask"].max() == 0:
            print("  [WARN] glyph mask 全黑，回退到 pass1")
            self._save_candidates_concat(candidates)
            return candidates.get("pass1_reference", list(candidates.values())[-1])

        # Pass 2 去噪
        background = self._run_pass2_denoising(clean_prompt, noise, config, injection_data)

        pass2_image = self._pixel_composite_text(background, injection_data)
        candidates["pass2_injection"] = pass2_image

        # Pass 3: 风格化（先卸载 QwenImage pipeline 腾显存）
        if config.use_harmonization:
            print(f"=== Pass 3: Klein 风格化 ===")
            self._offload_pipeline()
            pass3_image = self._run_pass3_klein(pass2_image, injection_data, typography_plan, config)
            if pass3_image is not None:
                candidates["pass3_klein"] = pass3_image

        self._save_candidates_concat(candidates)

        # VLM 选优
        all_text = " ".join(r["content"] for r in typography_plan.get("text_regions", []))
        pool = [(k, v) for k, v in candidates.items() if k != "pass1_reference"]
        if len(pool) > 1:
            names, images = zip(*pool)
            best_idx = self.vlm_agent.select_best_text_match(list(images), all_text)
            print(f"  VLM 选优: {names[best_idx]}")
            return images[best_idx]
        return pool[-1][1] if pool else candidates.get("pass1_reference")

    # ---- Pass 2 去噪（QwenImage 特有的 packed latent + CFG） ----

    def _run_pass2_denoising(self, clean_prompt, noise, config, injection_data):
        pipe = self.pipeline
        device = pipe._execution_device
        dtype = pipe.transformer.dtype

        latent_h = noise.shape[3]
        latent_w = noise.shape[4]
        num_channels = noise.shape[2]

        # Encode prompt
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            clean_prompt, device=device, max_sequence_length=512)
        neg_embeds, neg_mask = pipe.encode_prompt(
            "", device=device, max_sequence_length=512)

        # Pack noise
        latent = pipe._pack_latents(
            noise.to(device=device, dtype=dtype), 1, num_channels, latent_h, latent_w)

        # Scheduler
        sigmas = np.linspace(1.0, 1 / config.num_inference_steps, config.num_inference_steps)
        image_seq_len = latent.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = _retrieve_timesteps(pipe.scheduler, config.num_inference_steps, device, sigmas=sigmas, mu=mu)

        img_shapes = [[(1, latent_h // 2, latent_w // 2)]]
        txt_seq_lens = [prompt_embeds.shape[1]]
        neg_txt_seq_lens = [neg_embeds.shape[1]]

        # Guidance
        guidance = None
        if pipe.transformer.config.guidance_embeds and config.guidance_scale is not None:
            guidance = torch.full([1], config.guidance_scale, device=device, dtype=torch.float32)

        # Denoising loop with glyph injection
        pipe.scheduler.set_begin_index(0)
        for step_idx, t in enumerate(timesteps):
            timestep = t.expand(latent.shape[0]).to(latent.dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent.to(dtype),
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_seq_lens=txt_seq_lens,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]

            if config.true_cfg_scale > 1:
                neg_pred = pipe.transformer(
                    hidden_states=latent.to(dtype),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=neg_embeds,
                    txt_seq_lens=neg_txt_seq_lens,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                comb = neg_pred + config.true_cfg_scale * (noise_pred - neg_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred = comb * (cond_norm / comb_norm)

            latent = pipe.scheduler.step(noise_pred, t, latent, return_dict=False)[0]

            # Glyph injection: unpack → inject → repack
            if config.use_glyph_injection:
                spatial = pipe._unpack_latents(latent, config.height, config.width, pipe.vae_scale_factor)
                spatial_4d = spatial.squeeze(2)  # (B, C, H, W)
                spatial_4d = self.glyph_injector.inject_latent(
                    spatial_4d, injection_data, step_idx + 1,
                    config=config.injection_config,
                )
                spatial = spatial_4d.unsqueeze(2)  # (B, C, 1, H, W)
                latent = pipe._pack_latents(spatial, 1, num_channels, latent_h, latent_w)

        # Decode
        return self._decode_latent(latent, config.height, config.width)

    def _decode_latent(self, latent, height, width):
        pipe = self.pipeline
        latent = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
        latent = latent.to(pipe.vae.dtype)

        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std_inv = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        latent = latent / latents_std_inv + latents_mean

        with torch.no_grad():
            image = pipe.vae.decode(latent, return_dict=False)[0][:, :, 0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    # ---- 像素合成（同 ZImageInference） ----

    def _pixel_composite_text(self, background, injection_data):
        template = injection_data["combined_template"]
        mask = injection_data["full_mask"]
        bg_arr = np.array(background)
        tpl_arr = np.array(template)
        if tpl_arr.shape[:2] != bg_arr.shape[:2]:
            template = template.resize(background.size, Image.LANCZOS)
            tpl_arr = np.array(template)
        if mask.shape[:2] != bg_arr.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize(background.size, Image.NEAREST))
        result = bg_arr.copy()
        result[mask > 127] = tpl_arr[mask > 127]
        return Image.fromarray(result)

    def _offload_pipeline(self):
        """将 QwenImage pipeline 的所有模型组件移到 CPU 并释放显存。"""
        if self._pipeline is None:
            return
        for name in ("transformer", "vae", "text_encoder"):
            comp = getattr(self._pipeline, name, None)
            if comp is not None:
                comp.to("cpu")
        self._glyph_injector = None
        torch.cuda.empty_cache()
        print("  QwenImage pipeline 已卸载到 CPU")

    # ---- Pass 3: Klein ----

    def _run_pass3_klein(self, pass2_image, injection_data, typography_plan, config):
        image_analysis = typography_plan.get("image_analysis", {})
        klein_prompt = self.vlm_agent.generate_style_prompt(image_analysis)
        klein = self.get_klein_generator(config)
        klein_seed = config.klein_seed or config.seed or 42

        template = injection_data["combined_template"]
        if template.size != pass2_image.size:
            template = template.resize(pass2_image.size, Image.LANCZOS)

        result = klein.pipe(
            prompt=klein_prompt,
            image=[pass2_image, template],
            generator=torch.Generator(device=klein.device).manual_seed(klein_seed),
            num_inference_steps=config.klein_steps,
            guidance_scale=config.klein_guidance_scale,
        ).images[0]

        if result.mode != "RGB":
            result = result.convert("RGB")
        if result.size != pass2_image.size:
            result = result.resize(pass2_image.size, Image.LANCZOS)
        return result

    # ---- 工具方法 ----

    _CAT_IMG_DIR = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/logs/CAT_IMG_QWEN"

    def _save_candidates_concat(self, candidates):
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
        concat.save(os.path.join(self._CAT_IMG_DIR, f"{self._current_tag}.jpg"),
                     format="JPEG", quality=90)

    def _save_typography_plan(self, plan):
        if self.logger is None:
            return
        plan_dir = self.logger.run_dir / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        with open(plan_dir / f"typography_plan_{self._current_tag}.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

    def __call__(self, prompt, text_contents=None, text_regions=None, **kwargs):
        return self.generate(prompt, text_contents, text_regions, **kwargs)
