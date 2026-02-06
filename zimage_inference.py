"""
Z-Image 推理主入口

整合三个核心接口:
- PromptRefiner: 基于 VLM API 的 prompt 优化
- GlyphInjector: 文字渲染和 latent 注入
- TestTimeScaling: Beam search 策略的测试时缩放

支持 8 卡并行运行
"""

import os
import sys
from typing import Optional, Union
from dataclasses import dataclass, field

# 禁用 torch.compile 避免错误
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch
import torch.multiprocessing as mp
from PIL import Image

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.zimage_ip.pipeline_z_image import ZImagePipeline
from infer.prompt_refiner import PromptRefiner, refine_prompt
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
    renoise_ratio: float = 0.5


class ZImageInference:
    """
    Z-Image 推理类
    
    整合 PromptRefiner, GlyphInjector, TestTimeScaling 三个接口
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Union[str, list[str]] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        logger: TTSLogger = None,
    ):
        """
        初始化
        
        Args:
            model_path: 模型路径
            device: 设备，支持单卡 "cuda:0" 或多卡 ["cuda:0", "cuda:1", ...]
            dtype: 数据类型
            logger: TTSLogger 实例，为 None 时自动创建
        """
        self.model_path = model_path
        self.dtype = dtype
        self.logger = logger or TTSLogger(run_name="zimage_inference")
        
        # 处理设备列表
        if isinstance(device, str):
            self.devices = [device]
        else:
            self.devices = device
        self.primary_device = self.devices[0]
        
        # 延迟加载模型
        self._pipeline = None
        self._prompt_refiner = None
        self._glyph_injector = None
        self._tts = None
        
    @property
    def pipeline(self) -> ZImagePipeline:
        """获取 pipeline（延迟加载）"""
        if self._pipeline is None:
            print(f"正在加载 Z-Image 模型到 {self.primary_device}...")
            self._pipeline = ZImagePipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,
            )
            self._pipeline.to(self.primary_device)
            print("模型加载完成")
        return self._pipeline
    
    @property
    def prompt_refiner(self) -> PromptRefiner:
        """获取 PromptRefiner"""
        if self._prompt_refiner is None:
            self._prompt_refiner = PromptRefiner()
        return self._prompt_refiner
    
    @property
    def glyph_injector(self) -> GlyphInjector:
        """获取 GlyphInjector"""
        if self._glyph_injector is None:
            self._glyph_injector = create_glyph_injector(
                self.pipeline, 
                device=self.primary_device,
                logger=self.logger,
            )
        return self._glyph_injector
    
    @property
    def tts(self) -> TestTimeScaling:
        """获取 TestTimeScaling"""
        if self._tts is None:
            self._tts = create_test_time_scaling(
                self.pipeline,
                self.prompt_refiner,
                device=self.primary_device,
                logger=self.logger,
            )
        return self._tts
    
    def generate(
        self,
        prompt: str,
        text_regions: Optional[list[dict]] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Image.Image:
        """
        生成图像
        
        Args:
            prompt: 生成 prompt
            text_regions: 文字区域列表，格式 [{"bbox": [x1, y1, x2, y2], "content": "文字"}]
            config: 生成配置
            **kwargs: 额外参数，会覆盖 config 中的设置
            
        Returns:
            生成的 PIL Image
        """
        if config is None:
            config = GenerationConfig()
        
        # 使用 kwargs 覆盖 config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 提取文字内容
        text_content = None
        if text_regions:
            text_content = " ".join([r.get("content", "") for r in text_regions])
        
        # 1. Prompt 优化
        working_prompt = prompt
        if config.use_prompt_refiner and text_content:
            refined_prompts = self.prompt_refiner.refine(
                prompt, 
                text_content=text_content,
                num_variants=1
            )
            working_prompt = refined_prompts[0]
            print(f"优化后的 prompt: {working_prompt[:100]}...")
        
        # 2. 使用 TTS 或普通生成
        if config.use_tts and len(self.devices) >= 1:
            image, score = self.tts.generate_with_beam_search(
                prompt=working_prompt,
                text_content=text_content,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                beam_size=config.beam_size,
                early_stop_step=config.early_stop_step,
                keep_ratio=config.keep_ratio,
                renoise_ratio=config.renoise_ratio,
                seed=config.seed
            )
            print(f"TTS 最终得分: {score:.2f}")
            return image
        
        # 3. 普通生成（可能带 Glyph Injection）
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.primary_device).manual_seed(config.seed)
        
        if config.use_glyph_injection and text_regions:
            # 带文字注入的生成
            return self._generate_with_injection(
                prompt=working_prompt,
                text_regions=text_regions,
                config=config,
                generator=generator
            )
        else:
            # 普通生成
            result = self.pipeline(
                prompt=working_prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator
            ).images[0]
            return result
    
    def _generate_with_injection(
        self,
        prompt: str,
        text_regions: list[dict],
        config: GenerationConfig,
        generator: Optional[torch.Generator] = None
    ) -> Image.Image:
        """带 Glyph Injection 的生成
        
        局部重采样算法 (num_local_samples > 1):
            在 text-region 区域采样 K 个不同种子的噪声，区域外共享原始噪声。
            timestep_ratio 以内：K 个分支独立去噪，每步将 text-region 替换为 K 个结果的平均。
            timestep_ratio 以后：合并为单分支正常去噪。
        
        模板注入模式 (num_local_samples <= 1):
            使用原有的 text latent inversion + mask 混合注入。
        """
        # 转换文字区域格式
        regions = [
            TextRegion(bbox=tuple(r["bbox"]), content=r["content"])
            for r in text_regions
        ]
        
        # 准备 latent
        latent_height = 2 * (config.height // (self.pipeline.vae_scale_factor * 2))
        latent_width = 2 * (config.width // (self.pipeline.vae_scale_factor * 2))
        num_channels = self.pipeline.transformer.in_channels
        
        noise = torch.randn(
            (1, num_channels, latent_height, latent_width),
            generator=generator,
            device=self.primary_device,
            dtype=torch.float32
        )
        
        # 设置 scheduler
        self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.primary_device)
        timesteps = self.pipeline.scheduler.timesteps
        
        # 准备注入数据（mask + text latent）
        injection_data = self.glyph_injector.prepare_injection(
            text_regions=regions,
            image_size=(config.width, config.height),
            noise=noise,
            timesteps=timesteps
        )
        
        # 编码 prompt
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.primary_device,
            do_classifier_free_guidance=False
        )
        
        K = config.injection_config.num_local_samples
        total_steps = len(timesteps)
        inject_until_step = int(total_steps * config.injection_config.timestep_ratio)
        mask = injection_data["mask_latent"]       # (1, 1, h, w)
        mask_exp = mask.expand_as(noise)            # (1, C, h, w)
        
        if K > 1 and inject_until_step > 0:
            latent = self._denoise_local_resample(
                noise, timesteps, prompt_embeds,
                mask_exp, K, inject_until_step, config
            )
        else:
            latent = self._denoise_template_inject(
                noise, timesteps, prompt_embeds,
                injection_data, config
            )
        
        # 解码
        latent = latent.to(self.pipeline.vae.dtype)
        latent = (latent / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        
        with torch.no_grad():
            image = self.pipeline.vae.decode(latent, return_dict=False)[0]
        
        image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]
        return image
    
    # ------ 局部重采样 ------
    
    def _denoise_local_resample(
        self,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: list,
        mask_exp: torch.Tensor,
        K: int,
        inject_until_step: int,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """
        局部重采样去噪：text-region 用 K 个噪声分支，每步取平均。
        
        Args:
            noise: 主干噪声 (1, C, H, W)
            timesteps: scheduler 时间步
            prompt_embeds: 编码后的 prompt
            mask_exp: 扩展到 latent 维度的 text-region mask (1, C, H, W)
            K: 分支数
            inject_until_step: 局部重采样截止步数
            config: 生成配置
        """
        base_seed = config.seed or 0
        dtype = self.pipeline.transformer.dtype
        
        # ---------- 构造 K 个分支 ----------
        # 区域外共享 noise，区域内各用不同种子
        branches = []
        for k in range(K):
            gen_k = torch.Generator(device=self.primary_device).manual_seed(base_seed + 1000 + k)
            local_noise = torch.randn_like(noise, generator=gen_k)
            branch = noise * (1 - mask_exp) + local_noise * mask_exp
            branches.append(branch)
        
        print(f"局部重采样: K={K}, inject_until_step={inject_until_step}/{len(timesteps)}")
        
        # ---------- 阶段 1: 多分支去噪 ----------
        for step_idx, t in enumerate(timesteps[:inject_until_step]):
            # 批量前向：K 个 latent 拼成一个 batch
            batch = torch.cat(branches, dim=0)                # (K, C, H, W)
            timestep_batch = t.expand(K)
            timestep_norm = (1000 - timestep_batch) / 1000
            
            latent_input = batch.to(dtype).unsqueeze(2)       # (K, C, 1, H, W)
            latent_list = list(latent_input.unbind(dim=0))    # K 个 (C, 1, H, W)
            prompt_embeds_k = prompt_embeds * K               # 复制 K 份
            
            with torch.no_grad():
                model_out = self.pipeline.transformer(
                    latent_list, timestep_norm, prompt_embeds_k, return_dict=False
                )[0]
            
            # scheduler.step K 次，但只推进 step_index 一次
            if self.pipeline.scheduler._step_index is None:
                self.pipeline.scheduler._init_step_index(t)
            saved_idx = self.pipeline.scheduler._step_index
            
            for k in range(K):
                self.pipeline.scheduler._step_index = saved_idx
                noise_pred_k = -model_out[k].float().unsqueeze(0)
                branches[k] = self.pipeline.scheduler.step(
                    noise_pred_k, t, branches[k], return_dict=False
                )[0]
            # 最后一次 step 已将 _step_index 推进到 saved_idx + 1
            
            # 文字区域取 K 个分支的平均，然后回写到每个分支
            avg_region = torch.stack(branches, dim=0).mean(dim=0)   # (1, C, H, W)
            for k in range(K):
                branches[k] = branches[k] * (1 - mask_exp) + avg_region * mask_exp
        
        # 取第 0 分支作为合并后的 latent（此时所有分支完全一致）
        latent = branches[0]
        
        # ---------- 阶段 2: 单分支正常去噪 ----------
        for step_idx, t in enumerate(timesteps[inject_until_step:]):
            timestep = t.expand(1)
            timestep_norm = (1000 - timestep) / 1000
            
            latent_input = latent.to(dtype).unsqueeze(2)
            
            with torch.no_grad():
                model_out = self.pipeline.transformer(
                    [latent_input[0]], timestep_norm, prompt_embeds, return_dict=False
                )[0]
            
            noise_pred = -torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
            latent = self.pipeline.scheduler.step(
                noise_pred.to(torch.float32), t, latent, return_dict=False
            )[0]
        
        return latent
    
    # ------ 模板注入（原有逻辑） ------
    
    def _denoise_template_inject(
        self,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: list,
        injection_data: dict,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """原有的 text-latent 模板注入去噪"""
        dtype = self.pipeline.transformer.dtype
        latent = noise.clone()
        
        for step_idx, t in enumerate(timesteps):
            timestep = t.expand(1)
            timestep_norm = (1000 - timestep) / 1000
            
            latent_input = latent.to(dtype).unsqueeze(2)
            
            with torch.no_grad():
                model_out = self.pipeline.transformer(
                    [latent_input[0]], timestep_norm, prompt_embeds, return_dict=False
                )[0]
            
            noise_pred = -torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
            latent = self.pipeline.scheduler.step(
                noise_pred.to(torch.float32), t, latent, return_dict=False
            )[0]
            
            if config.use_glyph_injection:
                latent = self.glyph_injector.inject_latent(
                    latent, injection_data, step_idx + 1,
                    config=config.injection_config
                )
        
        return latent
    
    def __call__(
        self,
        prompt: str,
        text_regions: Optional[list[dict]] = None,
        **kwargs
    ) -> Image.Image:
        """调用接口"""
        return self.generate(prompt, text_regions, **kwargs)


# ============ 多 GPU 并行支持 ============

def _worker_init(rank: int, model_path: str, device: str, dtype: torch.dtype):
    """工作进程初始化"""
    global _worker_inference
    _worker_inference = ZImageInference(model_path, device=device, dtype=dtype)
    print(f"Worker {rank} 初始化完成，设备: {device}")


def _worker_generate(args: tuple) -> tuple:
    """工作进程生成函数"""
    global _worker_inference
    prompt, text_regions, config_dict, worker_id = args
    
    config = GenerationConfig(**config_dict)
    # 为每个 worker 使用不同的 seed
    if config.seed is not None:
        config.seed = config.seed + worker_id
    
    image = _worker_inference.generate(prompt, text_regions, config)
    return worker_id, image


class ParallelZImageInference:
    """
    多 GPU 并行推理
    
    支持在多张 GPU 上同时生成不同样本
    """
    
    def __init__(
        self,
        model_path: str,
        devices: list[str] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        初始化
        
        Args:
            model_path: 模型路径
            devices: GPU 设备列表，默认使用所有可用 GPU
            dtype: 数据类型
        """
        self.model_path = model_path
        self.dtype = dtype
        
        if devices is None:
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
        self.devices = devices
        self.num_workers = len(devices)
        
        print(f"并行推理初始化，使用 {self.num_workers} 张 GPU: {devices}")
        
        # 创建进程池
        self._pool = None
        
    def _ensure_pool(self):
        """确保进程池已创建"""
        if self._pool is None:
            mp.set_start_method('spawn', force=True)
            self._pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(0, self.model_path, self.devices[0], self.dtype)
            )
    
    def generate_batch(
        self,
        prompts: list[str],
        text_regions_list: Optional[list[list[dict]]] = None,
        config: Optional[GenerationConfig] = None
    ) -> list[Image.Image]:
        """
        批量生成图像
        
        Args:
            prompts: prompt 列表
            text_regions_list: 每个 prompt 对应的文字区域列表
            config: 生成配置
            
        Returns:
            生成的图像列表
        """
        if config is None:
            config = GenerationConfig()
        
        config_dict = {
            'height': config.height,
            'width': config.width,
            'num_inference_steps': config.num_inference_steps,
            'guidance_scale': config.guidance_scale,
            'seed': config.seed,
            'use_prompt_refiner': config.use_prompt_refiner,
            'use_glyph_injection': config.use_glyph_injection,
            'use_tts': config.use_tts,
            'beam_size': config.beam_size,
        }
        
        if text_regions_list is None:
            text_regions_list = [None] * len(prompts)
        
        # 准备任务
        tasks = [
            (prompts[i], text_regions_list[i], config_dict, i)
            for i in range(len(prompts))
        ]
        
        # 使用单 GPU 串行处理（简化版本）
        # 完整的并行版本需要更复杂的进程管理
        results = []
        inference = ZImageInference(self.model_path, device=self.devices[0], dtype=self.dtype)
        
        for i, (prompt, text_regions, _, worker_id) in enumerate(tasks):
            device_idx = i % len(self.devices)
            if device_idx != 0:
                # 切换设备（需要重新加载模型，这里简化处理）
                pass
            
            cfg = GenerationConfig(**config_dict)
            if cfg.seed is not None:
                cfg.seed = cfg.seed + worker_id
            
            image = inference.generate(prompt, text_regions, cfg)
            results.append(image)
            print(f"完成 {i+1}/{len(tasks)}")
        
        return results
    
    def close(self):
        """关闭进程池"""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def create_inference(
    model_path: str,
    device: Union[str, list[str]] = "cuda",
    parallel: bool = False,
    dtype: torch.dtype = torch.bfloat16
) -> Union[ZImageInference, ParallelZImageInference]:
    """
    创建推理实例
    
    Args:
        model_path: 模型路径
        device: 设备
        parallel: 是否使用并行推理
        dtype: 数据类型
        
    Returns:
        推理实例
    """
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
    # 简单测试
    print("Z-Image Inference 模块加载成功")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
