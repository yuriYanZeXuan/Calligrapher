"""
Test Time Scaling: 基于 Beam Search 的测试时缩放接口

支持多 GPU 并行：每个 GPU 处理不同的 beam 候选
"""

import os
from typing import Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

from infer.mylogger import TTSLogger
from infer.glyph_injector import InjectionConfig

# 加载环境变量
load_dotenv(Path(__file__).parent.parent / '.env')

# VLM API 配置
API_KEY = os.getenv("QST_API_KEY")
BASE_URL = os.getenv("QST_BASE_URL")
VLM_MODEL = "qwen3-vl-235b-a22b-instruct"


@dataclass
class BeamCandidate:
    """Beam search 候选项"""
    latent: torch.Tensor        # 当前步的含噪 latent
    noise: torch.Tensor
    prompt: str
    prompt_embeds: list
    score: float = 0.0
    step: int = 0
    device: str = "cuda:0"
    latent_0: torch.Tensor = None   # 模型预测的干净 latent (x0)
    injection_data: dict = None     # GlyphInjector 的注入数据


def get_vlm_client() -> OpenAI:
    """获取 VLM 客户端"""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _encode_image_b64(image: Image.Image) -> str:
    """PIL Image → base64 字符串"""
    import base64
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def score_image_with_vlm(
    image: Image.Image,
    prompt: str,
    text_content: Optional[str] = None
) -> float:
    """使用 VLM 对单张图像进行绝对评分（0-10）"""
    img_b64 = _encode_image_b64(image)
    
    system_prompt = """你是一个图像质量评估专家。请根据以下标准对图像进行评分：
1. 图像整体质量（清晰度、色彩、构图）：0-3分
2. 与 prompt 描述的符合程度：0-4分  
3. 如果有文字内容要求，文字的准确性和可读性：0-3分

请只输出一个 0-10 之间的数字分数，不要有任何其他内容。"""
    
    user_content = f"Prompt: {prompt}"
    if text_content:
        user_content += f"\n期望的文字内容: {text_content}"
    
    try:
        client = get_vlm_client()
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=16,
            temperature=0.1
        )
        return min(max(float(response.choices[0].message.content.strip()), 0.0), 10.0)
    except Exception as e:
        print(f"VLM 评分失败: {e}")
        return 5.0


def rank_images_with_vlm(
    images: list[Image.Image],
    prompt: str,
    text_content: Optional[str] = None,
    max_score: float = 10.0,
) -> list[float]:
    """让 VLM 对所有图片排名，然后按名次阶梯给分
    
    第 1 名 → max_score，之后均匀递减
    例如 4 张图: 10.0, 7.5, 5.0, 2.5
    
    Returns:
        scores: 与 images 顺序一一对应的分数列表
    """
    import re
    n = len(images)
    if n == 0:
        return []
    if n == 1:
        return [max_score]
    
    # 构造多图消息
    user_parts = []
    for i, img in enumerate(images):
        b64 = _encode_image_b64(img)
        user_parts.append({"type": "text", "text": f"图片 {i+1}:"})
        user_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    
    criteria = f"Prompt: {prompt}"
    if text_content:
        criteria += f"\n期望的文字内容: {text_content}"
    user_parts.insert(0, {"type": "text", "text": criteria})
    
    system_prompt = f"""你是一个图像质量评估专家。请综合以下标准对这 {n} 张图片从最好到最差排序：
1. 图像整体质量（清晰度、色彩、构图）
2. 与 prompt 描述的符合程度
3. 如果有文字内容要求，文字的准确性和可读性

请只输出排名结果，格式为图片编号从最好到最差用逗号分隔，例如: 3,1,4,2
不要有任何其他内容。"""
    
    try:
        client = get_vlm_client()
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_parts}
            ],
            max_tokens=64,
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        
        # 解析排名: "3,1,4,2" → [3, 1, 4, 2]
        nums = [int(x) for x in re.findall(r'\d+', raw)]
        
        # 验证排名完整性
        if sorted(nums) != list(range(1, n + 1)):
            # VLM 输出不合法，回退到按原序均分
            nums = list(range(1, n + 1))
        
        # 按名次阶梯打分: 第 k 名得分 = max_score * (n - k) / (n - 1)
        # 第 1 名 = max_score, 最后一名 = max_score / n
        scores = [0.0] * n
        step = max_score / n
        for rank, img_idx in enumerate(nums):
            scores[img_idx - 1] = max_score - rank * step
        
        return scores
    except Exception as e:
        print(f"VLM 排名失败: {e}")
        # 回退：均匀打分
        step = max_score / n
        return [max_score - i * step for i in range(n)]


class MultiGPUTestTimeScaling:
    """
    多 GPU 并行 Test Time Scaling
    
    每个 GPU 加载独立的模型，处理不同的 beam 候选
    """
    
    def __init__(
        self,
        model_path: str,
        devices: list[str],
        prompt_refiner=None,
        dtype: torch.dtype = torch.bfloat16,
        logger: TTSLogger = None,
        vlm_score_mode: str = "rank",
        glyph_injector=None,
        injection_strength: Union[InjectionConfig, float] = None,
    ):
        self.model_path = model_path
        self.devices = devices
        self.num_gpus = len(devices)
        self.prompt_refiner = prompt_refiner
        self.dtype = dtype
        self.logger = logger or TTSLogger(run_name="multi_gpu_tts")
        self.vlm_score_mode = vlm_score_mode  # "rank" 或 "abs_score"
        self.glyph_injector = glyph_injector
        self.injection_config = InjectionConfig.from_value(injection_strength)
        
        self.logger.info(f"多 GPU TTS 初始化，使用 {self.num_gpus} 张 GPU: {devices}")
        
        # 每个 GPU 加载一个 pipeline
        self.pipelines = {}
        self._load_pipelines()
    
    def _load_pipelines(self):
        """在每个 GPU 上加载 pipeline"""
        from train.zimage_ip.pipeline_z_image import ZImagePipeline
        
        for device in self.devices:
            print(f"加载模型到 {device}...")
            pipe = ZImagePipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,
            )
            pipe.to(device)
            self.pipelines[device] = pipe
        print("所有模型加载完成")
    
    def _prepare_latents(self, device: str, height: int, width: int, seed: int) -> torch.Tensor:
        """准备初始 latent"""
        pipe = self.pipelines[device]
        num_channels = pipe.transformer.in_channels
        vae_scale = pipe.vae_scale_factor
        latent_h = 2 * (height // (vae_scale * 2))
        latent_w = 2 * (width // (vae_scale * 2))
        
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn((1, num_channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)
    
    def _encode_prompt(self, device: str, prompt: str) -> list:
        """编码 prompt"""
        pipe = self.pipelines[device]
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device=device, do_classifier_free_guidance=False)
        return prompt_embeds
    
    def _decode_latent(self, device: str, latent: torch.Tensor) -> Image.Image:
        """解码 latent 为图像"""
        pipe = self.pipelines[device]
        latent = latent.to(pipe.vae.dtype)
        latent = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.no_grad():
            image = pipe.vae.decode(latent, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    def _score_candidates(self, candidates: list, images: list[Image.Image],
                          prompt: str, text_content: Optional[str],
                          stage: str, extra_base: dict = None):
        """统一评分：根据 vlm_score_mode 选择绝对评分或排名阶梯评分"""
        if self.vlm_score_mode == "abs_score":
            for idx, cand in enumerate(candidates):
                cand.score = score_image_with_vlm(images[idx], cand.prompt, text_content)
                extra = {"device": cand.device, "text_content": text_content or ""}
                if extra_base:
                    extra.update(extra_base)
                self.logger.log_vlm_score(
                    stage=stage, candidate_idx=idx, prompt=cand.prompt,
                    score=cand.score, image=images[idx], extra=extra,
                )
        else:
            # rank 模式：VLM 一次看所有图排名，按名次阶梯给分
            scores = rank_images_with_vlm(images, prompt, text_content)
            for idx, cand in enumerate(candidates):
                cand.score = scores[idx]
                extra = {"device": cand.device, "text_content": text_content or ""}
                if extra_base:
                    extra.update(extra_base)
                self.logger.log_vlm_score(
                    stage=stage, candidate_idx=idx, prompt=cand.prompt,
                    score=cand.score, image=images[idx], extra=extra,
                )
    
    def _denoise_step(self, device: str, latent: torch.Tensor, prompt_embeds: list, t,
                      injection_data: dict = None, step_idx: int = 0,
                      injection_config: Union[InjectionConfig, float, None] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """单步去噪，返回 (next_latent, predicted_x0)
        
        flow matching: x_t = x_0 + σ·v  →  x_0 = x_t - σ·v
        支持可选的 GlyphInjector latent 注入
        """
        pipe = self.pipelines[device]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)
        timestep = t.expand(1).to(device)
        timestep_norm = (1000 - timestep) / 1000
        
        # Glyph Injection: 去噪前将文字 latent 混合进来
        if injection_data is not None and self.glyph_injector is not None:
            latent = self.glyph_injector.inject_latent(
                latent, injection_data, step_idx,
                injection_strength=injection_config or self.injection_config
            )
        
        latent_input = latent.to(pipe.transformer.dtype).unsqueeze(2)
        
        with torch.no_grad():
            model_out = pipe.transformer([latent_input[0]], timestep_norm, prompt_embeds, return_dict=False)[0]
        
        noise_pred = torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
        noise_pred = -noise_pred
        
        # 从 scheduler 的 sigma 表查找当前 sigma
        t_val = timestep[0].item()
        s_idx = (pipe.scheduler.timesteps - t_val).abs().argmin().item()
        sigma = pipe.scheduler.sigmas[s_idx].to(device)
        
        # predicted x0: x_0 = x_t - σ · velocity
        latent_0 = latent - sigma * noise_pred
        
        next_latent = pipe.scheduler.step(noise_pred.to(torch.float32), t, latent, return_dict=False)[0]
        return next_latent, latent_0
    
    def _process_candidate_on_gpu(
        self,
        cand_idx: int,
        device: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        timesteps: list,
        num_steps: int,
        text_regions: list = None,
    ) -> tuple:
        """在指定 GPU 上处理一个候选"""
        pipe = self.pipelines[device]
        
        # 准备
        latent = self._prepare_latents(device, height, width, seed)
        prompt_embeds = self._encode_prompt(device, prompt)
        
        # 为该候选准备独立的 glyph injection 数据（每个候选噪声不同）
        injection_data = None
        if text_regions and self.glyph_injector is not None:
            ts_tensor = pipe.scheduler.timesteps
            injection_data = self.glyph_injector.prepare_injection(
                text_regions=text_regions,
                image_size=(width, height),
                noise=latent,
                timesteps=ts_tensor,
            )
        
        # 去噪 num_steps 步，保留最后一步的 latent_0
        latent_0 = None
        for i, t in enumerate(timesteps[:num_steps]):
            latent, latent_0 = self._denoise_step(
                device, latent, prompt_embeds, t,
                injection_data=injection_data, step_idx=i,
                injection_config=self.injection_config,
            )
        
        return cand_idx, device, latent, prompt, prompt_embeds, latent_0, injection_data
    
    def generate_with_beam_search(
        self,
        prompt: str,
        text_content: Optional[str] = None,
        text_regions: list = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        beam_size: int = 8,
        early_stop_step: int = 3,
        keep_ratio: float = 0.25,
        renoise_ratio: float = 0.5,
        seed: int = 42,
        callback: Optional[Callable] = None
    ) -> tuple[Image.Image, float]:
        """
        多 GPU 并行 Beam Search 生成
        
        beam_size 个候选分配到 num_gpus 张 GPU 并行处理
        支持同时启用 GlyphInjector（传入 text_regions）
        """
        # 准备 prompt 变体
        if self.prompt_refiner and beam_size > 1:
            prompts = self.prompt_refiner(prompt, text_content, num_variants=min(beam_size, 4))
            while len(prompts) < beam_size:
                prompts.append(prompts[len(prompts) % len(prompts)])
        else:
            prompts = [prompt] * beam_size
        
        # 设置 scheduler（使用第一个 pipeline）
        first_pipe = self.pipelines[self.devices[0]]
        first_pipe.scheduler.set_timesteps(num_inference_steps, device=self.devices[0])
        timesteps = first_pipe.scheduler.timesteps.tolist()
        
        # 同步所有 GPU 的 scheduler
        for device in self.devices[1:]:
            self.pipelines[device].scheduler.set_timesteps(num_inference_steps, device=device)
        
        self.logger.info(f"=== 阶段 1: 并行去噪 {early_stop_step} 步 ===")
        
        # 并行处理前 early_stop_step 步
        candidates = []
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {}
            for i in range(beam_size):
                device = self.devices[i % self.num_gpus]
                future = executor.submit(
                    self._process_candidate_on_gpu,
                    i, device, prompts[i], seed + i, height, width, timesteps, early_stop_step,
                    text_regions=text_regions,
                )
                futures[future] = i
            
            for future in as_completed(futures):
                cand_idx, device, latent, cand_prompt, prompt_embeds, latent_0, inj_data = future.result()
                candidates.append(BeamCandidate(
                    latent=latent,
                    noise=latent.clone(),
                    prompt=cand_prompt,
                    prompt_embeds=prompt_embeds,
                    device=device,
                    step=early_stop_step,
                    latent_0=latent_0,
                    injection_data=inj_data,
                ))
                self.logger.info(f"  候选 {cand_idx} 完成 ({device})")
        
        self.logger.info(f"=== 阶段 2: VLM 早停评分 (step={early_stop_step}) ===")
        
        # 评分：解码模型预测的 latent_0 得到干净图像（无需额外去噪步数）
        early_images = [self._decode_latent(c.device, c.latent_0) for c in candidates]
        self._score_candidates(candidates, early_images, prompt, text_content, stage="early_stop",
                               extra_base={"step": early_stop_step})
        
        # 排序筛选
        candidates.sort(key=lambda x: x.score, reverse=True)
        n_keep = max(1, int(beam_size * keep_ratio))
        candidates = candidates[:n_keep]
        self.logger.info(f"保留 top {n_keep} 候选，最高分: {candidates[0].score:.2f}")
        
        self.logger.info(f"=== 阶段 3: 继续去噪剩余步骤 ===")
        
        # 继续去噪
        for step_idx in range(early_stop_step, len(timesteps)):
            t = torch.tensor(timesteps[step_idx])
            if callback:
                callback(step_idx, len(timesteps), len(candidates))
            
            for cand in candidates:
                cand.latent, cand.latent_0 = self._denoise_step(
                    cand.device, cand.latent, cand.prompt_embeds, t.to(cand.device),
                    injection_data=cand.injection_data, step_idx=step_idx,
                    injection_config=self.injection_config,
                )
            self.logger.info(f"  步骤 {step_idx + 1}/{len(timesteps)} 完成")
        
        # 最终评分
        self.logger.info(f"=== 阶段 4: 最终评分 ===")
        final_images = [self._decode_latent(c.device, c.latent) for c in candidates]
        self._score_candidates(candidates, final_images, prompt, text_content, stage="final")
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        best = candidates[0]
        final_image = self._decode_latent(best.device, best.latent)
        
        # 保存最终选中的图像
        self.logger.save_image(final_image, "best_result", caption=f"BEST  score={best.score:.2f}\n{best.prompt[:120]}")
        self.logger.info(f"最终结果: score={best.score:.2f}, prompt={best.prompt[:80]}...")
        
        return final_image, best.score


class TestTimeScaling:
    """单 GPU Test Time Scaling（兼容旧接口）"""
    
    def __init__(self, pipeline, prompt_refiner=None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 logger: TTSLogger = None, vlm_score_mode: str = "rank"):
        self.pipeline = pipeline
        self.prompt_refiner = prompt_refiner
        self.device = device
        self.dtype = dtype
        self.vae_scale_factor = pipeline.vae_scale_factor
        self.logger = logger or TTSLogger(run_name="single_gpu_tts")
        self.vlm_score_mode = vlm_score_mode # "rank" 或 "abs_score"
    
    def _prepare_latents(self, batch_size: int, height: int, width: int, generator=None) -> torch.Tensor:
        num_channels = self.pipeline.transformer.in_channels
        latent_h = 2 * (height // (self.vae_scale_factor * 2))
        latent_w = 2 * (width // (self.vae_scale_factor * 2))
        return torch.randn((batch_size, num_channels, latent_h, latent_w), generator=generator, device=self.device, dtype=torch.float32)
    
    def _encode_prompt(self, prompt: str) -> list:
        prompt_embeds, _ = self.pipeline.encode_prompt(prompt=prompt, device=self.device, do_classifier_free_guidance=False)
        return prompt_embeds
    
    def _decode_latent_to_image(self, latent: torch.Tensor) -> Image.Image:
        latent = latent.to(self.pipeline.vae.dtype)
        latent = (latent / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        with torch.no_grad():
            image = self.pipeline.vae.decode(latent, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]
    
    def _score_candidates(self, candidates: list, images: list[Image.Image],
                          prompt: str, text_content: Optional[str],
                          stage: str, extra_base: dict = None):
        """统一评分：根据 vlm_score_mode 选择绝对评分或排名阶梯评分"""
        if self.vlm_score_mode == "abs_score":
            for idx, cand in enumerate(candidates):
                cand.score = score_image_with_vlm(images[idx], cand.prompt, text_content)
                extra = {"text_content": text_content or ""}
                if extra_base:
                    extra.update(extra_base)
                self.logger.log_vlm_score(
                    stage=stage, candidate_idx=idx, prompt=cand.prompt,
                    score=cand.score, image=images[idx], extra=extra,
                )
        else:
            scores = rank_images_with_vlm(images, prompt, text_content)
            for idx, cand in enumerate(candidates):
                cand.score = scores[idx]
                extra = {"text_content": text_content or ""}
                if extra_base:
                    extra.update(extra_base)
                self.logger.log_vlm_score(
                    stage=stage, candidate_idx=idx, prompt=cand.prompt,
                    score=cand.score, image=images[idx], extra=extra,
                )
    
    def generate_with_beam_search(
        self,
        prompt: str,
        text_content: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        beam_size: int = 8,
        early_stop_step: int = 3,
        keep_ratio: float = 0.25,
        renoise_ratio: float = 0.5,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> tuple[Image.Image, float]:
        """单 GPU beam search"""
        if seed is not None:
            torch.manual_seed(seed)
        
        # 准备 prompts
        if self.prompt_refiner and beam_size > 1:
            prompts = self.prompt_refiner(prompt, text_content, num_variants=min(beam_size, 4))
            while len(prompts) < beam_size:
                prompts.append(prompts[len(prompts) % len(prompts)])
        else:
            prompts = [prompt] * beam_size
        
        # 初始化候选
        candidates = []
        for i in range(beam_size):
            generator = torch.Generator(device=self.device).manual_seed(seed + i if seed else i)
            noise = self._prepare_latents(1, height, width, generator)
            prompt_embeds = self._encode_prompt(prompts[i])
            candidates.append(BeamCandidate(latent=noise.clone(), noise=noise.clone(), prompt=prompts[i], prompt_embeds=prompt_embeds))
        
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps
        
        # 去噪
        for step_idx, t in enumerate(timesteps):
            if callback:
                callback(step_idx, len(timesteps), len(candidates))
            
            for cand in candidates:
                timestep = t.expand(1)
                timestep_norm = (1000 - timestep) / 1000
                latent_input = cand.latent.to(self.pipeline.transformer.dtype).unsqueeze(2)
                
                with torch.no_grad():
                    model_out = self.pipeline.transformer([latent_input[0]], timestep_norm, cand.prompt_embeds, return_dict=False)[0]
                
                noise_pred = torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
                noise_pred = -noise_pred
                
                # 计算 predicted x0: x_0 = x_t - σ · velocity
                t_val = timestep[0].item()
                s_idx = (self.pipeline.scheduler.timesteps - t_val).abs().argmin().item()
                sigma = self.pipeline.scheduler.sigmas[s_idx].to(self.device)
                cand.latent_0 = cand.latent - sigma * noise_pred
                
                cand.latent = self.pipeline.scheduler.step(noise_pred.to(torch.float32), t, cand.latent, return_dict=False)[0]
            
            # 早停评分
            if step_idx + 1 == early_stop_step and beam_size > 1:
                self.logger.info(f"=== 早停评分 (step={early_stop_step}) ===")
                early_imgs = [self._decode_latent_to_image(c.latent_0) for c in candidates]
                self._score_candidates(candidates, early_imgs, prompt, text_content,
                                       stage="early_stop", extra_base={"step": early_stop_step})
                
                candidates.sort(key=lambda x: x.score, reverse=True)
                n_keep = max(1, int(beam_size * keep_ratio))
                candidates = candidates[:n_keep]
                self.logger.info(f"保留 top {n_keep}，最高分: {candidates[0].score:.2f}")
        
        # 最终选择
        if len(candidates) > 1:
            self.logger.info("=== 最终评分 ===")
            final_imgs = [self._decode_latent_to_image(c.latent) for c in candidates]
            self._score_candidates(candidates, final_imgs, prompt, text_content, stage="final")
            candidates.sort(key=lambda x: x.score, reverse=True)
        
        best = candidates[0]
        final_image = self._decode_latent_to_image(best.latent)
        self.logger.save_image(final_image, "best_result", caption=f"BEST  score={best.score:.2f}\n{best.prompt[:120]}")
        self.logger.info(f"最终结果: score={best.score:.2f}")
        return final_image, best.score
    
    def __call__(self, prompt: str, text_content: Optional[str] = None, **kwargs) -> tuple[Image.Image, float]:
        return self.generate_with_beam_search(prompt, text_content, **kwargs)


def create_test_time_scaling(pipeline, prompt_refiner=None, device: str = "cuda",
                             logger: TTSLogger = None, vlm_score_mode: str = "rank") -> TestTimeScaling:
    """创建单 GPU TTS"""
    return TestTimeScaling(pipeline=pipeline, prompt_refiner=prompt_refiner, device=device,
                           dtype=pipeline.transformer.dtype, logger=logger, vlm_score_mode=vlm_score_mode)


def create_multi_gpu_tts(model_path: str, devices: list[str] = None, prompt_refiner=None,
                         logger: TTSLogger = None, vlm_score_mode: str = "rank",
                         glyph_injector=None, injection_strength: Union[InjectionConfig, float, None] = None) -> MultiGPUTestTimeScaling:
    """创建多 GPU TTS"""
    if devices is None:
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    return MultiGPUTestTimeScaling(model_path=model_path, devices=devices, prompt_refiner=prompt_refiner,
                                   logger=logger, vlm_score_mode=vlm_score_mode,
                                   glyph_injector=glyph_injector, injection_strength=injection_strength)
