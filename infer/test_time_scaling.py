"""
Test Time Scaling: 基于 Beam Search 的测试时缩放接口

支持多 GPU 并行：每个 GPU 处理不同的 beam 候选
"""

import os
from typing import Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv(Path(__file__).parent.parent / '.env')

# VLM API 配置
API_KEY = os.getenv("QST_API_KEY")
BASE_URL = os.getenv("QST_BASE_URL")
VLM_MODEL = "qwen3-vl-235b-a22b-instruct"


@dataclass
class BeamCandidate:
    """Beam search 候选项"""
    latent: torch.Tensor
    noise: torch.Tensor
    prompt: str
    prompt_embeds: list
    score: float = 0.0
    step: int = 0
    device: str = "cuda:0"


def get_vlm_client() -> OpenAI:
    """获取 VLM 客户端"""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def score_image_with_vlm(
    image: Image.Image,
    prompt: str,
    text_content: Optional[str] = None
) -> float:
    """使用 VLM 对图像进行评分"""
    import base64
    from io import BytesIO
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
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
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model_path = model_path
        self.devices = devices
        self.num_gpus = len(devices)
        self.prompt_refiner = prompt_refiner
        self.dtype = dtype
        
        print(f"多 GPU TTS 初始化，使用 {self.num_gpus} 张 GPU: {devices}")
        
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
    
    def _denoise_step(self, device: str, latent: torch.Tensor, prompt_embeds: list, t: torch.Tensor) -> torch.Tensor:
        """单步去噪"""
        pipe = self.pipelines[device]
        timestep = t.expand(1).to(device)
        timestep_norm = (1000 - timestep) / 1000
        
        latent_input = latent.to(pipe.transformer.dtype).unsqueeze(2)
        
        with torch.no_grad():
            model_out = pipe.transformer([latent_input[0]], timestep_norm, prompt_embeds, return_dict=False)[0]
        
        noise_pred = torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
        noise_pred = -noise_pred
        
        return pipe.scheduler.step(noise_pred.to(torch.float32), t, latent, return_dict=False)[0]
    
    def _process_candidate_on_gpu(
        self,
        cand_idx: int,
        device: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        timesteps: list,
        num_steps: int
    ) -> tuple:
        """在指定 GPU 上处理一个候选"""
        pipe = self.pipelines[device]
        
        # 准备
        latent = self._prepare_latents(device, height, width, seed)
        prompt_embeds = self._encode_prompt(device, prompt)
        
        # 去噪 num_steps 步
        for i, t in enumerate(timesteps[:num_steps]):
            latent = self._denoise_step(device, latent, prompt_embeds, t)
        
        return cand_idx, device, latent, prompt, prompt_embeds
    
    def generate_with_beam_search(
        self,
        prompt: str,
        text_content: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
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
        
        print(f"\n=== 阶段 1: 并行去噪 {early_stop_step} 步 ===")
        
        # 并行处理前 early_stop_step 步
        candidates = []
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {}
            for i in range(beam_size):
                device = self.devices[i % self.num_gpus]
                future = executor.submit(
                    self._process_candidate_on_gpu,
                    i, device, prompts[i], seed + i, height, width, timesteps, early_stop_step
                )
                futures[future] = i
            
            for future in as_completed(futures):
                cand_idx, device, latent, cand_prompt, prompt_embeds = future.result()
                candidates.append(BeamCandidate(
                    latent=latent,
                    noise=latent.clone(),
                    prompt=cand_prompt,
                    prompt_embeds=prompt_embeds,
                    device=device,
                    step=early_stop_step
                ))
                print(f"  候选 {cand_idx} 完成 ({device})")
        
        print(f"\n=== 阶段 2: VLM 评分 ===")
        
        # 并行评分
        for cand in candidates:
            image = self._decode_latent(cand.device, cand.latent)
            cand.score = score_image_with_vlm(image, cand.prompt, text_content)
            print(f"  {cand.device}: 得分 {cand.score:.2f}")
        
        # 排序筛选
        candidates.sort(key=lambda x: x.score, reverse=True)
        n_keep = max(1, int(beam_size * keep_ratio))
        candidates = candidates[:n_keep]
        print(f"\n保留 top {n_keep} 候选，最高分: {candidates[0].score:.2f}")
        
        print(f"\n=== 阶段 3: 继续去噪剩余步骤 ===")
        
        # 继续去噪
        for step_idx in range(early_stop_step, len(timesteps)):
            t = torch.tensor(timesteps[step_idx])
            if callback:
                callback(step_idx, len(timesteps), len(candidates))
            
            for cand in candidates:
                cand.latent = self._denoise_step(cand.device, cand.latent, cand.prompt_embeds, t.to(cand.device))
            print(f"  步骤 {step_idx + 1}/{len(timesteps)} 完成")
        
        # 最终评分
        print(f"\n=== 最终评分 ===")
        for cand in candidates:
            image = self._decode_latent(cand.device, cand.latent)
            cand.score = score_image_with_vlm(image, cand.prompt, text_content)
            print(f"  {cand.device}: 最终得分 {cand.score:.2f}")
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        best = candidates[0]
        final_image = self._decode_latent(best.device, best.latent)
        
        return final_image, best.score


class TestTimeScaling:
    """单 GPU Test Time Scaling（兼容旧接口）"""
    
    def __init__(self, pipeline, prompt_refiner=None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.pipeline = pipeline
        self.prompt_refiner = prompt_refiner
        self.device = device
        self.dtype = dtype
        self.vae_scale_factor = pipeline.vae_scale_factor
    
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
    
    def generate_with_beam_search(
        self,
        prompt: str,
        text_content: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
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
                cand.latent = self.pipeline.scheduler.step(noise_pred.to(torch.float32), t, cand.latent, return_dict=False)[0]
            
            # 早停评分
            if step_idx + 1 == early_stop_step and beam_size > 1:
                for cand in candidates:
                    image = self._decode_latent_to_image(cand.latent)
                    cand.score = score_image_with_vlm(image, cand.prompt, text_content)
                
                candidates.sort(key=lambda x: x.score, reverse=True)
                n_keep = max(1, int(beam_size * keep_ratio))
                candidates = candidates[:n_keep]
        
        # 最终选择
        if len(candidates) > 1:
            for cand in candidates:
                image = self._decode_latent_to_image(cand.latent)
                cand.score = score_image_with_vlm(image, cand.prompt, text_content)
            candidates.sort(key=lambda x: x.score, reverse=True)
        
        best = candidates[0]
        return self._decode_latent_to_image(best.latent), best.score
    
    def __call__(self, prompt: str, text_content: Optional[str] = None, **kwargs) -> tuple[Image.Image, float]:
        return self.generate_with_beam_search(prompt, text_content, **kwargs)


def create_test_time_scaling(pipeline, prompt_refiner=None, device: str = "cuda") -> TestTimeScaling:
    """创建单 GPU TTS"""
    return TestTimeScaling(pipeline=pipeline, prompt_refiner=prompt_refiner, device=device, dtype=pipeline.transformer.dtype)


def create_multi_gpu_tts(model_path: str, devices: list[str] = None, prompt_refiner=None) -> MultiGPUTestTimeScaling:
    """创建多 GPU TTS"""
    if devices is None:
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    return MultiGPUTestTimeScaling(model_path=model_path, devices=devices, prompt_refiner=prompt_refiner)
