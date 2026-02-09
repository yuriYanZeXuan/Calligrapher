"""
Z-Image 推理模块

包含核心接口:
- VLMAgent: 统一的 VLM 调用中心（排版分析、prompt 改写、评分）
- PromptRefiner: 基于 VLM API 的 prompt 优化
- GlyphInjector: 文字渲染和 latent 注入
- TestTimeScaling: Beam search 策略的测试时缩放
- AttentionEnhancement: 注意力增强
"""

from .VLM_agent import VLMAgent, PROMPT_TEMPLATES
from .prompt_refiner import PromptRefiner, refine_prompt
from .glyph_injector import GlyphInjector
from .test_time_scaling import TestTimeScaling, MultiGPUTestTimeScaling, create_multi_gpu_tts
from .attn_enhancement import AttentionEnhancement

__all__ = [
    "VLMAgent",
    "PROMPT_TEMPLATES",
    "PromptRefiner",
    "refine_prompt",
    "GlyphInjector",
    "TestTimeScaling",
    "MultiGPUTestTimeScaling",
    "create_multi_gpu_tts",
    "AttentionEnhancement",
]
