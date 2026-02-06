"""
Prompt Refiner: 基于 VLM API 的 prompt 优化接口

使用 LLM 对用户输入的 prompt 进行优化，使其更适合图像生成模型。
"""

import os
from pathlib import Path
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).parent.parent / '.env')

# API 配置
API_KEY = os.getenv("QST_API_KEY")
BASE_URL = os.getenv("QST_BASE_URL")
MODEL = "qwen3-vl-235b-a22b-instruct"

# 创建客户端
_client = None

def _get_client() -> OpenAI:
    """获取或创建 OpenAI 客户端"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def call_llm(
    system_prompt: str, 
    user_prompt: str, 
    max_tokens: int = 4096, 
    temperature: float = 0.7
) -> str:
    """调用 LLM API"""
    client = _get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content


REFINE_SYSTEM_PROMPT = """你是一个专业的图像生成 prompt 优化专家。你的任务是将用户提供的简单描述优化为详细、具体的图像生成 prompt。

优化原则：
1. 保持原始意图不变
2. 添加视觉细节（光线、色彩、构图、风格等）
3. 如果涉及文字内容，明确描述文字的位置、大小、字体风格
4. 使用清晰、具体的描述词汇
5. 保持 prompt 长度适中（50-150 词）

只输出优化后的 prompt，不要添加任何解释。"""


REFINE_WITH_TEXT_SYSTEM_PROMPT = """你是一个专业的图像生成 prompt 优化专家。你的任务是将用户提供的简单描述优化为详细、具体的图像生成 prompt。

用户的描述中包含需要在图像中显示的文字内容。

优化原则：
1. 保持原始意图不变
2. 添加视觉细节（光线、色彩、构图、风格等）
3. 明确描述文字应该出现的位置和视觉效果
4. 描述文字的风格（手写、印刷、粉笔字等）
5. 使用清晰、具体的描述词汇
6. 保持 prompt 长度适中（50-150 词）

只输出优化后的 prompt，不要添加任何解释。"""


def refine_prompt(
    prompt: str,
    text_content: Optional[str] = None,
    temperature: float = 0.7,
    num_variants: int = 1
) -> list[str]:
    """
    优化用户 prompt
    
    Args:
        prompt: 原始 prompt
        text_content: 图像中需要显示的文字内容（可选）
        temperature: 生成温度，越高越多样
        num_variants: 生成变体数量
        
    Returns:
        优化后的 prompt 列表
    """
    if text_content:
        system_prompt = REFINE_WITH_TEXT_SYSTEM_PROMPT
        user_prompt = f"原始描述：{prompt}\n\n需要显示的文字内容：{text_content}"
    else:
        system_prompt = REFINE_SYSTEM_PROMPT
        user_prompt = f"原始描述：{prompt}"
    
    results = []
    for _ in range(num_variants):
        refined = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=512,
            temperature=temperature
        )
        results.append(refined.strip())
    
    return results


class PromptRefiner:
    """Prompt 优化器类"""
    
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        
    def refine(
        self, 
        prompt: str, 
        text_content: Optional[str] = None,
        num_variants: int = 1
    ) -> list[str]:
        """
        优化 prompt
        
        Args:
            prompt: 原始 prompt
            text_content: 需要显示的文字内容
            num_variants: 变体数量
            
        Returns:
            优化后的 prompt 列表
        """
        return refine_prompt(
            prompt=prompt,
            text_content=text_content,
            temperature=self.temperature,
            num_variants=num_variants
        )
    
    def __call__(
        self, 
        prompt: str, 
        text_content: Optional[str] = None,
        num_variants: int = 1
    ) -> list[str]:
        return self.refine(prompt, text_content, num_variants)


if __name__ == "__main__":
    # 测试
    test_prompt = "爱因斯坦在黑板前写公式"
    test_text = "x = (-b ± √(b²-4ac)) / 2a"
    
    print("原始 prompt:", test_prompt)
    print("文字内容:", test_text)
    print("\n优化结果:")
    
    results = refine_prompt(test_prompt, test_text, num_variants=2)
    for i, r in enumerate(results, 1):
        print(f"\n变体 {i}:")
        print(r)
