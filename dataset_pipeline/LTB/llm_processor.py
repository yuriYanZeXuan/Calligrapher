#!/usr/bin/env python3
"""
LLM API 调用模块：翻译、改写、prompt生成
参考 dev/qwen_directllm.py 的调用方式
"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).parent.parent.parent / '.env')

# API 配置（写死）
API_KEY = os.getenv("QST_API_KEY")
BASE_URL = os.getenv("QST_BASE_URL")
MODEL = "qwen3-vl-235b-a22b-instruct"

# 创建客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 语言映射
LANG_MAP = {
    "en": "English",
    "zh": "简体中文",
    "ko": "한국어",
    "ja": "日本語",
    "ar": "العربية",
    "fr": "Français",
}


def call_llm(system_prompt: str, user_prompt: str, 
             max_tokens: int = 4096, temperature: float = 0.7) -> str:
    """调用 LLM API"""
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


def translate(content: str, target_lang: str) -> str:
    """翻译文本到目标语言，保持公式格式"""
    lang_name = LANG_MAP.get(target_lang, target_lang)
    
    system_prompt = f"""你是一个专业的学术翻译专家。请将用户提供的学术文本翻译成{lang_name}。
要求：
1. 保持所有数学公式格式不变（如 \\( \\) 或 \\[ \\] 包裹的公式）
2. 保持表格的Markdown格式不变
3. 翻译要准确、学术化
4. 只输出翻译结果，不要添加任何解释"""
    
    user_prompt = content
    return call_llm(system_prompt, user_prompt)


def rewrite_to_length(content: str, target_words: int, lang: str = "en") -> str:
    """改写文本到目标词数"""
    lang_name = LANG_MAP.get(lang, lang)
    
    system_prompt = f"""你是一个专业的学术写作专家。请将用户提供的学术文本改写为约{target_words}词的版本。
要求：
1. 保持核心含义不变
2. 保持所有数学公式格式不变
3. 使用{lang_name}输出
4. 只输出改写后的文本，不要添加任何解释"""
    
    user_prompt = content
    return call_llm(system_prompt, user_prompt)


def generate_image_prompt(content: str, lang: str = "en") -> str:
    """生成图像生成的prompt描述"""
    lang_name = LANG_MAP.get(lang, lang)
    
    system_prompt = f"""你是一个专业的视觉设计专家。根据用户提供的学术文本，生成一个详细的图像生成prompt。
该prompt用于描述如何将这段文本渲染为富文本图片。

要求：
1. 使用{lang_name}输出
2. 描述文本布局、字体风格、公式位置
3. 描述整体视觉效果和色彩方案
4. prompt应该详细且可执行
5. 只输出prompt，不要添加任何解释

输出格式示例：
一张学术风格的富文本图片，白色背景，正文使用深灰色衬线字体居中排列。标题使用粗体，段落间距适中。公式使用标准LaTeX渲染样式，居中显示并与文本保持适当间距..."""
    
    user_prompt = f"请为以下学术文本生成图像渲染prompt：\n\n{content}"
    return call_llm(system_prompt, user_prompt)


def generate_text_array(content: str, lang: str = "en") -> list[str]:
    """从内容中提取关键文本片段（用于 text 字段）"""
    lang_name = LANG_MAP.get(lang, lang)
    
    system_prompt = f"""你是一个文本分析专家。请从用户提供的学术文本中提取3-5个关键文本片段。
要求：
1. 每个片段应该是完整的短语或句子
2. 片段应该代表文本的核心内容
3. 使用{lang_name}输出
4. 输出格式为每行一个片段，不要编号或其他标记"""
    
    user_prompt = content
    result = call_llm(system_prompt, user_prompt)
    # 分割成列表
    texts = [line.strip() for line in result.strip().split('\n') if line.strip()]
    return texts[:5]  # 最多5个


def process_content(content: str, target_lang: str, target_words: int = None) -> dict:
    """处理单条内容：翻译、改写、生成prompt
    
    Args:
        content: 原始内容（英文）
        target_lang: 目标语言
        target_words: 目标词数（可选，用于改写）
    
    Returns:
        dict: {
            "translated": str,  # 翻译后的内容
            "prompt": str,      # 图像生成prompt
            "text": list[str],  # 关键文本片段
        }
    """
    # 如果需要改写
    if target_words:
        content = rewrite_to_length(content, target_words, "en")
    
    # 翻译（如果不是英文）
    if target_lang == "en":
        translated = content
    else:
        translated = translate(content, target_lang)
    
    # 生成图像prompt
    prompt = generate_image_prompt(translated, target_lang)
    
    # 提取关键文本
    text_array = generate_text_array(translated, target_lang)
    
    return {
        "translated": translated,
        "prompt": prompt,
        "text": text_array,
    }


if __name__ == '__main__':
    # 测试
    test_content = """The Transformer architecture relies entirely on self-attention mechanisms. 
    The attention function can be described as mapping a query and a set of key-value pairs to an output.
    The output is computed as a weighted sum of the values, where the weight is computed by a compatibility 
    function of the query with the corresponding key."""
    
    print("测试翻译到中文:")
    result = translate(test_content, "zh")
    print(result[:200])
    
    print("\n测试生成图像prompt:")
    prompt = generate_image_prompt(test_content, "en")
    print(prompt[:200])
