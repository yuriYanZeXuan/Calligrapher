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
    """翻译文本到目标语言，保持公式格式，确保富文本输出"""
    lang_name = LANG_MAP.get(target_lang, target_lang)
    
    system_prompt = f"""你是一个专业的学术翻译专家。请将用户提供的学术文本翻译成{lang_name}。

【重要】你必须输出真正的富文本 Markdown，而非纯文本！

具体要求：
1. **标题**：使用 # ## ### 标记章节标题，不要省略
2. **强调**：关键术语用 **粗体**，定义或强调用 *斜体*
3. **列表**：枚举内容用 - 或 1. 2. 3. 格式
4. **引用**：重要引述用 > 块引用
5. **代码**：技术术语用 `行内代码`
6. **公式**：数学公式用 $...$ 或 $$...$$ 包裹（LaTeX格式）
7. **表格**：保持 Markdown 表格格式 |---|---|

翻译要准确流畅。只输出翻译结果，不要添加任何解释或额外说明。"""
    
    user_prompt = content
    return call_llm(system_prompt, user_prompt)


def rewrite_to_length(content: str, target_words: int, lang: str = "en") -> str:
    """改写文本到目标词数，确保富文本输出"""
    lang_name = LANG_MAP.get(lang, lang)
    
    system_prompt = f"""你是一个专业的学术写作专家。请将用户提供的学术文本改写为约{target_words}词的版本。

【重要】你必须输出真正的富文本 Markdown，而非纯文本！

具体要求：
1. **标题**：使用 # ## ### 标记章节结构
2. **强调**：关键概念用 **粗体**，强调用 *斜体*
3. **列表**：多个要点用 - 或数字列表
4. **引用**：重要内容用 > 块引用
5. **代码**：技术术语用 `行内代码`
6. **公式**：保持所有数学公式的 $...$ 或 $$...$$ 格式
7. 使用{lang_name}输出
8. 保持核心含义不变

只输出改写后的文本，不要添加任何解释。"""
    
    user_prompt = content
    return call_llm(system_prompt, user_prompt)


def generate_image_prompt(content: str, lang: str = "en") -> str:
    """生成图像生成的prompt描述（符合标准Markdown渲染效果）"""
    lang_name = LANG_MAP.get(lang, lang)
    
    system_prompt = f"""你是一个视觉描述专家。根据用户提供的文本，生成一个描述该文本渲染成图片后视觉效果的简短描述。

【重要】描述必须符合标准 Markdown/HTML 渲染的实际效果，不要描述过于复杂的设计。

可描述的视觉元素（都是 Markdown 可实现的）：
- 标题层级（大标题、小标题）和粗体/斜体文字
- 列表项（项目符号或编号）
- 块引用（左侧竖线样式）
- 代码块（灰色背景）
- 表格（带边框的网格）
- 数学公式（LaTeX渲染样式）
- 白色背景、深色文字、简洁排版

不要描述：
- 复杂的配色方案或渐变
- 装饰性图形或图标
- 复杂的布局（多栏、浮动元素等）

使用{lang_name}输出，只输出视觉描述，1-2句话即可。"""
    
    user_prompt = f"请描述以下文本渲染为图片后的视觉效果：\n\n{content[:500]}"
    return call_llm(system_prompt, user_prompt, max_tokens=256)


def generate_text_array(content: str, lang: str = "en") -> list[str]:
    """从内容中提取关键文本片段（用于 text 字段）
    
    【重要】提取的片段必须是原文的精确子串，不能改写或总结
    """
    lang_name = LANG_MAP.get(lang, lang)
    
    system_prompt = f"""你是一个文本提取专家。请从用户提供的文本中【精确复制】3-5个关键片段。

【关键要求】
- 你必须从原文中精确复制，一字不差
- 不要改写、总结或重新措辞
- 每个片段是原文中的完整句子或短语
- 片段应代表文本的核心内容

使用{lang_name}输出，每行一个片段，不要编号或其他标记。"""
    
    user_prompt = content
    result = call_llm(system_prompt, user_prompt, max_tokens=512)
    # 分割成列表
    texts = [line.strip() for line in result.strip().split('\n') if line.strip()]
    return texts[:5]  # 最多5个


def clean_extracted_content(content: str) -> str:
    """使用 LLM 清洗提取的原始内容，去除噪声保留正文
    
    清洗目标：
    - 去除引用标记、编号残留（如 [1], (2), Fig.1 等）
    - 去除不完整的句子片段
    - 去除无意义的符号和格式残留
    - 去除元信息（如作者、日期、页码等）
    - 保留完整、连贯、有意义的正文内容
    - 保留数学公式（$...$, $$...$$）
    - 保留表格结构
    """
    system_prompt = """你是一个学术文本清洗专家。请清洗用户提供的从论文中提取的原始文本。

【清洗目标】
去除以下噪声：
- 引用标记：[1], [2,3], (Smith 2020), ¹ 等
- 编号残留：Fig.1, Table 2, Eq.(3) 等孤立引用
- 不完整句子：开头或结尾被截断的片段
- 格式残留：多余的符号、乱码、特殊字符
- 元信息：页码、作者信息、期刊名等
- 无意义片段：单独的数字、字母、标点

【必须保留】
- 完整连贯的正文段落
- 数学公式（保持 $...$ 或 $$...$$ 格式）
- 表格结构（保持 Markdown 表格格式）
- Markdown 格式标记（#标题, **粗体**, *斜体*, -列表 等）

【输出要求】
- 只输出清洗后的正文内容
- 保持原文语言（不要翻译）
- 如果整段内容都是噪声，输出 [EMPTY]
- 不要添加任何解释"""

    user_prompt = f"请清洗以下文本：\n\n{content}"
    result = call_llm(system_prompt, user_prompt, max_tokens=2048, temperature=0.3)
    
    # 检查是否为空内容
    if result.strip() == "[EMPTY]" or not result.strip():
        return ""
    
    return result.strip()


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
