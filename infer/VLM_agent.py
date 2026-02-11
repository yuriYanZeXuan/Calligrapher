"""
VLM Agent: 统一的 VLM 调用中心

所有 VLM/LLM 调用、客户端初始化和 prompt 模板集中在此文件。
提供 Agent 式的排版分析、prompt 改写、图像评分等功能。

设计原则:
- 不使用 try-except fallback，调用失败直接抛异常
- 所有 prompt 模板用全局字典 PROMPT_TEMPLATES 维护
- VLMAgent 类封装所有 VLM 交互逻辑
"""

import os
import re
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# 加载环境变量
load_dotenv(Path(__file__).parent.parent / ".env")


# ============ 全局 Prompt 模板字典 ============


PROMPT_TEMPLATES = {
    # ---- 核心：排版分析（Pass 1 参考图 → 排版规划 JSON）----
    "analyze_typography": (
        "你是一个专业的图像排版分析专家。给你一张带有5×5网格和坐标标注的参考图和一组待渲染的文本/公式内容。\n"
        "请分析参考图中文字的自然渲染风格和整体场景，然后为每个待渲染的文本/公式内容规划最佳排版方案。\n\n"
        "图像上的5×5网格帮助你精确定位：\n"
        "- 网格将图像分为4×4的16个区域\n"
        "- 网格线交点处标注了归一化坐标 (0.0,0.0) 到 (1.0,1.0)\n"
        "- 你可以参照这些坐标来确定文本区域的位置\n\n"
        "你需要自主决定：\n"
        "1. 将提供的文本内容拆分到合适数量的 block 中，保证每个block只包含一行文本/公式内容\n"
        "2. 每个 block 在图像中的精确位置 (bbox，归一化坐标 [x_min, y_min, x_max, y_max]，范围 0-1)\n"
        "   提示：可以参照网格坐标来确定，如(0.2,0.2)表示从左边20%、从上往20%的位置\n"
        "3. 每个 block 的字体粗细 (font_weight: light/regular/bold)\n"
        "4. 每个 block 的大小比例 (font_size_ratio: 0.1-1.0，相对于 block 高度)\n"
        "5. 每个 block 的文字颜色 (color: hex 格式如 #FFFFFF)\n"
        "6. 每个 block 的背景颜色 (background_color: hex 格式，用于渲染字形模版)\n"
        "7. 是否为 LaTeX 公式 (is_latex: true/false)\n"
        "8. 对齐方式 (alignment: left/center/right)\n\n"
        "规划原则：\n"
        "- bbox 不能重叠，不能超出图像边界 (0-1 范围)\n"
        "- 利用网格坐标精确定位，如 bbox [0.2, 0.3, 0.8, 0.5] 表示从(0.2,0.3)到(0.8,0.5)的区域\n"
        "- 文字颜色应与参考图的背景形成足够对比度\n"
        "- background_color 应选择与参考图中文字区域背景相近的颜色\n"
        "- 文字大小和位置应符合参考图中的自然布局风格\n"
        "- 公式内容保持完整，不要拆分单个公式\n\n"
        "请严格输出以下 JSON 格式，不要输出任何其他内容：\n"
        "```json\n"
        '{{\n'
        '  "image_analysis": {{\n'
        '    "background_style": "描述背景风格",\n'
        '    "dominant_colors": ["#hex1", "#hex2"],\n'
        '    "text_style_hint": "描述参考图中文字的视觉风格"\n'
        '  }},\n'
        '  "text_regions": [\n'
        '    {{\n'
        '      "content": "文本内容",\n'
        '      "bbox": [x_min, y_min, x_max, y_max],\n'
        '      "font_weight": "regular",\n'
        '      "font_size_ratio": 0.7,\n'
        '      "color": "#FFFFFF",\n'
        '      "background_color": "#000000",\n'
        '      "is_latex": false,\n'
        '      "alignment": "center"\n'
        '    }}\n'
        '  ]\n'
        '}}\n'
        "```"
    ),

    # ---- 生成 clean prompt（去除文字/公式描述）----
    "generate_clean_prompt": (
        "你是一个图像生成 prompt 改写专家。你的任务是将用户提供的 prompt 改写为一个"
        "明确不包含任何文字、公式、数学符号、字母、数字渲染的版本。\n\n"
        "改写原则：\n"
        "1. 保留原始场景、风格、构图、色彩等视觉描述\n"
        "2. 删除所有关于文字内容的描述（如 'with text \"Hello\"'、'写着XXX'等）\n"
        "3. 在适当位置添加强调：不渲染任何文字、字母、数字或公式\n"
        "4. 可以用 'blank area'、'empty space'、'clean surface' 等替代原有文字区域的描述\n"
        "5. 保持 prompt 长度适中\n\n"
        "只输出改写后的 prompt，不要添加任何解释。"
    ),

    # ---- 生成 FluxKlein 风格化提示词（根据背景选择文字风格）----
    "generate_klein_style_prompt": (
        "你是一个专业的图像编辑提示词专家。根据提供的背景场景描述，生成一个图像编辑指令，"
        "要求将文字区域重绘为与背景协调但形成对比的文字风格。\n\n"
        "核心原则：\n"
        "1. 文字颜色必须与背景形成高对比度（深色背景用浅色字，浅色背景用深色字）\n"
        "2. 文字风格要与整体场景协调（例如：黑板用粉笔字、白板用马克笔、纸张用墨水/印刷体）\n"
        "3. 文字要有自然的纹理和质感（不是完美的数字字体）\n"
        "4. 只修改文字区域，背景保持原样\n\n"
        "常见场景对应风格：\n"
        "- 黑板/深色背景 → 白色粉笔字，有粉笔纹理和轻微飞白\n"
        "- 白板/浅色背景 → 黑色马克笔，有手写痕迹\n"
        "- 纸张/羊皮纸 → 深色墨水/印刷体，可能有晕染效果\n"
        "- 金属/混凝土 → 喷漆/刻字效果\n\n"
        "只输出生成的编辑指令（英文），不要添加任何解释。格式示例：\n"
        "'Rewrite the text in white chalk style with natural texture on the blackboard. No other changes.'"
    ),

    # ---- Prompt 优化（从 prompt_refiner.py 迁移）----
    "refine_prompt": (
        "你是一个专业的图像生成 prompt 优化专家。你的任务是将用户提供的简单描述"
        "优化为详细、具体的图像生成 prompt。\n\n"
        "优化原则：\n"
        "1. 保持原始意图不变\n"
        "2. 添加视觉细节（光线、色彩、构图、风格等）\n"
        "3. 如果涉及文字内容，明确描述文字的位置、大小、字体风格\n"
        "4. 使用清晰、具体的描述词汇\n"
        "5. 保持 prompt 长度适中（50-150 词）\n\n"
        "只输出优化后的 prompt，不要添加任何解释。"
    ),

    "refine_prompt_with_text": (
        "你是一个专业的图像生成 prompt 优化专家。你的任务是将用户提供的简单描述"
        "优化为详细、具体的图像生成 prompt。\n\n"
        "用户的描述中包含需要在图像中显示的文字内容。\n\n"
        "优化原则：\n"
        "1. 保持原始意图不变\n"
        "2. 添加视觉细节（光线、色彩、构图、风格等）\n"
        "3. 明确描述文字应该出现的位置和视觉效果\n"
        "4. 描述文字的风格（手写、印刷、粉笔字等）\n"
        "5. 使用清晰、具体的描述词汇\n"
        "6. 保持 prompt 长度适中（50-150 词）\n\n"
        "只输出优化后的 prompt，不要添加任何解释。"
    ),

    # ---- 图像评分（从 test_time_scaling.py 迁移）----
    "score_image": (
        "你是一个图像质量评估专家。请根据以下标准对图像进行评分：\n"
        "1. 图像整体质量（清晰度、色彩、构图）：0-3分\n"
        "2. 与 prompt 描述的符合程度：0-4分\n"
        "3. 如果有文字内容要求，文字的准确性和可读性：0-3分\n\n"
        "请只输出一个 0-10 之间的数字分数，不要有任何其他内容。"
    ),

    # ---- 图像排名（从 test_time_scaling.py 迁移）----
    "rank_images": (
        "你是一个图像质量评估专家。请综合以下标准对这 {n} 张图片从最好到最差排序：\n"
        "1. 图像整体质量（清晰度、色彩、构图）\n"
        "2. 与 prompt 描述的符合程度\n"
        "3. 如果有文字内容要求，文字的准确性和可读性\n\n"
        "请只输出排名结果，格式为图片编号从最好到最差用逗号分隔，例如: 3,1,4,2\n"
        "不要有任何其他内容。"
    ),
}


# ============ 工具函数 ============


def _encode_image_b64(image: Image.Image) -> str:
    """PIL Image -> base64 字符串"""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _get_grid_font() -> ImageFont.FreeTypeFont:
    """获取用于网格坐标标注的字体。"""
    font_path = Path(__file__).parent.parent / "assets" / "ChalkboardSE.ttc"
    
    if font_path.exists():
        return ImageFont.truetype(str(font_path), 16)
    return ImageFont.load_default()


def _add_grid_overlay(image: Image.Image, grid_size: int = 5) -> Image.Image:
    """在图像上添加5×5网格和坐标标注。

    Args:
        image: 输入图像
        grid_size: 网格数量（默认5，即4×4区域）

    Returns:
        带网格和坐标标注的图像副本
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # 颜色配置
    grid_color = (255, 0, 0)  # 红色网格线
    text_color = (255, 0, 0)  # 红色文字

    # 加载字体
    font = _get_grid_font()

    # 绘制竖线和横线，并在交点处标注坐标
    for i in range(grid_size):
        # 计算归一化坐标 (0.0 到 1.0)
        t = i / (grid_size - 1)
        x = int(t * width)
        y = int(t * height)

        # 绘制竖线
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        # 绘制横线
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

        # 在交点处标注坐标
        for j in range(grid_size):
            s = j / (grid_size - 1)
            coord_x = int(s * width)
            coord_y = int(t * height)
            coord_text = f"({s:.1f},{t:.1f})"

            # 计算文字位置（稍微偏移避免遮挡网格点）
            text_x = min(coord_x + 3, width - 50)
            text_y = max(coord_y - 15, 0)

            draw.text((text_x, text_y), coord_text, fill=text_color, font=font)

    return img


def _extract_json_from_response(text: str) -> dict:
    """从 VLM 响应中提取 JSON 对象。

    支持 ```json ... ``` 包裹和裸 JSON 两种形式。
    """
    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    # 尝试直接解析整个文本
    # 找到第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"无法从 VLM 响应中提取 JSON:\n{text[:500]}")


# ============ VLMAgent 类 ============


class VLMAgent:
    """统一的 VLM 调用接口

    集中管理所有 VLM/LLM 交互：排版分析、prompt 改写、图像评分等。
    不使用 try-except fallback，调用失败直接向上层抛出异常。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen3-vl-235b-a22b-instruct",
    ):
        self._api_key = api_key or os.getenv("QST_API_KEY")
        self._base_url = base_url or os.getenv("QST_BASE_URL")
        self._model = model
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """延迟创建 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    # ---- 核心调用 ----

    def call_vlm(
        self,
        template_key: str,
        user_content: str,
        images: Optional[list[Image.Image]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **format_kwargs,
    ) -> str:
        """通用 VLM 调用。

        从 PROMPT_TEMPLATES[template_key] 取 system prompt，
        用 format_kwargs 格式化后发送请求。

        Args:
            template_key: PROMPT_TEMPLATES 中的 key
            user_content: 用户消息文本
            images: 可选的 PIL Image 列表（以 base64 嵌入）
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            **format_kwargs: 用于格式化 system prompt 的参数

        Returns:
            VLM 响应文本
        """
        system_prompt = PROMPT_TEMPLATES[template_key]
        if format_kwargs:
            system_prompt = system_prompt.format(**format_kwargs)

        # 构建 user message content
        user_parts: list[dict] = [{"type": "text", "text": user_content}]
        if images:
            for img in images:
                b64 = _encode_image_b64(img)
                user_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_parts},
            ],
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    # ---- 排版分析（核心）----

    def analyze_typography(
        self,
        image: Image.Image,
        prompt: str,
        text_contents: list[str],
    ) -> dict:
        """分析 Pass 1 参考图，自主规划文本排版。

        VLM 根据参考图（带网格坐标）的视觉布局，为每个待渲染的文本/公式内容
        自主决定 block 数量、位置 (bbox)、字体粗细、大小、颜色等。

        Args:
            image: Pass 1 生成的参考图
            prompt: 用户的原始 prompt
            text_contents: 待渲染的文本/公式内容列表

        Returns:
            typography_plan 字典，包含 image_analysis 和 text_regions
        """
        # 为图像添加5×5网格和坐标标注
        image_with_grid = _add_grid_overlay(image, grid_size=5)

        contents_desc = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(text_contents))
        user_content = (
            f"原始 prompt: {prompt}\n\n"
            f"待渲染的文本/公式内容列表:\n{contents_desc}\n\n"
            f"图像上的红色网格线和坐标标注可以帮助你精确定位文本区域。"
        )

        raw = self.call_vlm(
            "analyze_typography",
            user_content,
            images=[image_with_grid],  # 传入带网格的图像
            max_tokens=2048,
            temperature=0.3,
        )
        return _extract_json_from_response(raw)

    # ---- Clean prompt 生成 ----

    def generate_clean_prompt(self, prompt: str) -> str:
        """将含文字描述的 prompt 改写为不渲染任何文本的 clean 版本。"""
        user_content = f"原始 prompt:\n{prompt}"
        raw = self.call_vlm(
            "generate_clean_prompt",
            user_content,
            max_tokens=512,
            temperature=0.3,
        )
        return raw.strip()

    # ---- FluxKlein 风格化提示词生成 ----

    def generate_klein_style_prompt(self, image_analysis: dict) -> str:
        """根据背景场景生成 FluxKlein 风格化提示词。
        
        Args:
            image_analysis: VLM 排版分析返回的 image_analysis 字段
            
        Returns:
            FluxKlein 编辑提示词（英文）
        """
        bg_style = image_analysis.get("background_style", "")
        dominant_colors = image_analysis.get("dominant_colors", [])
        text_hint = image_analysis.get("text_style_hint", "")
        
        user_content = (
            f"背景风格: {bg_style}\n"
            f"主导颜色: {', '.join(dominant_colors)}\n"
            f"文字风格提示: {text_hint}\n\n"
            "请根据以上场景信息，生成一个图像编辑指令，要求将文字重绘为与背景协调但形成对比的风格。"
        )
        
        raw = self.call_vlm(
            "generate_klein_style_prompt",
            user_content,
            max_tokens=256,
            temperature=0.4,
        )
        return raw.strip().strip('"\'')  # 去除可能的引号

    # ---- Prompt 优化 ----

    def refine_prompt(
        self,
        prompt: str,
        text_content: Optional[str] = None,
        num_variants: int = 1,
        temperature: float = 0.7,
    ) -> list[str]:
        """优化用户 prompt（原 PromptRefiner 功能）。"""
        if text_content:
            template_key = "refine_prompt_with_text"
            user_content = f"原始描述：{prompt}\n\n需要显示的文字内容：{text_content}"
        else:
            template_key = "refine_prompt"
            user_content = f"原始描述：{prompt}"

        results = []
        for _ in range(num_variants):
            raw = self.call_vlm(
                template_key,
                user_content,
                max_tokens=512,
                temperature=temperature,
            )
            results.append(raw.strip())
        return results

    # ---- 图像评分 ----

    def score_image(
        self,
        image: Image.Image,
        prompt: str,
        text_content: Optional[str] = None,
    ) -> float:
        """使用 VLM 对单张图像进行绝对评分（0-10）。"""
        user_content = f"Prompt: {prompt}"
        if text_content:
            user_content += f"\n期望的文字内容: {text_content}"

        raw = self.call_vlm(
            "score_image",
            user_content,
            images=[image],
            max_tokens=16,
            temperature=0.1,
        )
        return min(max(float(raw.strip()), 0.0), 10.0)

    # ---- 图像排名 ----

    def rank_images(
        self,
        images: list[Image.Image],
        prompt: str,
        text_content: Optional[str] = None,
        max_score: float = 10.0,
    ) -> list[float]:
        """让 VLM 对多张图片排名，按名次阶梯给分。

        第 1 名 -> max_score，之后均匀递减。
        """
        n = len(images)
        if n == 0:
            return []
        if n == 1:
            return [max_score]

        # 构造多图消息
        parts = []
        criteria = f"Prompt: {prompt}"
        if text_content:
            criteria += f"\n期望的文字内容: {text_content}"
        parts.append({"type": "text", "text": criteria})

        for i, img in enumerate(images):
            b64 = _encode_image_b64(img)
            parts.append({"type": "text", "text": f"图片 {i+1}:"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        system_prompt = PROMPT_TEMPLATES["rank_images"].format(n=n)

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": parts},
            ],
            max_tokens=64,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # 解析排名
        nums = [int(x) for x in re.findall(r"\d+", raw)]
        if sorted(nums) != list(range(1, n + 1)):
            nums = list(range(1, n + 1))

        step = max_score / n
        scores = [0.0] * n
        for rank, img_idx in enumerate(nums):
            scores[img_idx - 1] = max_score - rank * step

        return scores
