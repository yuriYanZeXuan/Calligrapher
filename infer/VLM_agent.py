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
        "You are an expert in image typography analysis. Given a reference image with a 10×10 grid and coordinate annotations, "
        "analyze the natural text rendering style and overall scene. Then plan the best typography layout for each text/formula item.\n\n"
        "CRITICAL: The reference image shows text that is FLAT and FACING the screen directly (frontal view, no perspective distortion). "
        "You must plan bboxes that are also flat and frontal - bboxes should have parallel top and bottom edges (approximately equal y_min and y_max across the width). "
        "NO angled, slanted, or perspective-distorted text regions.\n\n"
        "The 10×10 grid (11×11 lines with 0.1 step, covering 0.0-1.0) provides high-density positioning reference. "
        "Use the grid coordinates for precise bbox placement (normalized coordinates 0.0-1.0).\n\n"
        "For each text block, determine:\n"
        "- content: the text to render (one line per block)\n"
        "- bbox: [x_min, y_min, x_max, y_max] in 0-1 range. MUST be flat/horizontal with y_min ≈ constant across width (frontal view, no perspective tilting)\n"
        "- font_weight: light/regular/bold\n"
        "- font_size_ratio: 0.1-1.0 relative to bbox height\n"
        "- color: hex color matching the original text color in the reference image\n"
        "- is_latex: true/false\n"
        "- alignment: left/center/right\n"
        "- rotation: text rotation angle in degrees. 0 = horizontal (left to right). "
        "Positive = counter-clockwise (tilting upper-right ↗). Negative = clockwise (tilting lower-right ↘). "
        "Typical range: -30 to 30. Use 0 for most horizontal text.\n\n"
        "Rules:\n"
        "- bboxes must not overlap or exceed image bounds\n"
        "- bboxes must be FLAT and FACING the screen (y_min approximately equal for left and right sides, same for y_max)\n"
        "- color must match the original text color in the reference image\n"
        "- keep formulas intact\n"
        "- match the reference image's natural layout style\n\n"
        "Output strictly in this JSON format:\n"
        "```json\n"
        '{{\n'
        '  "image_analysis": {{\n'
        '    "background_style": "description",\n'
        '    "dominant_colors": ["#hex1", "#hex2"],\n'
        '    "text_style_hint": "description"\n'
        '  }},\n'
        '  "text_regions": [\n'
        '    {{\n'
        '      "content": "text",\n'
        '      "bbox": [x_min, y_min, x_max, y_max],\n'
        '      "font_weight": "regular",\n'
        '      "font_size_ratio": 0.7,\n'
        '      "color": "#FFFFFF",\n'
        '      "is_latex": false,\n'
        '      "alignment": "center",\n'
        '      "rotation": 0\n'
        '    }}\n'
        '  ]\n'
        '}}\n'
        "```"
    ),

    # ---- Generate clean prompt (remove text/formula descriptions) ----
    "generate_clean_prompt": (
        "Rewrite the user's prompt to explicitly exclude any text, formulas, math symbols, letters, or numbers from rendering."
        "You MUST:"
        "1. Remove any text related to formulas, math symbols, letters, or numbers."
        "2. Preserve the description related to backgrounds, styles, and compositions."
        "3. Output only the rewritten prompt, no explanations."
    ),

    # ---- Generate style prompt for FluxKlein (text style matching background) ----
    "generate_style_prompt": (
        "You are a helpful assistant that output key style instructions, keep background unedited and make foreground text harmonize with total picture."
        "Do NOT move, resize, or alter any text content or position. "
    ),

    # ---- Prompt refinement ----
    "refine_prompt": (
        "Optimize the user's simple description into a detailed image generation prompt (50-150 words). "
        "Preserve the original intent while adding visual details (lighting, colors, composition, style). "
        "Output only the optimized prompt, no explanations."
    ),

    "refine_prompt_with_text": (
        "Optimize the user's simple description into a detailed image generation prompt (50-150 words). "
        "The description includes text to be displayed in the image. Preserve intent, add visual details, "
        "and specify text position, appearance, and style (handwritten, printed, chalk, etc.). "
        "Output only the optimized prompt, no explanations."
    ),

    # ---- Image scoring ----
    "score_image": (
        "Rate this image 0-10 based on:\n"
        "- Overall quality (clarity, color, composition): 0-3\n"
        "- Alignment with prompt: 0-4\n"
        "- Text accuracy and readability (if applicable): 0-3\n\n"
        "Output only the numeric score, nothing else."
    ),

    # ---- Image ranking ----
    "rank_images": (
        "Rank these {n} images from best to worst based on: overall quality, prompt alignment, and text accuracy (if applicable).\n\n"
        "Output only the ranking as comma-separated indices (e.g., 3,1,4,2), nothing else."
    ),

    # ---- Best image selection ----
    "select_best_image": (
        "Choose the best image from these {n} candidates. "
        "Consider: text clarity, visual harmony with background, overall quality.\n\n"
        "Output ONLY the image number (1, 2, 3...), nothing else."
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
    font_path = Path(__file__).parent.parent / "assets" / "Arial-Unicode-Bold.ttf"
    
    if font_path.exists():
        return ImageFont.truetype(str(font_path), 16)
    return ImageFont.load_default()


def _add_grid_overlay(image: Image.Image, grid_size: int = 6) -> Image.Image:
    """在图像上添加10×10网格和坐标标注（11×11条线，步长0.1）。

    Args:
        image: 输入图像
        grid_size: 网格线数量（默认11条线: 0.0, 0.1, ..., 1.0，形成10×10区域）

    Returns:
        带网格和坐标标注的图像副本
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # 颜色配置
    grid_color = (255, 0, 0)  # 红色网格线
    text_color = (255, 0, 0)  # 红色文字
    edge_color = (180, 0, 0)  # 边缘线用稍暗的红色

    # 加载字体 - 使用更小字体避免拥挤
    font = _get_grid_font()
    
    # 网格步长：11条线形成10个间隔（0.0, 0.1, 0.2, ..., 1.0）
    step = 1.0 / (grid_size - 1)  # 0.1
    
    # 标注间隔：每2条线标注一次（0.0, 0.2, 0.4, 0.6, 0.8, 1.0）
    label_interval = 2

    # 绘制竖线和横线
    for i in range(grid_size):
        # 计算归一化坐标 (0.0 到 1.0，步长0.1)
        t = i * step
        x = int(t * width)
        y = int(t * height)
        
        # 边缘线 (0.0 和 1.0) 使用细线，中间使用标准线宽
        is_edge = (i == 0 or i == grid_size - 1)
        line_width = 1 if is_edge else 2
        color = edge_color if is_edge else grid_color
        
        # 绘制竖线（从顶部到底部）
        draw.line([(x, 0), (x, height)], fill=color, width=line_width)
        # 绘制横线
        draw.line([(0, y), (width, y)], fill=color, width=line_width)

    # 标注坐标（只在网格交点处，减少密度避免拥挤）
    for i in range(0, grid_size, label_interval):
        t = i * step
        y = int(t * height)
        
        for j in range(0, grid_size, label_interval):
            s = j * step
            x = int(s * width)
            coord_text = f"({s:.1f},{t:.1f})"

            # 计算文字位置 - 边缘文字向内偏移
            text_bbox = draw.textbbox((0, 0), coord_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # x方向偏移：左边缘向右，右边缘向左
            if j == 0:
                text_x = 2
            elif j == grid_size - 1:
                text_x = width - text_w - 2
            else:
                text_x = x - text_w // 2
            
            # y方向偏移：顶部向下，底部向上
            if i == 0:
                text_y = 2
            elif i == grid_size - 1:
                text_y = height - text_h - 2
            else:
                text_y = y - text_h // 2

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
        # 为图像添加10×10网格和坐标标注（11×11条线，步长0.1）
        image_with_grid = _add_grid_overlay(image)

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

    def generate_clean_prompt(self, prompt: str, typography_plan: dict = None) -> str:
        """将含文字描述的 prompt 改写为不渲染任何文本的 clean 版本。

        当提供 typography_plan 时，会将文字区域的位置、占比和背景颜色信息
        传递给 VLM，使 clean prompt 在对应位置描述正确的空白背景材质，
        从而让 Pass 2 的背景尽量贴合 Pass 1。
        """
        user_content = f"Original prompt:\n{prompt}"

        # if typography_plan:
        #     analysis = typography_plan.get("image_analysis", {})
        #     regions = typography_plan.get("text_regions", [])

        #     if analysis:
        #         user_content += (
        #             f"\n\nImage analysis from reference:"
        #             f"\n- Background style: {analysis.get('background_style', 'unknown')}"
        #             f"\n- Dominant colors: {', '.join(analysis.get('dominant_colors', []))}"
        #             f"\n- Text style hint: {analysis.get('text_style_hint', 'unknown')}"
        #         )

        #     if regions:
        #         user_content += "\n\nText regions to keep as BLANK areas (bbox in normalized 0-1 coords):"
        #         for i, r in enumerate(regions):
        #             bbox = r.get("bbox", [0, 0, 1, 1])
        #             user_content += f"\n  Region {i+1}: bbox={bbox}"

        raw = self.call_vlm(
            "generate_clean_prompt",
            user_content,
            max_tokens=512,
            temperature=0.3,
        )
        return raw.strip()

    # ---- FluxKlein 风格化提示词生成 ----

    def generate_style_prompt(self, image_analysis: dict) -> str:
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
            "生成一个简洁的图像编辑指令，要求将文字重绘为与背景协调但形成对比的风格。"
        )
        
        raw = self.call_vlm(
            "generate_style_prompt",
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

    # ---- 最佳图像选择 ----

    def select_best_image(
        self,
        images: list[Image.Image],
        prompt: str,
    ) -> int:
        """从候选图像中选择最佳的一张，返回 0-based 索引。"""
        n = len(images)
        if n <= 1:
            return 0

        parts: list[dict] = [{"type": "text", "text": f"Prompt: {prompt}"}]

        for i, img in enumerate(images):
            b64 = _encode_image_b64(img)
            parts.append({"type": "text", "text": f"Image {i+1}:"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        system_prompt = PROMPT_TEMPLATES["select_best_image"].format(n=n)

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": parts},
            ],
            max_tokens=16,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        nums = [int(x) for x in re.findall(r"\d+", raw)]
        if nums and 1 <= nums[0] <= n:
            return nums[0] - 1
        return n - 1  # 默认返回最后一个（通常是最精细的）

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
