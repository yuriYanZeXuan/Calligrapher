"""
公式渲染辅助模块

渲染优先级：
1. MathJax (Node.js) — 完整 LaTeX 支持，包括 array/matrix/cases 等环境，绝对不能渲染中文！
2. matplotlib mathtext — 无需 Node.js，支持常用 LaTeX 子集
3. PIL 纯文本 — 最后兜底

支持：
- LaTeX 数学公式渲染
- Unicode 数学符号自动转换为 LaTeX
- 纯文本字体渲染（PIL）
- 自动检测并选择渲染路径
"""

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# MathJax 渲染脚本路径
_MATHJAX_SCRIPT = Path(__file__).parent / "render_mathjax.js"
# 缓存 node 可用性检测结果
from typing import Optional as _Opt
_node_available: _Opt[bool] = None


# ============ LaTeX 检测 ============


def is_latex(text: str) -> bool:
    """启发式检测文本是否包含 LaTeX 公式。

    匹配条件（任一即触发）：
    - 包含 LaTeX 命令：\\frac, \\sqrt, \\int, \\sum 等
    - 被 $ 包裹
    - 包含上下标花括号组合：^{...} 或 _{...}
    - 包含 \\begin / \\end 环境
    """
    latex_patterns = [
        r'\\(?:frac|sqrt|int|oint|sum|prod|lim|infty|partial|nabla|left|right|'
        r'begin|end|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|'
        r'pi|phi|psi|chi|rho|tau|eta|zeta|xi|kappa|nu|'
        r'mathbb|mathcal|mathbf|mathrm|text|hat|bar|vec|dot|tilde|'
        r'cdot|times|div|pm|mp|leq|geq|neq|approx|equiv|sim|'
        r'rightarrow|leftarrow|Rightarrow|Leftarrow|mapsto|to)',
        r'\$.*\$',                     # $...$
        r'[_\^]\{[^}]+\}',            # ^{...} or _{...}
        r'\\begin\{',                  # \begin{...}
    ]
    return any(re.search(p, text) for p in latex_patterns)


# ============ Unicode → LaTeX 转换 ============


_UNICODE_TO_LATEX = [
    ("∫∫∫", r"\iiint"),
    ("∫∫", r"\iint"),
    ("∫", r"\int"),
    ("∑", r"\sum"),
    ("∏", r"\prod"),
    ("∞", r"\infty"),
    ("±", r"\pm"),
    ("≈", r"\approx"),
    ("≠", r"\neq"),
    ("≤", r"\leq"),
    ("≥", r"\geq"),
    ("→", r"\rightarrow"),
    ("←", r"\leftarrow"),
    ("⟨", r"\langle"),
    ("⟩", r"\rangle"),
    # Unicode 上下标数字
    ("₀", "_0"), ("₁", "_1"), ("₂", "_2"), ("₃", "_3"), ("₄", "_4"),
    ("₅", "_5"), ("₆", "_6"), ("₇", "_7"), ("₈", "_8"), ("₉", "_9"),
    ("⁰", "^0"), ("¹", "^1"), ("²", "^2"), ("³", "^3"), ("⁴", "^4"),
    ("⁵", "^5"), ("⁶", "^6"), ("⁷", "^7"), ("⁸", "^8"), ("⁹", "^9"),
    # Greek letters
    ("α", r"\alpha"),
    ("β", r"\beta"),
    ("γ", r"\gamma"),
    ("δ", r"\delta"),
    ("ε", r"\epsilon"),
    ("ζ", r"\zeta"),
    ("η", r"\eta"),
    ("θ", r"\theta"),
    ("λ", r"\lambda"),
    ("μ", r"\mu"),
    ("ν", r"\nu"),
    ("ξ", r"\xi"),
    ("π", r"\pi"),
    ("ρ", r"\rho"),
    ("σ", r"\sigma"),
    ("τ", r"\tau"),
    ("φ", r"\varphi"),
    ("ψ", r"\psi"),
    ("ω", r"\omega"),
    ("Δ", r"\Delta"),
    ("Σ", r"\Sigma"),
    ("Ω", r"\Omega"),
    # Special
    ("ℏ", r"\hbar"),
]


def _check_node() -> bool:
    """检测 Node.js 是否可用（结果缓存）。"""
    global _node_available
    if _node_available is None:
        _node_available = shutil.which("node") is not None
    return _node_available


def _fix_sqrt_parens(text: str) -> str:
    """将 \\sqrt(...) 修正为 \\sqrt{...}（matplotlib 要求花括号）。"""
    # 匹配 \sqrt( 并找到对应的闭括号，替换为花括号
    result = []
    i = 0
    pat = r"\sqrt"
    while i < len(text):
        if text[i:i+5] == pat + "(" :
            result.append(pat + "{")
            depth = 1
            j = i + 6
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                    result.append("(")
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        result.append("}")
                    else:
                        result.append(")")
                else:
                    result.append(text[j])
                j += 1
            i = j
        else:
            # 也处理单独的 √( 未被转换的情况
            if text[i] == "√" and i + 1 < len(text) and text[i+1] == "(":
                result.append(r"\sqrt{")
                depth = 1
                j = i + 2
                while j < len(text) and depth > 0:
                    if text[j] == "(":
                        depth += 1
                        result.append("(")
                    elif text[j] == ")":
                        depth -= 1
                        if depth == 0:
                            result.append("}")
                        else:
                            result.append(")")
                    else:
                        result.append(text[j])
                    j += 1
                i = j
            else:
                result.append(text[i])
                i += 1
    return "".join(result)


def plaintext_to_latex(text: str) -> str:
    """将含 Unicode 数学符号的 plaintext 转换为 LaTeX 命令。"""
    result = text
    # 先处理 √(...) → \sqrt{...}（带括号的情况）
    result = _fix_sqrt_parens(result)
    # 再做逐符号替换（剩余的 bare √ → \sqrt 等）
    for old, new in _UNICODE_TO_LATEX:
        result = result.replace(old, new)
    # 替换后可能产生新的 \sqrt(...)，再修一次
    result = _fix_sqrt_parens(result)
    return result


# ============ LaTeX 渲染 ============


def render_mathjax(
    latex: str,
    width: int,
    height: int,
    text_color: str = "black",
    background_color: str = "white",
) -> _Opt[Image.Image]:
    """使用 MathJax (Node.js) 渲染 LaTeX → SVG → PIL Image。

    支持完整的 LaTeX 语法（array, matrix, cases 等环境）。
    需要 Node.js 和 cairosvg（或 Pillow SVG 支持）。

    Returns:
        PIL Image，如果 Node.js 不可用或渲染失败则返回 None。
    """
    if not _check_node() or not _MATHJAX_SCRIPT.exists():
        return None

    # 去掉 $ 包裹（MathJax 自己处理）
    formula = latex.strip().strip("$")

    # 调用 Node.js 渲染 SVG
    try:
        result = subprocess.run(
            ["node", str(_MATHJAX_SCRIPT), formula],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"[MathJax] node 错误: {result.stderr.strip()}")
            return None
        svg_str = result.stdout
        if not svg_str or "<svg" not in svg_str:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[MathJax] 调用失败: {e}")
        return None

    # SVG → PNG：使用 cairosvg 转换
    img = _convert_svg_to_png(svg_str, width, height, text_color, background_color)
    
    if img is None:
        print("[MathJax] SVG→PNG 转换失败")
        return None

    return img


def _convert_svg_to_png(svg_str: str, width: int, height: int, 
                        text_color: str = "black", background_color: str = "white") -> _Opt[Image.Image]:
    """将 MathJax 生成的 SVG 转换为 PNG。
    
    MathJax 输出包裹在 <mjx-container> 中，需要提取内部 SVG 并添加样式。
    """
    try:
        from xml.etree import ElementTree as ET
        import re
        
        # 解析 XML
        root = ET.fromstring(svg_str)
        
        # 查找内部 SVG 元素（MathJax 包裹在 mjx-container 中）
        svg_element = None
        if 'mjx-container' in root.tag:
            for child in root.iter():
                if child.tag.endswith('svg') or child.tag == 'svg':
                    svg_element = child
                    break
        
        if svg_element is None:
            print("[MathJax] 无法在 mjx-container 中找到 SVG 元素")
            return None
        
        # 获取原始尺寸（用于计算宽高比）
        orig_width = svg_element.get('width', '')
        orig_height = svg_element.get('height', '')
        
        # 解析尺寸（支持 ex 单位，1ex ≈ 8px）
        def parse_size(s):
            if not s:
                return None
            s = s.strip()
            if s.endswith('ex'):
                return float(s[:-2]) * 8
            try:
                return float(s)
            except:
                return None
        
        orig_w = parse_size(orig_width)
        orig_h = parse_size(orig_height)
        
        # 计算保持宽高比的新尺寸
        if orig_w and orig_h:
            scale_w = width / orig_w
            scale_h = height / orig_h
            scale = min(scale_w, scale_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
        else:
            new_w, new_h = width, height
        
        # 修改 SVG 属性
        svg_element.set('width', str(new_w))
        svg_element.set('height', str(new_h))
        
        # 构建 style：只设置文字颜色，背景透明
        existing_style = svg_element.get('style', '')
        # 移除 MathJax 的 vertical-align 和 background-color
        style_parts = []
        if existing_style:
            for part in existing_style.split(';'):
                part = part.strip()
                # 过滤掉 vertical-align 和 background-color
                if part and not part.startswith(('vertical-align', 'background-color')):
                    style_parts.append(part)
        
        # MathJax 使用 currentColor，设置 color 即可改变文字颜色
        # 默认黑色文字，透明背景
        svg_element.set('style', f"color: {text_color or 'black'}")
        
        # 序列化修改后的 SVG
        svg_bytes = ET.tostring(svg_element, encoding='utf-8')
        
        # 使用 cairosvg 转换为 PNG（保留透明背景）
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=new_w,
            output_height=new_h,
        )
        
        # 使用 RGBA 保留透明通道
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        return img
        
    except Exception as e:
        print(f"[MathJax] SVG 转换失败: {e}")
        return None


def render_latex(
    latex: str,
    width: int,
    height: int,
    text_color: str = "black",
    background_color: str = "white",
) -> Image.Image:
    """使用 matplotlib 渲染 LaTeX 公式为 PIL Image。

    支持完整的 LaTeX math mode 语法（分数、积分、矩阵、上下标等）。
    自动二分搜索字号以填满给定区域。

    Args:
        latex: LaTeX 公式字符串（可以带或不带 $ 包裹）
        width, height: 输出图像尺寸
        text_color: 文字颜色
        background_color: 背景颜色

    Returns:
        渲染后的 PIL Image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 确保被 $ 包裹
    formula = latex.strip()
    if not formula.startswith("$"):
        formula = f"${formula}$"

    fg = text_color
    bg = background_color

    # 自适应字号：二分搜索
    dpi = 150
    best_fontsize = 12
    lo, hi = 8, 200

    for _ in range(15):
        mid = (lo + hi) // 2
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor(bg)
        try:
            fig.text(
                0.5, 0.5, formula,
                fontsize=mid, color=fg,
                ha="center", va="center",
                math_fontfamily="cm",
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            texts = fig.texts
            if texts:
                bb = texts[0].get_window_extent(renderer)
                tw, th = bb.width, bb.height
                if tw <= width * 0.95 and th <= height * 0.9:
                    best_fontsize = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            else:
                hi = mid - 1
        except Exception:
            hi = mid - 1
        finally:
            plt.close(fig)

    # 最终渲染
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    fig.text(
        0.5, 0.5, formula,
        fontsize=best_fontsize, color=fg,
        ha="center", va="center",
        math_fontfamily="cm",
    )

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, facecolor=bg, bbox_inches="tight", pad_inches=0.05)
    except (ValueError, RuntimeError) as e:
        plt.close(fig)
        print(f"[formula_helper] matplotlib 渲染失败，降级为纯文本: {e}")
        plain = latex.strip().strip("$")
        return render_plaintext(plain, width, height, text_color, background_color)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)

    return img


# ============ 纯文本字体渲染 ============


def get_available_font(size: int = 100) -> ImageFont.FreeTypeFont:
    """获取系统中可用的字体"""
    possible_fonts = [
        # macOS
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    print("警告：使用默认字体")
    return ImageFont.load_default()


def calculate_font_size(text: str, bbox_width: int, bbox_height: int) -> int:
    """计算能填满 bbox 的字体大小"""
    estimated_size = int(bbox_height * 0.8)
    font = get_available_font(estimated_size)

    test_img = Image.new("RGB", (bbox_width * 2, bbox_height * 2), "white")
    draw = ImageDraw.Draw(test_img)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    scale_w = bbox_width / max(text_width, 1)
    scale_h = bbox_height / max(text_height, 1)
    scale = min(scale_w, scale_h) * 0.9

    return max(int(estimated_size * scale), 12)


def render_plaintext(
    text: str,
    width: int,
    height: int,
    text_color: str = "black",
    background_color: str = "white",
) -> Image.Image:
    """使用 PIL + 系统字体渲染纯文本。"""
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)

    font_size = calculate_font_size(text, width, height)
    font = get_available_font(font_size)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=text_color, font=font)
    return img


# ============ 统一入口 ============


def render_formula(
    text: str,
    width: int,
    height: int,
    text_color: str = "black",
    background_color: str = "white",
    force_latex: bool = False,
) -> Image.Image:
    """渲染公式/文本图像（自动检测渲染路径）。

    优先级：
    1. MathJax (Node.js) — 完整 LaTeX 支持（array, matrix, cases 等）
    2. matplotlib mathtext — 无需 Node.js，支持常用 LaTeX 子集
    3. PIL 纯文本 — 最后兜底
    """
    use_latex = force_latex or is_latex(text)

    if not use_latex:
        converted = plaintext_to_latex(text)
        if converted != text:
            use_latex = True
            text = converted

    if use_latex:
        # 优先尝试 MathJax（完整 LaTeX 支持）
        img = render_mathjax(text, width, height, text_color, background_color)
        if img is not None:
            return img
        # fallback 到 matplotlib（子集支持，失败时降级纯文本）
        return render_latex(text, width, height, text_color, background_color)
    else:
        return render_plaintext(text, width, height, text_color, background_color)


# ============ 测试入口 ============


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path("output/formula_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = 512, 192

    test_cases = [
        # (名称, 公式文本, 是否强制latex)
        ("25_bmatrix_jacobian",
         r"$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$",
         False),
    ]

    print(f"渲染 {len(test_cases)} 个测试用例到 {out_dir}")
    print(f"图像尺寸: {W}x{H}")
    print("=" * 60)

    for name, text, force in test_cases:
        detected = "latex" if (force or is_latex(text)) else "auto"
        converted = plaintext_to_latex(text)
        if not force and not is_latex(text) and converted != text:
            detected = "unicode→latex"

        print(f"\n[{name}] 检测: {detected}")
        print(f"  输入: {text[:80]}{'...' if len(text) > 80 else ''}")

        img = render_formula(text, W, H, force_latex=force)
        path = out_dir / f"{name}.png"
        img.save(path)
        print(f"  保存: {path}")

    print(f"\n{'=' * 60}")
    print(f"全部完成！共 {len(test_cases)} 张图，保存在 {out_dir}")
