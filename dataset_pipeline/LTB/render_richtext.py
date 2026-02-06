#!/usr/bin/env python3
"""
富文本渲染模块：将 Markdown 内容渲染为单页图片
字体大小自适应，确保内容居中填充
"""

import tempfile
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright


def calculate_font_size(text_length: int, min_size: int = 14, max_size: int = 32) -> int:
    """根据文本长度计算合适的字体大小"""
    # 短文本用大字体，长文本用小字体
    if text_length <= 50:
        return max_size
    elif text_length <= 100:
        return 28
    elif text_length <= 200:
        return 24
    elif text_length <= 400:
        return 20
    elif text_length <= 800:
        return 16
    else:
        return min_size


def get_html_template(content_html: str, font_size: int = 20, 
                      page_width: int = 1000, lang: str = "en") -> str:
    """生成 HTML 模板"""
    # 获取本地 MathJax 路径
    mathjax_path = Path(__file__).parent / "assets" / "mathjax" / "es5" / "tex-svg.js"
    mathjax_url = f"file://{mathjax_path.absolute()}"
    
    # 根据语言设置字体和方向
    font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
    direction = "ltr"
    
    if lang == "ar":
        font_family = '"Noto Sans Arabic", "Segoe UI", Arial, sans-serif'
        direction = "rtl"
    elif lang in ["zh", "ja"]:
        font_family = '"Noto Sans SC", "Noto Sans JP", "PingFang SC", "Hiragino Sans", sans-serif'
    elif lang == "ko":
        font_family = '"Noto Sans KR", "Malgun Gothic", sans-serif'
    
    return f'''<!DOCTYPE html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="utf-8">
<style>
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}
html, body {{
    width: {page_width}px;
    min-height: 100vh;
    background: #ffffff;
}}
body {{
    font-family: {font_family};
    font-size: {font_size}px;
    line-height: 1.8;
    color: #1a1a1a;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 60px 80px;
}}
.content {{
    max-width: {page_width - 160}px;
    text-align: {"right" if lang == "ar" else "left"};
}}
h1, h2, h3 {{
    margin: 0.8em 0 0.5em 0;
    font-weight: 600;
    line-height: 1.4;
}}
h1 {{ font-size: 1.5em; }}
h2 {{ font-size: 1.3em; }}
h3 {{ font-size: 1.15em; }}
p {{
    margin: 0.6em 0;
    text-align: justify;
}}
code {{
    background: #f0f0f0;
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
    font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
}}
pre {{
    background: #f5f5f5;
    padding: 1em;
    overflow-x: auto;
    border-radius: 8px;
    margin: 0.8em 0;
}}
blockquote {{
    border-left: 4px solid #ddd;
    padding-left: 1em;
    color: #555;
    margin: 0.8em 0;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
}}
table th, table td {{
    border: 1px solid #ddd;
    padding: 0.5em 0.8em;
    text-align: {"right" if lang == "ar" else "left"};
}}
table th {{
    background: #f5f5f5;
    font-weight: 600;
}}
table tr:nth-child(even) {{
    background: #fafafa;
}}
.MathJax {{
    font-size: 1.1em !important;
}}
.MathJax_Display {{
    margin: 0.8em 0 !important;
    overflow-x: auto;
}}
</style>
<script>
window.MathJax = {{
    tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
    }},
    svg: {{ fontCache: 'global' }},
    startup: {{
        pageReady: () => {{
            return MathJax.startup.defaultPageReady();
        }}
    }}
}};
</script>
<script src="{mathjax_url}"></script>
</head>
<body>
<div class="content">
{content_html}
</div>
</body>
</html>'''


def render_content_to_image(content: str, output_path: Path, 
                            text_length: int = None,
                            page_width: int = 1000,
                            lang: str = "en") -> bool:
    """将 Markdown 内容渲染为图片
    
    Args:
        content: Markdown 格式的内容
        output_path: 输出图片路径
        text_length: 文本长度（用于计算字体大小）
        page_width: 页面宽度
        lang: 语言代码
    
    Returns:
        bool: 是否成功
    """
    # 计算文本长度（如果未提供）
    if text_length is None:
        import re
        clean = re.sub(r'[\$\\{}\[\]]', '', content)
        chinese = len(re.findall(r'[\u4e00-\u9fff]', clean))
        words = len(clean.split())
        text_length = chinese + words
    
    # 计算字体大小
    font_size = calculate_font_size(text_length)
    
    # 转换 Markdown 为 HTML
    html_body = markdown.markdown(
        content,
        extensions=["extra", "codehilite", "toc", "nl2br"],
    )
    
    # 生成完整 HTML
    html_full = get_html_template(html_body, font_size, page_width, lang)
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, 
                                      encoding='utf-8') as f:
        f.write(html_full)
        html_path = Path(f.name)
    
    # 使用 Playwright 渲染
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": page_width, "height": 800})
        page.goto(f"file://{html_path.absolute()}")
        page.wait_for_load_state("networkidle")
        
        # 等待 MathJax 渲染（如果有公式的话）
        # 检查是否有公式需要渲染
        has_math = page.evaluate("""() => {
            const text = document.body.textContent;
            return text.includes('$') || text.includes('\\\\(') || text.includes('\\\\[');
        }""")
        
        if has_math:
            # 等待 MathJax 加载和渲染
            try:
                page.wait_for_function(
                    "typeof window.MathJax !== 'undefined' && window.MathJax.typesetPromise",
                    timeout=5000
                )
                # 触发渲染
                page.evaluate("window.MathJax.typesetPromise()")
                # 等待渲染完成
                page.wait_for_timeout(1000)
            except Exception:
                # MathJax 加载失败或超时，继续执行
                pass
        else:
            # 没有公式，短暂等待页面稳定
            page.wait_for_timeout(300)
        
        # 获取内容实际高度
        content_height = page.evaluate('''() => {
            const content = document.querySelector('.content');
            const rect = content.getBoundingClientRect();
            return Math.ceil(rect.height + 120);  // 加上 padding
        }''')
        
        # 设置页面高度并截图
        min_height = 400
        page_height = max(min_height, content_height)
        
        # 调整尺寸为 32 的倍数 (向上取整)
        target_width = ((page_width + 31) // 32) * 32
        target_height = ((page_height + 31) // 32) * 32
        
        page.set_viewport_size({"width": target_width, "height": target_height})
        
        # 注入 CSS 确保背景为白色且填满视口
        page.add_style_tag(content=f"""
            html, body {{
                width: {target_width}px;
                height: {target_height}px;
                background-color: white;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
            body {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 60px 80px; /* 保持原有的内边距 */
            }}
            .content {{
                width: 100%;
                max-width: {page_width - 160}px; /* 保持内容宽度限制 */
            }}
        """)
        
        page.wait_for_timeout(200)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 截图 (clip 确保只截取视口大小，虽然 viewport 已经设置好了)
        page.screenshot(
            path=str(output_path), 
            clip={"x": 0, "y": 0, "width": target_width, "height": target_height}
        )
        browser.close()
    
    # 清理临时文件
    html_path.unlink()
    
    return True


def render_batch(items: list[dict], output_dir: Path, lang: str = "en") -> list[str]:
    """批量渲染
    
    Args:
        items: [{"id": str, "content": str, "text_length": int}, ...]
        output_dir: 输出目录
        lang: 语言代码
    
    Returns:
        list[str]: 成功渲染的图片路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    for item in items:
        output_path = output_dir / f"{item['id']}.png"
        render_content_to_image(
            content=item['content'],
            output_path=output_path,
            text_length=item.get('text_length'),
            lang=lang
        )
        results.append(str(output_path))
    
    return results


if __name__ == '__main__':
    # 测试
    test_content = """## Attention Mechanism

The attention function can be described as mapping a **query** and a set of key-value pairs to an output.

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

Where:
- $Q$ represents the query matrix
- $K$ represents the key matrix  
- $V$ represents the value matrix

| Component | Dimension |
|-----------|-----------|
| Query | $d_k$ |
| Key | $d_k$ |
| Value | $d_v$ |
"""
    
    output_path = Path(__file__).parent.parent / 'Web_rendered' / 'test_render.png'
    render_content_to_image(test_content, output_path, text_length=80, lang="en")
    print(f"渲染完成: {output_path}")
