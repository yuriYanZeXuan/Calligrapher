#!/usr/bin/env python3
"""
诊断 MathJax 渲染回退问题

针对 unseen_ez_sci_10 的公式 "∇·E=ρ"，逐步检查 MathJax 管线每个环节。
"""

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from formula_helper import (
    _check_node, _MATHJAX_SCRIPT, _convert_svg_to_png,
    render_mathjax, render_latex, render_formula,
    plaintext_to_latex, is_latex,
)


FORMULA = "∇·E=ρ"
WIDTH, HEIGHT = 512, 192
OUT_DIR = Path("output/debug_mathjax")


def step(name: str):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step("0. 原始输入")
    print(f"  公式: {FORMULA!r}")
    print(f"  is_latex: {is_latex(FORMULA)}")
    converted = plaintext_to_latex(FORMULA)
    print(f"  plaintext_to_latex: {converted!r}")
    print(f"  转换后 is_latex: {is_latex(converted)}")

    step("1. Node.js 可用性")
    node_path = shutil.which("node")
    print(f"  shutil.which('node'): {node_path}")
    print(f"  _check_node(): {_check_node()}")
    if node_path:
        ver = subprocess.run(["node", "--version"], capture_output=True, text=True)
        print(f"  node version: {ver.stdout.strip()}")

    step("2. MathJax 脚本")
    print(f"  路径: {_MATHJAX_SCRIPT}")
    print(f"  exists: {_MATHJAX_SCRIPT.exists()}")
    mathjax_es5 = _MATHJAX_SCRIPT.parent.parent / "dataset_pipeline" / "LTB" / "assets" / "mathjax" / "es5"
    print(f"  MathJax es5 目录: {mathjax_es5}")
    print(f"  es5 exists: {mathjax_es5.exists()}")
    if mathjax_es5.exists():
        startup = mathjax_es5 / "startup.js"
        print(f"  startup.js exists: {startup.exists()}")

    step("3. 直接调用 node 渲染 SVG")
    test_latex = converted.strip("$") if converted.startswith("$") else converted
    print(f"  传入公式: {test_latex!r}")
    if _check_node() and _MATHJAX_SCRIPT.exists():
        result = subprocess.run(
            ["node", str(_MATHJAX_SCRIPT), test_latex],
            capture_output=True, text=True, timeout=15,
        )
        print(f"  returncode: {result.returncode}")
        print(f"  stderr: {result.stderr.strip()[:300]}")
        svg_str = result.stdout
        has_svg = "<svg" in svg_str if svg_str else False
        print(f"  stdout 长度: {len(svg_str)}")
        print(f"  含 <svg>: {has_svg}")
        if svg_str:
            print(f"  SVG 前 300 字符: {svg_str[:300]}")

        if has_svg:
            step("4. SVG → PNG (cairosvg)")
            img = _convert_svg_to_png(svg_str, WIDTH, HEIGHT, "white")
            print(f"  结果: {img}")
            if img is not None:
                path = OUT_DIR / "step4_svg_to_png.png"
                img.save(path)
                print(f"  保存: {path}")
    else:
        print("  [SKIP] node 或 MathJax 脚本不可用")

    step("5. cairosvg 可用性")
    try:
        import cairosvg
        print(f"  cairosvg version: {cairosvg.__version__}")
    except ImportError as e:
        print(f"  [FAIL] cairosvg 未安装: {e}")

    step("6. render_mathjax() 完整调用")
    img = render_mathjax(converted, WIDTH, HEIGHT, "white")
    print(f"  结果: {img}")
    if img is not None:
        path = OUT_DIR / "step6_mathjax.png"
        img.save(path)
        print(f"  保存: {path}")
    else:
        print("  [FAIL] MathJax 返回 None → 会回退到 matplotlib")

    step("7. render_latex() (matplotlib 回退)")
    img = render_latex(converted, WIDTH, HEIGHT, "white")
    path = OUT_DIR / "step7_matplotlib.png"
    img.save(path)
    print(f"  保存: {path}")

    step("8. render_formula() 完整流程")
    img = render_formula(FORMULA, WIDTH, HEIGHT, text_color="white")
    path = OUT_DIR / "step8_final.png"
    img.save(path)
    print(f"  保存: {path}")

    print(f"\n{'='*60}")
    print(f"诊断完成！输出目录: {OUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
