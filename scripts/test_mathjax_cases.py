#!/usr/bin/env python3
"""诊断 MathJax 渲染链路 — 定位 Node.js 不可用的具体原因。

用法:
  python scripts/test_mathjax_cases.py          # 基本诊断
  python scripts/test_mathjax_cases.py --render  # 诊断 + 实际渲染测试
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def diagnose_node():
    """诊断 Node.js 环境。"""
    print("=" * 60)
    print("1. 环境信息")
    print("=" * 60)
    print(f"  Python:       {sys.executable}")
    print(f"  PID:          {os.getpid()}")
    print(f"  CWD:          {os.getcwd()}")
    print(f"  USER:         {os.environ.get('USER', 'N/A')}")
    print(f"  SHELL:        {os.environ.get('SHELL', 'N/A')}")
    print(f"  CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'N/A')}")
    print(f"  VIRTUAL_ENV:  {os.environ.get('VIRTUAL_ENV', 'N/A')}")
    print()

    print("=" * 60)
    print("2. PATH 分析")
    print("=" * 60)
    path_dirs = os.environ.get("PATH", "").split(":")
    print(f"  PATH 共 {len(path_dirs)} 项:")
    node_found_in = []
    for i, d in enumerate(path_dirs):
        node_path = os.path.join(d, "node")
        exists = os.path.isfile(node_path) and os.access(node_path, os.X_OK)
        marker = " ★ node HERE" if exists else ""
        if exists:
            node_found_in.append(node_path)
        print(f"    [{i:2d}] {d}{marker}")
    print()

    print("=" * 60)
    print("3. shutil.which('node') — formula_helper 用的检测方式")
    print("=" * 60)
    which_result = shutil.which("node")
    print(f"  shutil.which('node') = {which_result}")
    if which_result is None and node_found_in:
        print(f"  ⚠ 但 PATH 中确实有 node: {node_found_in}")
        print("    可能原因: 文件权限问题 / PATH 项末尾有空格")
    print()

    print("=" * 60)
    print("4. 常见安装路径探测")
    print("=" * 60)
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", "node"),
        os.path.join(home, ".nvm", "current", "bin", "node"),
        "/usr/local/bin/node",
        "/usr/bin/node",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.insert(0, os.path.join(conda_prefix, "bin", "node"))

    for p in candidates:
        if os.path.isfile(p):
            executable = os.access(p, os.X_OK)
            in_path = p in node_found_in or shutil.which("node") == p
            print(f"  ✓ {p}  (executable={executable}, in_PATH={in_path})")
        else:
            print(f"  ✗ {p}")
    print()

    print("=" * 60)
    print("5. subprocess 直接调用测试")
    print("=" * 60)
    # 5a. 直接用 "node"
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        print(f"  node --version: {r.stdout.strip()} (exit={r.returncode})")
    except FileNotFoundError:
        print("  node --version: FileNotFoundError — 系统找不到 node")
    except Exception as e:
        print(f"  node --version: {e}")

    # 5b. 用绝对路径（如果找到了）
    for abs_node in node_found_in:
        try:
            r = subprocess.run([abs_node, "--version"], capture_output=True, text=True, timeout=5)
            print(f"  {abs_node} --version: {r.stdout.strip()}")
        except Exception as e:
            print(f"  {abs_node} --version: {e}")
    print()

    print("=" * 60)
    print("6. MathJax 脚本检测")
    print("=" * 60)
    mathjax_js = Path(__file__).parent.parent / "infer" / "render_mathjax.js"
    print(f"  脚本路径: {mathjax_js}")
    print(f"  存在: {mathjax_js.exists()}")
    if mathjax_js.exists():
        mathjax_dir = mathjax_js.parent.parent / "dataset_pipeline" / "LTB" / "assets" / "mathjax" / "es5"
        print(f"  MathJax es5 目录: {mathjax_dir}")
        print(f"  es5 目录存在: {mathjax_dir.exists()}")
        if mathjax_dir.exists():
            startup = mathjax_dir / "startup.js"
            print(f"  startup.js 存在: {startup.exists()}")
    print()

    return which_result is not None


def render_test():
    """实际渲染测试。"""
    from infer.formula_helper import render_mathjax, _check_node

    # 强制重新检测（绕过缓存）
    import infer.formula_helper as fh
    fh._node_available = None

    print("=" * 60)
    print("7. 渲染测试 (重新检测 node)")
    print("=" * 60)
    print(f"  _check_node() = {_check_node()}")
    if not _check_node():
        print("  Node.js 不可用，跳过渲染测试")
        return

    cases = [
        ("欧拉恒等式", r"e^{i\pi}+1=0"),
        ("质能方程", r"E=mc^2"),
        ("化学方程式", r"3Cu+8HNO_3(dilute)\rightarrow 3Cu(NO_3)_2+2NO\uparrow +4H_2O"),
        ("极限", r"\lim_{x\rightarrow \infty} (1+1/x)^x=e"),
        ("薛定谔方程", r"i\hbar \frac{\partial \psi}{\partial t}=\hat{H}\psi"),
        ("哈密顿算符", r"\hat{H}=-\frac{\hbar^2}{2m}\nabla^2+V"),
        ("本征方程", r"\hat{H}\psi =E\psi"),
        ("蔗糖水解", r"C_{12}H_{22}O_{11}+H_2O\rightarrow"),
    ]

    passed = failed = 0
    for name, latex in cases:
        try:
            img = render_mathjax(latex, 512, 128)
            if img is not None:
                print(f"  ✓ {name:10s}  {latex[:50]}")
                passed += 1
            else:
                print(f"  ✗ {name:10s}  返回 None  {latex[:50]}")
                failed += 1
        except Exception as e:
            print(f"  ✗ {name:10s}  异常: {e}")
            failed += 1

    print(f"\n  结果: {passed} 通过, {failed} 失败 / 共 {len(cases)} 条")


def main():
    node_ok = diagnose_node()

    if "--render" in sys.argv:
        render_test()
    elif not node_ok:
        print("提示: 加 --render 参数可跳过缓存重新检测并渲染测试")

    print()
    if not node_ok:
        print("=" * 60)
        print("修复建议")
        print("=" * 60)
        home = os.path.expanduser("~")
        local_node = os.path.join(home, ".local", "bin", "node")
        if os.path.isfile(local_node):
            print(f"  node 已安装在 {local_node} 但不在 PATH 中。")
            print(f"  在启动脚本(.bashrc)或运行命令前加:")
            print(f'    export PATH="{home}/.local/bin:$PATH"')
        else:
            print("  运行: bash scripts/setup_mathjax.sh")
            print("  然后确保新 shell 或 source ~/.bashrc 后再启动推理进程")


if __name__ == "__main__":
    main()
