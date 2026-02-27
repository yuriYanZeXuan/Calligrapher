#!/usr/bin/env python3
"""
单条 case 调试脚本 — 定位 glyph 渲染 / mask / Pass 2/3 问题。

用法:
  # 1. 直接指定 prompt + text
  python debug_single_case.py \
      --prompt 'A sign displays "小狗" in bold font.' \
      --text '小狗'

  # 2. 从 benchmark jsonl 中按 id 提取
  python debug_single_case.py \
      --benchmark UnseenWords \
      --sample-id unseen_ez_zh_3

  # 3. 自定义选项
  python debug_single_case.py \
      --prompt 'A poster shows "$E=mc^2$".' \
      --text '$E=mc^2$' \
      --no-harmonize \
      --freq-decompose
"""

import os
import sys
import json
import glob
import argparse

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def find_sample_in_benchmark(benchmark: str, sample_id: str) -> dict:
    """从 benchmark 数据集中查找指定 id 的 sample。"""
    eval_dir = os.path.join(ROOT, "eval")

    if benchmark == "UnseenWords":
        search_dir = os.path.join(eval_dir, "UnseenWords")
        for f in sorted(glob.glob(os.path.join(search_dir, "*.jsonl"))):
            prefix = os.path.splitext(os.path.basename(f))[0]
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    pid = item.get("prompt_id", "")
                    full_id = f"{prefix}_{pid}"
                    if full_id == sample_id:
                        return {
                            "id": full_id,
                            "prompt": item["prompt"],
                            "text": item.get("text", []),
                        }

    elif benchmark == "LongText-Bench":
        search_dir = os.path.join(eval_dir, "LongText-Bench")
        for f in sorted(glob.glob(os.path.join(search_dir, "*.jsonl"))):
            lang = "zh" if "zh" in f else "en"
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    pid = item.get("prompt_id", "")
                    full_id = f"longtext_{lang}_{pid}"
                    if full_id == sample_id:
                        return {
                            "id": full_id,
                            "prompt": item["prompt"],
                            "text": item.get("text", []),
                        }

    raise ValueError(f"sample_id '{sample_id}' not found in {benchmark}")


def main():
    parser = argparse.ArgumentParser(description="单条 case 调试")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", type=str, help="直接指定 prompt")
    g.add_argument("--sample-id", type=str, help="从 benchmark 中按 id 提取")

    parser.add_argument("--text", nargs="+", default=None, help="待渲染文本列表")
    parser.add_argument("--benchmark", type=str, default="UnseenWords",
                        choices=["UnseenWords", "LongText-Bench"])
    parser.add_argument("--output", type=str, default="debug_output.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--no-harmonize", action="store_true")
    parser.add_argument("--no-refiner", action="store_true")
    parser.add_argument("--freq-decompose", action="store_true")
    parser.add_argument("--harmonizer-type", type=str, default="klein",
                        choices=["klein", "qwenedit"])

    args = parser.parse_args()

    # 获取 sample
    if args.sample_id:
        sample = find_sample_in_benchmark(args.benchmark, args.sample_id)
        prompt = sample["prompt"]
        text = sample["text"]
        run_name = sample["id"]
        print(f"=== 从 {args.benchmark} 提取: {args.sample_id} ===")
    else:
        prompt = args.prompt
        text = args.text or []
        run_name = "debug"

    print(f"Prompt: {prompt}")
    print(f"Text:   {text}")
    print()

    from zimage_inference import ZImageInference, GenerationConfig
    from infer.glyph_injector import InjectionConfig
    from infer.mylogger import TTSLogger

    logger = TTSLogger(run_name=f"debug_{run_name}")

    injection_config = InjectionConfig(freq_decompose=args.freq_decompose)
    config = GenerationConfig(
        seed=args.seed,
        use_prompt_refiner=not args.no_refiner,
        use_glyph_injection=True,
        injection_config=injection_config,
        use_harmonization=not args.no_harmonize,
        harmonizer_type=args.harmonizer_type,
        klein_model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein",
        qwenedit_model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_edit_2511",
    )

    inference = ZImageInference(device=args.device, logger=logger)

    image = inference.generate(
        prompt=prompt,
        text_contents=text if text else None,
        config=config,
        run_name=run_name,
    )

    image.save(args.output)
    print(f"\n最终结果: {args.output}")
    print(f"日志目录: {logger.run_dir}")


if __name__ == "__main__":
    main()
