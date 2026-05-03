#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='统计 text_prompts*.jsonl 中 "text" 字段的平均字符数'
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="可选输入文件；默认仅统计 text_prompts.jsonl 和 text_prompts_zh.jsonl",
    )
    return parser.parse_args()


def load_default_inputs(base_dir: Path) -> list[Path]:
    return [
        base_dir / "text_prompts.jsonl",
        base_dir / "text_prompts_zh.jsonl",
    ]


def collect_sample_lengths(path: Path) -> list[int]:
    lengths: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        texts = record.get("text")
        if isinstance(texts, str):
            lengths.append(len(texts))
            continue
        if isinstance(texts, list):
            lengths.append(sum(len(text) for text in texts if isinstance(text, str)))
    return lengths


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    paths = [Path(p).expanduser().resolve() for p in args.inputs] if args.inputs else load_default_inputs(base_dir)

    all_lengths: list[int] = []
    print('=== 仅统计 "text" 字段的字符长度（len），按样本级平均 ===')
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        lengths = collect_sample_lengths(path)
        if not lengths:
            print(f"{path.name}: 没有可统计的 text 字段")
            continue
        all_lengths.extend(lengths)
        print(
            f"{path.name}: 样本数={len(lengths)}, 平均字符数={sum(lengths) / len(lengths):.2f}"
        )

    if not all_lengths:
        raise ValueError("未在输入文件中找到可统计的 text 字段")
    print(
        f"overall: 样本数={len(all_lengths)}, 平均字符数={sum(all_lengths) / len(all_lengths):.2f}"
    )


if __name__ == "__main__":
    main()
