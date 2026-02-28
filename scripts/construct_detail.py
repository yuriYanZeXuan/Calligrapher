#!/usr/bin/env python3
"""
Construct detail.jsonl for result directories that don't have one.

Two modes:
  1. --no_vlm (default for baselines): real_prompt = original prompt
  2. --use_vlm: call VLM to generate clean_prompt, then real_prompt = clean_prompt + text

Reads benchmark JSONL to get prompt + text info, scans result images,
and writes detail.jsonl.  Supports --resume (skip already-existing entries).

Usage:
    python scripts/construct_detail.py \
        --results_dir baselines/results/ours/LongText-Bench \
        --benchmark eval/LongText-Bench \
        --benchmark_type longtext \
        --use_vlm

    python scripts/construct_detail.py \
        --results_dir baselines/results/textflux/LongText-Bench \
        --benchmark eval/LongText-Bench \
        --benchmark_type longtext
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _build_real_prompt(clean_prompt: str, text_list: list) -> str:
    if not text_list:
        return clean_prompt
    text_desc = ", ".join(f"'{t}'" for t in text_list)
    return f"{clean_prompt}, with text {text_desc}"


def load_benchmark(benchmark_path: str, benchmark_type: str) -> List[Dict]:
    from eval.scripts.eval_parallel import load_benchmark as _load
    return _load(benchmark_path, benchmark_type)


def load_existing_detail(detail_path: str) -> Dict[str, Dict]:
    existing = {}
    if not os.path.exists(detail_path):
        return existing
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                existing[entry.get("id", "")] = entry
            except json.JSONDecodeError:
                continue
    return existing


def find_image_for_sample(results_dir: str, sample_id: str) -> str | None:
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(results_dir, f"result_{sample_id}{ext}")
        if os.path.exists(p):
            return p
        p = os.path.join(results_dir, f"{sample_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Construct detail.jsonl for results without one")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--benchmark_type", type=str, default="longtext",
                        choices=["longtext", "oneig", "cvtg", "unseenwords", "generic"])
    parser.add_argument("--use_vlm", action="store_true",
                        help="Call VLM to generate clean_prompt (for our pipeline results)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already in detail.jsonl")
    args = parser.parse_args()

    detail_path = os.path.join(args.results_dir, "detail.jsonl")
    existing = load_existing_detail(detail_path) if args.resume else {}

    dataset = load_benchmark(args.benchmark, args.benchmark_type)
    print(f"Loaded {len(dataset)} benchmark samples")

    vlm_agent = None
    if args.use_vlm:
        from infer.VLM_agent import VLMAgent
        vlm_agent = VLMAgent()

    written = 0
    skipped = 0
    for sample in dataset:
        sample_id = sample["id"]
        result_id = f"result_{sample_id}"

        if args.resume and result_id in existing:
            skipped += 1
            continue

        img_path = find_image_for_sample(args.results_dir, sample_id)
        if img_path is None:
            continue

        prompt = sample["prompt"]
        text_list = sample.get("text", [])
        if isinstance(text_list, str):
            text_list = [text_list] if text_list.strip() else []

        if args.use_vlm and vlm_agent is not None:
            clean_prompt = vlm_agent.generate_clean_prompt(prompt)
        else:
            clean_prompt = prompt

        real_prompt = _build_real_prompt(clean_prompt, text_list)

        entry = {
            "id": result_id,
            "prompt": prompt,
            "clean_prompt": clean_prompt,
            "real_prompt": real_prompt,
            "text": text_list,
            "image_path": img_path,
        }

        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        written += 1

    print(f"Done. Written {written} entries, skipped {skipped} (resume).")
    print(f"detail.jsonl: {detail_path}")


if __name__ == "__main__":
    main()
