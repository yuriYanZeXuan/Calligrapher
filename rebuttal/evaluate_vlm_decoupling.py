#!/usr/bin/env python3
"""Evaluate VLM circularity by decoupling planner and judge models.

This script reads generated images from planner-specific result directories and
evaluates each directory with one or more VLM judge backends. It writes:

- vlm_decoupling_details.csv: per-sample VLM scores.
- vlm_decoupling_summary.csv: mean scores per planner/judge pair.
- vlm_decoupling_summary.md: compact rebuttal-ready table.

API keys are read from environment variables only and are never written to disk.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from baselines.run_parallel_benchmark import load_dataset  # noqa: E402
from eval.core.metrics import VLMMetrics  # noqa: E402


@dataclass(frozen=True)
class VLMBackend:
    alias: str
    model: str
    base_url: str | None
    api_key: str | None
    api_key2: str | None = None


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def resolve_backend(alias: str) -> VLMBackend:
    """Resolve common backend aliases.

    For Gemini, start `Paper2Slides/gemini_proxy.py` first and set
    `GEMINI_PROXY_BASE_URL` if not using the default local endpoint.
    """
    alias_l = alias.lower()
    if alias_l in {"qwen", "qwen3", "qwen3-vl"}:
        return VLMBackend(
            alias="qwen",
            model=_env("QWEN_VLM_MODEL") or "qwen3-vl-235b-a22b-instruct",
            base_url=_env("QST_BASE_URL"),
            api_key=_env("QST_API_KEY"),
            api_key2=_env("QST_API_KEY2"),
        )
    if alias_l in {"gemini", "gemini3", "gemini-3-pro"}:
        return VLMBackend(
            alias="gemini",
            model=_env("GEMINI_VLM_MODEL") or "gemini-3-pro",
            base_url=_env("GEMINI_PROXY_BASE_URL") or "http://127.0.0.1:51958/v1",
            api_key=_env("GEMINI3_API_KEY"),
        )
    if alias_l in {"gpt4o", "gpt-4o", "gpt"}:
        return VLMBackend(
            alias="gpt4o",
            model=_env("GPT4O_VLM_MODEL") or "gpt-4o",
            base_url=_env("GPT4O_BASE_URL") or "https://runway.devops.rednote.life/openai",
            api_key=_env("GPT4O_API_KEY"),
        )

    prefix = f"VLM_{alias.upper()}_"
    return VLMBackend(
        alias=alias,
        model=_env(prefix + "MODEL") or alias,
        base_url=_env(prefix + "BASE_URL"),
        api_key=_env(prefix + "API_KEY"),
        api_key2=_env(prefix + "API_KEY2"),
    )


def set_eval_env(backend: VLMBackend) -> None:
    os.environ["GLYPH_EVAL_VLM_MODEL"] = backend.model
    if backend.base_url:
        os.environ["GLYPH_EVAL_VLM_BASE_URL"] = backend.base_url
    else:
        os.environ.pop("GLYPH_EVAL_VLM_BASE_URL", None)
    if backend.api_key:
        os.environ["GLYPH_EVAL_VLM_API_KEY"] = backend.api_key
    else:
        os.environ.pop("GLYPH_EVAL_VLM_API_KEY", None)
    if backend.api_key2:
        os.environ["GLYPH_EVAL_VLM_API_KEY2"] = backend.api_key2
    else:
        os.environ.pop("GLYPH_EVAL_VLM_API_KEY2", None)


def parse_planner_dirs(values: Iterable[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Planner dir must be alias=/path, got: {value}")
        alias, path = value.split("=", 1)
        result.append((alias.strip(), Path(path).expanduser().resolve()))
    return result


def item_image_path(result_dir: Path, item: dict) -> Path:
    return result_dir / f"result_{item['id']}.png"


def evaluate_pair(
    planner_alias: str,
    result_dir: Path,
    judge: VLMBackend,
    dataset: list[dict],
    limit: int | None = None,
) -> list[dict]:
    set_eval_env(judge)
    metric = VLMMetrics(model_path="ApiCall")
    rows: list[dict] = []
    subset = dataset[:limit] if limit else dataset
    for item in subset:
        img_path = item_image_path(result_dir, item)
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        try:
            scores = metric.evaluate_text_rendering(image, item["prompt"])
            error = ""
        except Exception as exc:
            scores = {
                "text_accuracy": None,
                "text_ned": None,
                "image_quality": None,
                "faithfulness": None,
                "overall": None,
                "recognized_text": "",
                "ground_truth": "",
            }
            error = str(exc)
        rows.append({
            "planner": planner_alias,
            "judge": judge.alias,
            "judge_model": judge.model,
            "id": item["id"],
            "prompt": item["prompt"],
            "image_path": str(img_path),
            "text_accuracy": scores.get("text_accuracy"),
            "text_ned": scores.get("text_ned"),
            "image_quality": scores.get("image_quality"),
            "faithfulness": scores.get("faithfulness"),
            "overall": scores.get("overall"),
            "ground_truth": scores.get("ground_truth", ""),
            "recognized_text": scores.get("recognized_text", ""),
            "error": error,
        })
    return rows


def write_markdown_summary(summary: pd.DataFrame, path: Path) -> None:
    display_cols = [
        "planner", "judge", "n", "text_accuracy", "text_ned",
        "image_quality", "faithfulness", "overall",
    ]
    rows = []
    for _, row in summary[display_cols].iterrows():
        rows.append(
            "| {planner} | {judge} | {n} | {text_accuracy:.3f} | {text_ned:.3f} | "
            "{image_quality:.3f} | {faithfulness:.3f} | {overall:.3f} |".format(**row.to_dict())
        )
    content = "\n".join([
        "| Planner VLM | Judge VLM | N | Text Acc. | Text NED | Quality | Faith. | Overall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ])
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="UnseenWords",
                        choices=["CVTG-2K", "LongText-Bench", "OneIG-Bench", "OneIG-Bench-ZH", "UnseenWords"])
    parser.add_argument("--planner-dir", action="append", required=True,
                        help="Planner alias and result directory, e.g. qwen=/path/to/results")
    parser.add_argument("--judge", action="append", required=True,
                        help="Judge backend alias: qwen, gemini, gpt4o, or custom alias with VLM_<ALIAS>_* env vars")
    parser.add_argument("--output-dir", default=str(ROOT / "rebuttal" / "results" / "vlm_decoupling"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.benchmark, str(ROOT / "eval"))
    if args.sample_size is not None:
        rng = random.Random(args.sample_seed)
        sample_n = min(args.sample_size, len(dataset))
        dataset = sorted(rng.sample(dataset, sample_n), key=lambda x: str(x.get("id", "")))
        print(f"[sample] {sample_n} random samples (seed={args.sample_seed})")
    planner_dirs = parse_planner_dirs(args.planner_dir)
    judges = [resolve_backend(j) for j in args.judge]

    all_rows: list[dict] = []
    for planner_alias, result_dir in planner_dirs:
        for judge in judges:
            print(f"[eval] planner={planner_alias} dir={result_dir} judge={judge.alias}/{judge.model}")
            all_rows.extend(evaluate_pair(planner_alias, result_dir, judge, dataset, limit=args.limit))

    detail_path = output_dir / "vlm_decoupling_details.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        if all_rows:
            writer.writeheader()
            writer.writerows(all_rows)

    if not all_rows:
        raise RuntimeError("No images were evaluated. Check --planner-dir paths and benchmark IDs.")

    df = pd.DataFrame(all_rows)
    numeric_cols = ["text_accuracy", "text_ned", "image_quality", "faithfulness", "overall"]
    summary = (
        df.groupby(["planner", "judge"], as_index=False)
        .agg(n=("id", "count"), **{col: (col, "mean") for col in numeric_cols})
        .sort_values(["planner", "judge"])
    )
    summary_csv = output_dir / "vlm_decoupling_summary.csv"
    summary_md = output_dir / "vlm_decoupling_summary.md"
    summary.to_csv(summary_csv, index=False)
    write_markdown_summary(summary, summary_md)
    print(f"[done] details: {detail_path}")
    print(f"[done] summary: {summary_csv}")
    print(f"[done] markdown: {summary_md}")


if __name__ == "__main__":
    main()
