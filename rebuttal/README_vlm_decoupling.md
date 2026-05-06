# VLM Decoupling Rebuttal Experiment

Purpose: address evaluation circularity by decoupling the VLM used during generation/planning from the VLM used for judging/evaluation.

## Generation

Run one generation job per planner VLM:

```bash
cd /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher

PLANNER=qwen BENCHMARK=UnseenWords GPUS=1 SAMPLE_SIZE=5 SAMPLE_SEED=42 \
  ./rebuttal/run_vlm_decoupling_generation.sh

# For Gemini, start Paper2Slides/gemini_proxy.py first.
PLANNER=gemini BENCHMARK=UnseenWords GPUS=1 SAMPLE_SIZE=5 SAMPLE_SEED=42 \
  ./rebuttal/run_vlm_decoupling_generation.sh
```

Default outputs:

```text
rebuttal/results/generation/<planner>/<benchmark>/result_<id>.png
```

## Evaluation

Evaluate all planner outputs with independent judge VLMs:

```bash
BENCHMARK=UnseenWords SAMPLE_SIZE=5 SAMPLE_SEED=42 ./rebuttal/run_vlm_decoupling_eval.sh \
  --planner-dir qwen=rebuttal/results/generation/qwen/UnseenWords \
  --planner-dir gemini=rebuttal/results/generation/gemini/UnseenWords \
  --judge qwen \
  --judge gemini
```

Outputs:

```text
rebuttal/results/vlm_decoupling/<benchmark>/vlm_decoupling_details.csv
rebuttal/results/vlm_decoupling/<benchmark>/vlm_decoupling_summary.csv
rebuttal/results/vlm_decoupling/<benchmark>/vlm_decoupling_summary.md
```

## Rebuttal Table

Use `vlm_decoupling_summary.md` directly in the rebuttal draft. The intended comparison is:

- planner=qwen, judge=gemini
- planner=gemini, judge=qwen
- optionally planner=qwen, judge=qwen and planner=gemini, judge=gemini as diagonal references

If the off-diagonal settings remain strong, this addresses the concern that the reported gain only comes from using the same VLM for planning and evaluation.
