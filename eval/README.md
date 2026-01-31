# Calligrapher Evaluation Framework

统一的文本渲染评测框架，支持生成任务（Generation）和编辑任务（Editing）两种评测模式。

## 目录结构

```
eval/
├── core/                      # 核心评测模块（推荐）
│   ├── __init__.py
│   ├── base_evaluator.py      # 基础评测器类
│   ├── generation_evaluator.py # 生成评测器
│   ├── editing_evaluator.py   # 编辑评测器
│   └── metrics.py             # 统一指标计算模块
├── configs/                   # 配置文件
│   ├── generation_config.yaml # 生成评测配置示例
│   └── editing_config.yaml    # 编辑评测配置示例
├── scripts/                   # 启动脚本
│   ├── evaluate.py            # 统一入口脚本（推荐）
│   ├── run_generation_eval.py # 生成评测脚本
│   ├── run_editing_eval.py    # 编辑评测脚本
│   ├── example_generation.sh  # 生成评测示例
│   └── example_editing.sh     # 编辑评测示例
├── bak/                       # 旧版兼容脚本（归档）
│   ├── eval_ocr.py            # OCR 评测（旧版）
│   ├── eval_dino.py           # DINO 评测（旧版）
│   ├── eval_fid.py            # FID 评测（旧版）
│   ├── eval_vlm.py            # VLM 评测（旧版）
│   ├── mineru_ocr.py          # MinerU OCR 工具
│   ├── run_evaluation.py      # 旧版入口脚本
│   └── utils.py               # 旧版工具函数
├── OneIG-Bench/               # OneIG-Bench 数据集
├── CVTG-2K/                   # CVTG-2K 数据集
├── LongText-Bench/            # LongText-Bench 数据集
└── README.md                  # 本文件
```

## 快速开始

### 1. 生成评测（Text-to-Image Generation）

适用于评测从文本生成图像的模型，如 OneIG-Bench、CVTG-2K 等。

```bash
# 使用配置文件
python eval/scripts/evaluate.py \
    --mode generation \
    --config eval/configs/generation_config.yaml

# 使用命令行参数
python eval/scripts/evaluate.py \
    --mode generation \
    --benchmark eval/OneIG-Bench/OneIG-Bench.json \
    --benchmark_type oneig \
    --generated outputs/my_results \
    --output eval_results/my_results.json \
    --metrics ocr clip

# 筛选特定类别
python eval/scripts/evaluate.py \
    --mode generation \
    --benchmark eval/OneIG-Bench/OneIG-Bench.json \
    --filter_categories Text_Rendering \
    --generated outputs/my_results \
    --metrics ocr clip
```

### 2. 编辑评测（Image Editing with Mask）

适用于评测基于掩码的图像编辑模型，如 Calligrapher 编辑任务。

```bash
# 使用配置文件
python eval/scripts/evaluate.py \
    --mode editing \
    --config eval/configs/editing_config.yaml

# 使用命令行参数
python eval/scripts/evaluate.py \
    --mode editing \
    --benchmark eval/Calligrapher_bench_testing \
    --benchmark_format directory \
    --generated outputs/editing_results \
    --output eval_results/editing_results.json \
    --metrics ocr dino clip
```

## 支持的 Benchmarks

### 生成评测支持的 Benchmarks

| Benchmark | Type | 描述 |
|-----------|------|------|
| OneIG-Bench | `oneig` | 包含 Text_Rendering、Anime_Stylization 等类别 |
| CVTG-2K | `cvtg` | 包含 CVTG 和 CVTG-Style 两个子集 |
| LongText-Bench | `longtext` | 长文本生成评测 |
| 自定义 | `generic` | 通用 JSON 格式 |

### 编辑评测支持的 Benchmark Formats

| Format | 描述 | 示例 |
|--------|------|------|
| `directory` | 目录结构，包含 source/mask/ref | `{id}_source.png`, `{id}_mask.png` |
| `txt` | TXT 文件列表 | `id\tsource\tref\tprompt` |
| `json` | JSON 格式 | `[{"id": "", "source": "", "mask": "", "prompt": ""}]` |

## 支持的评测指标

| 指标 | 生成评测 | 编辑评测 | 说明 |
|------|---------|---------|------|
| `ocr` | ✓ | ✓ | 基于 MinerU 的字符级 OCR 准确率 |
| `clip` | ✓ | ✓ | CLIP 分数，衡量图文匹配度 |
| `dino` | - | ✓ | DINOv2 特征相似度（需要参考图像） |
| `fid` | ✓ | ✓ | Frechet Inception Distance（分布质量） |
| `vlm` | ✓ | ✓ | 视觉语言模型评测（需要 API_KEY） |

## 配置文件说明

### 生成评测配置示例

```yaml
benchmark:
  path: "eval/OneIG-Bench/OneIG-Bench.json"
  type: "oneig"
  filter_categories:
    - "Text_Rendering"

generation:
  output_dir: "outputs/generation"

metrics:
  - ocr
  - clip

output:
  result_file: "eval_results/generation_results.json"

device:
  type: "auto"
```

### 编辑评测配置示例

```yaml
benchmark:
  path: "eval/Calligrapher_bench_testing"
  format: "directory"

generation:
  output_dir: "outputs/editing"

metrics:
  - ocr
  - dino
  - clip

mask:
  required: true
  use_masked_metrics: true

output:
  result_file: "eval_results/editing_results.json"
```

## 从项目根目录调用

```bash
# 生成评测
python -m eval.scripts.evaluate \
    --mode generation \
    --benchmark eval/OneIG-Bench/OneIG-Bench.json \
    --generated outputs/results \
    --metrics ocr clip

# 编辑评测
python -m eval.scripts.evaluate \
    --mode editing \
    --benchmark eval/Calligrapher_bench_testing \
    --generated outputs/editing \
    --metrics ocr dino
```

## 环境要求

```bash
pip install pyyaml pandas tqdm pillow numpy torch torchvision

# OCR 评测需要
pip install levenshtein
# 安装 MinerU（参考项目文档）

# CLIP 评测需要
pip install git+https://github.com/openai/CLIP.git

# VLM 评测需要设置 API_KEY
export API_KEY="your-api-key"
```

## 结果格式

评测结果以 JSON 格式保存，包含：

```json
{
  "detailed_results": [
    {
      "id": "sample_001",
      "prompt": "A sign saying 'Hello'",
      "ocr_accuracy": 0.95,
      "clip_score": 28.5
    }
  ],
  "summary": {
    "mean_ocr_accuracy": 0.92,
    "mean_clip_score": 27.8,
    "std_ocr_accuracy": 0.05
  }
}
```

## 扩展开发

如需添加新的评测器或指标，参考 `core/` 目录下的实现：

1. 继承 `BaseEvaluator` 创建新的评测器
2. 在 `metrics.py` 中添加新的指标计算类
3. 更新配置文件示例

## 与旧版评测的兼容性

旧版独立脚本已归档在 `bak/` 目录中：
- `bak/eval_ocr.py` - 独立 OCR 评测
- `bak/eval_dino.py` - 独立 DINO 评测  
- `bak/eval_fid.py` - 独立 FID 评测
- `bak/eval_vlm.py` - 独立 VLM 评测
- `bak/run_evaluation.py` - 旧版入口脚本

**建议**：新项目请使用新的统一入口 `eval/scripts/evaluate.py`，
旧版脚本仅保留用于向后兼容。
