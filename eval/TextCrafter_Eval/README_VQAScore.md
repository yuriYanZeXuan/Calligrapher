# VQAScore 本地实现

本项目在 `eval/TextCrafter_Eval/vqascore.py` 中提供了兼容 `t2v_metrics.VQAScore` 的精简实现。

## 特点

- **无需安装 t2v_metrics 包**：直接使用 transformers 库实现
- **兼容原有接口**：`VQAScore(model='clip-flant5-xxl', device='cuda')`
- **支持批量处理**：`forward()` 和 `batch_forward()` 方法

## 使用方法

### 方法1: 直接导入

```python
from eval.TextCrafter_Eval.vqascore import VQAScore

# 初始化模型
model = VQAScore(model='clip-flant5-xxl', device='cuda')

# 计算分数
scores = model(images=['image1.png', 'image2.png'], 
               texts=['A photo of a cat', 'A photo of a dog'])
# scores: tensor([0.85, 0.92])
```

### 方法2: 通过 metrics 模块

```python
from eval.core.metrics import VQAScoreMetrics

# 初始化
vqa = VQAScoreMetrics(model='clip-flant5-xxl', device='cuda')

# 计算单个分数
score = vqa.compute_score('image.png', 'A photo of a cat')

# 批量计算
scores = vqa.compute_batch(['img1.png', 'img2.png'], 
                           ['text1', 'text2'])
```

### 方法3: TextCrafter Evaluator

```python
from eval.TextCrafter_Eval.unified_metrics_eval import UnifiedMetricsEvaluator

evaluator = UnifiedMetricsEvaluator()
score = evaluator.compute_vqa_score('image.png', 'prompt text')
```

## 与原始 t2v_metrics 的区别

| 特性 | t2v_metrics.VQAScore | 本地 VQAScore |
|------|---------------------|---------------|
| 依赖包 | 需要完整 t2v_metrics | 仅需 transformers |
| 模型加载 | 通过 t2v_metrics 封装 | 直接使用 HuggingFace |
| 接口 | `model(images=..., texts=...)` | 完全相同 |
| 返回值 | torch.Tensor | torch.Tensor |

## 支持的模型

- `clip-flant5-xxl` (推荐，默认)
- `clip-flant5-xl` (较小，更快)

## 注意事项

1. 首次运行会下载模型权重（约 10GB+ for xxl）
2. 需要足够的 GPU 显存（建议 24GB+）
3. 如果显存不足，可以使用 `device='cpu'` 但速度较慢
