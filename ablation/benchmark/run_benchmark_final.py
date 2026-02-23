#!/usr/bin/env python3
"""
最终版一键启动脚本：生成 + 评测
- 支持 DrawTextExt, AnyText, CVTG-2K, LongText-Bench
- 每个数据集随机取100条
- OCR 使用 VLM 提取，准确度计算使用 Levenshtein
- Style score 使用 vlm_image_quality
"""

import os
import sys
import json
import re
import random
import torch
from PIL import Image
from pathlib import Path

# 固定配置
ROOT_DIR = "/Users/yanzexuan/code/Calligrapher"
BASELINES_DIR = os.path.join(ROOT_DIR, "baselines")
OUTPUT_DIR = os.path.join(ROOT_DIR, "ablation", "benchmark", "results")
BENCHMARK_DIR = os.path.join(ROOT_DIR, "ablation", "benchmark")
EVAL_DIR = os.path.join(ROOT_DIR, "eval")

# 模型路径
FLUX_KLEIN_PATH = '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein'
QWEN_IMAGE_PATH = '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen-image-2512'

sys.path.extend([ROOT_DIR, BASELINES_DIR])


def extract_text_from_prompt(prompt):
    """从 prompt 中提取引号内文本（与 metrics.py 对齐）"""
    patterns = [
        r'"([^"]+)"', r"'([^']+)'", r'"([^"]+)"', r'「([^」]+)」', r'『([^』]+)』'
    ]
    pattern = '|'.join(patterns)
    matches = re.findall(pattern, prompt)
    
    quoted_texts = []
    for match in matches:
        for group in match if isinstance(match, tuple) else [match]:
            if group:
                quoted_texts.append(group)
    
    return ' '.join(quoted_texts) if quoted_texts else prompt


def prepare_cvtg2k_subset():
    """准备 CVTG-2K 子集（100条）"""
    output_path = os.path.join(BENCHMARK_DIR, 'CVTG2K_subset.jsonl')
    if os.path.exists(output_path):
        print(f"CVTG-2K 子集已存在: {output_path}")
        return output_path
    
    random.seed(42)
    cvtg_dir = os.path.join(EVAL_DIR, 'CVTG-2K', 'CVTG')
    
    all_data = []
    for json_file in Path(cvtg_dir).glob('*_combined.json'):
        area = json_file.stem.replace('_combined', '')
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for idx_str, prompt in data.items():
            all_data.append({
                'id': f"cvtg_{area}_{idx_str}",
                'prompt': prompt,
                'area': area,
                'index': int(idx_str)
            })
    
    # 随机抽取100条
    sampled = random.sample(all_data, min(100, len(all_data)))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"CVTG-2K 子集已创建: {len(sampled)} 条 -> {output_path}")
    return output_path


def prepare_longtext_subset():
    """准备 LongText-Bench 子集（100条，中英文各50）"""
    output_path = os.path.join(BENCHMARK_DIR, 'LongText_subset.jsonl')
    if os.path.exists(output_path):
        print(f"LongText-Bench 子集已存在: {output_path}")
        return output_path
    
    random.seed(42)
    longtext_dir = os.path.join(EVAL_DIR, 'LongText-Bench')
    
    all_data = []
    for jsonl_file in Path(longtext_dir).glob('*.jsonl'):
        lang = 'zh' if 'zh' in jsonl_file.name else 'en'
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt_id = item.get('prompt_id', len(all_data))
                all_data.append({
                    'id': f"longtext_{lang}_{prompt_id}",
                    'prompt': item['prompt'],
                    'lang': lang,
                    'category': item.get('category', ''),
                    'length': item.get('length', '')
                })
    
    # 随机抽取100条
    sampled = random.sample(all_data, min(100, len(all_data)))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"LongText-Bench 子集已创建: {len(sampled)} 条 -> {output_path}")
    return output_path


class VLMEvaluator:
    """VLM 评测器"""
    def __init__(self):
        sys.path.insert(0, os.path.join(ROOT_DIR, "eval", "core"))
        from metrics import VLMMetrics
        self.metrics = VLMMetrics(model_path="ApiCall", device="cpu")
    
    def evaluate(self, image, prompt):
        """评测图像，返回 OCR 和 style 分数（style使用vlm_image_quality）"""
        result = self.metrics.evaluate_text_rendering(image, prompt)
        return {
            'pred_text': result.get('recognized_text', ''),
            'ocr_acc': result.get('text_accuracy', 0.0),
            'ocr_ned': result.get('text_ned', 0.0),
            'vlm_text_accuracy': result.get('text_accuracy', 0.0),
            'vlm_text_ned': result.get('text_ned', 0.0),
            # Style score 使用 vlm_image_quality
            'vlm_style': result.get('image_quality', 0.0),
            'vlm_faithfulness': result.get('faithfulness', 0.0),
            'vlm_overall': result.get('overall', 0.0),
        }


class FluxKleinGenerator:
    """FluxKlein 生成器"""
    def __init__(self, device="cuda:0"):
        sys.path.insert(0, os.path.join(BASELINES_DIR, 'fluxklein'))
        from inference_fluxklein import FluxKleinGenerator
        self.gen = FluxKleinGenerator(
            model_path=FLUX_KLEIN_PATH,
            device=device,
            enable_cpu_offload=False
        )

    def generate(self, prompt, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.gen.generate(
            prompt=prompt, image=None, output_path=output_path,
            seed=42, num_inference_steps=50, guidance_scale=4.0,
            height=1024, width=1024
        )


class QwenImageGenerator:
    """QwenImage 生成器"""
    def __init__(self, device="cuda:0"):
        from diffusers import DiffusionPipeline
        dtype = torch.bfloat16
        self.pipe = DiffusionPipeline.from_pretrained(QWEN_IMAGE_PATH, torch_dtype=dtype).to(device)
        self.device = device

    def generate(self, prompt, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result = self.pipe(
            prompt=prompt + ", Ultra HD, 4K, cinematic composition.",
            width=1024, height=1024, num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device=self.device).manual_seed(42)
        )
        result.images[0].save(output_path)


def run_generation(model_name, dataset, output_subdir):
    """运行生成"""
    output_dir = os.path.join(OUTPUT_DIR, model_name, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[{model_name}] 初始化...")
    if model_name == 'fluxklein':
        gen = FluxKleinGenerator(device="cuda:0")
    else:
        gen = QwenImageGenerator(device="cuda:0")
    
    print(f"[{model_name}] 生成 {len(dataset)} 张图像...")
    for item in dataset:
        item_id, prompt = item['id'], item['prompt']
        output_path = os.path.join(output_dir, f"result_{item_id}.png")
        
        if os.path.exists(output_path):
            print(f"  跳过 [{item_id}]")
            continue
        
        print(f"  生成 [{item_id}]: {prompt[:50]}...")
        try:
            gen.generate(prompt, output_path)
        except Exception as e:
            print(f"    失败: {e}")
    
    return output_dir


def run_evaluation(model_name, dataset, results_dir, output_jsonl):
    """运行评测"""
    print(f"\n[{model_name}] 初始化 VLM...")
    evaluator = VLMEvaluator()
    
    results = []
    print(f"[{model_name}] 评测中...")
    
    for item in dataset:
        item_id, prompt = item['id'], item['prompt']
        image_path = os.path.join(results_dir, f"result_{item_id}.png")
        
        if not os.path.exists(image_path):
            continue
        
        gt_text = extract_text_from_prompt(prompt)
        print(f"  [{item_id}] GT: '{gt_text[:30]}...' ", end="", flush=True)
        
        try:
            image = Image.open(image_path).convert('RGB')
            eval_result = evaluator.evaluate(image, prompt)
            
            result = {
                'id': item_id,
                'prompt': prompt,
                'gt_text': gt_text,
                'pred_text': eval_result['pred_text'],
                'ocr_acc': eval_result['ocr_acc'],
                'ocr_ned': eval_result['ocr_ned'],
                # Style score 使用 vlm_image_quality
                'style_score': eval_result['vlm_style'],
                'vlm_faithfulness': eval_result['vlm_faithfulness'],
                'vlm_overall': eval_result['vlm_overall'],
            }
            results.append(result)
            print(f"OCR={eval_result['ocr_acc']:.2f} Style={eval_result['vlm_style']:.2f}")
            
        except Exception as e:
            print(f"失败: {e}")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # 计算摘要（使用 style_score 即 vlm_image_quality）
    if results:
        summary = {
            'model': model_name,
            'total': len(results),
            'ocr_acc_mean': round(sum(r['ocr_acc'] for r in results) / len(results), 4),
            'ocr_ned_mean': round(sum(r['ocr_ned'] for r in results) / len(results), 4),
            'style_score_mean': round(sum(r['style_score'] for r in results) / len(results), 4),
        }
        summary_path = output_jsonl.replace('.jsonl', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[{model_name}] 完成!")
        print(f"  OCR Acc: {summary['ocr_acc_mean']:.4f}")
        print(f"  OCR NED: {summary['ocr_ned_mean']:.4f}")
        print(f"  Style Score: {summary['style_score_mean']:.4f}")


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def run_benchmark(model_name, jsonl_path, name):
    """运行单个 benchmark"""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Dataset: {name}")
    print(f"{'='*60}")
    
    dataset = load_jsonl(jsonl_path)
    print(f"数据: {len(dataset)} 条")
    
    results_dir = run_generation(model_name, dataset, name)
    output_jsonl = os.path.join(OUTPUT_DIR, model_name, name, "eval_results.jsonl")
    run_evaluation(model_name, dataset, results_dir, output_jsonl)


def main():
    """主函数：准备数据并运行所有 benchmark"""
    
    # 准备子集
    print("="*60)
    print("准备数据子集...")
    print("="*60)
    
    cvtg_path = prepare_cvtg2k_subset()
    longtext_path = prepare_longtext_subset()
    
    benchmarks = [
        ('DrawTextExt', os.path.join(BENCHMARK_DIR, 'DrawTextExt.jsonl')),
        ('AnyText', os.path.join(BENCHMARK_DIR, 'AnyText.jsonl')),
        ('CVTG2K', cvtg_path),
        ('LongText', longtext_path),
    ]
    models = ['fluxklein', 'qwenimage']
    
    print("\n" + "="*60)
    print("最终版 Benchmark 脚本")
    print("- Style Score 使用 vlm_image_quality")
    print("- 支持 4 个数据集（各100条）")
    print("="*60)
    
    for model_name in models:
        for name, path in benchmarks:
            if os.path.exists(path):
                run_benchmark(model_name, path, name)
            else:
                print(f"跳过: {path}")
    
    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)


if __name__ == "__main__":
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    main()
