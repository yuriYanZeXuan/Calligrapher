#!/usr/bin/env python3
"""
生成并行 + 评测串行版一键启动脚本
- 生成阶段：8 GPU 并行
- 评测阶段：1 GPU 串行（使用 VLM API，无需多卡）
- 支持 DrawTextExt, AnyText, CVTG-2K, LongText-Bench
"""

import os
import sys
import json
import re
import random
import math
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
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
    """从 prompt 中提取引号内文本"""
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
    
    sampled = random.sample(all_data, min(100, len(all_data)))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"CVTG-2K 子集已创建: {len(sampled)} 条")
    return output_path


def prepare_longtext_subset():
    """准备 LongText-Bench 子集（100条）"""
    output_path = os.path.join(BENCHMARK_DIR, 'LongText_subset.jsonl')
    if os.path.exists(output_path):
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
    
    sampled = random.sample(all_data, min(100, len(all_data)))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"LongText-Bench 子集已创建: {len(sampled)} 条")
    return output_path


class FluxKleinGenerator:
    """FluxKlein 生成器"""
    def __init__(self, device="cuda:0"):
        import time
        time.sleep(int(device.split(':')[1]) * 2)
        
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
        import time
        time.sleep(int(device.split(':')[1]) * 2)
        
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


def generation_worker(rank, world_size, model_name, dataset, output_dir):
    """生成阶段的工作进程"""
    import torch
    def no_op_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda x: x
        return model
    torch.compile = no_op_compile
    
    device = f"cuda:{rank}"
    
    # 数据分片
    total = len(dataset)
    per_gpu = math.ceil(total / world_size)
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, total)
    my_data = dataset[start_idx:end_idx]
    
    if not my_data:
        print(f"[GPU {rank}] 无数据分配")
        return
    
    print(f"[GPU {rank}] 初始化 {model_name}...")
    if model_name == 'fluxklein':
        gen = FluxKleinGenerator(device=device)
    else:
        gen = QwenImageGenerator(device=device)
    
    print(f"[GPU {rank}] 生成 {len(my_data)} 张图像...")
    for item in tqdm(my_data, desc=f"GPU {rank}", position=rank):
        item_id = item['id']
        prompt = item['prompt']
        output_path = os.path.join(output_dir, f"result_{item_id}.png")
        
        if os.path.exists(output_path):
            continue
        
        try:
            gen.generate(prompt, output_path)
        except Exception as e:
            print(f"[GPU {rank}] 生成失败 [{item_id}]: {e}")


def run_generation_parallel(model_name, dataset, output_subdir, num_gpus=8):
    """并行生成图像（多GPU）"""
    output_dir = os.path.join(OUTPUT_DIR, model_name, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[{model_name}] 启动 {num_gpus} GPU 并行生成...")
    
    mp.spawn(
        generation_worker,
        args=(num_gpus, model_name, dataset, output_dir),
        nprocs=num_gpus,
        join=True
    )
    
    print(f"[{model_name}] 生成完成")
    return output_dir


def run_evaluation_single(model_name, dataset, results_dir, output_jsonl):
    """串行评测（单GPU，使用 VLM API）"""
    print(f"\n[{model_name}] 串行评测（单GPU，VLM API）...")
    
    # 只在 cuda:0 上初始化
    sys.path.insert(0, os.path.join(ROOT_DIR, "eval", "core"))
    from metrics import VLMMetrics
    
    # VLM API 模式，device 参数不实际使用 GPU 计算
    metrics = VLMMetrics(model_path="ApiCall", device="cpu")
    
    results = []
    print(f"[{model_name}] 评测 {len(dataset)} 个样本...")
    
    for item in tqdm(dataset, desc="Evaluating"):
        item_id = item['id']
        prompt = item['prompt']
        image_path = os.path.join(results_dir, f"result_{item_id}.png")
        
        if not os.path.exists(image_path):
            continue
        
        gt_text = extract_text_from_prompt(prompt)
        
        try:
            image = Image.open(image_path).convert('RGB')
            result = metrics.evaluate_text_rendering(image, prompt)
            
            results.append({
                'id': item_id,
                'prompt': prompt,
                'gt_text': gt_text,
                'pred_text': result.get('recognized_text', ''),
                'ocr_acc': result.get('text_accuracy', 0.0),
                'ocr_ned': result.get('text_ned', 0.0),
                'style_score': result.get('image_quality', 0.0),
                'vlm_faithfulness': result.get('faithfulness', 0.0),
                'vlm_overall': result.get('overall', 0.0),
            })
        except Exception as e:
            print(f"  评测失败 [{item_id}]: {e}")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # 计算摘要
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
        
        print(f"\n[{model_name}] 评测完成!")
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


def run_benchmark(model_name, jsonl_path, name, num_gpus=8):
    """运行单个 benchmark"""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Dataset: {name}")
    print(f"生成: {num_gpus} GPU 并行 | 评测: 1 GPU 串行")
    print(f"{'='*60}")
    
    dataset = load_jsonl(jsonl_path)
    print(f"数据: {len(dataset)} 条")
    
    # 并行生成
    results_dir = run_generation_parallel(model_name, dataset, name, num_gpus)
    
    # 串行评测（单GPU）
    output_jsonl = os.path.join(OUTPUT_DIR, model_name, name, "eval_results.jsonl")
    run_evaluation_single(model_name, dataset, results_dir, output_jsonl)


def main():
    """主函数"""
    # 准备子集
    print("="*60)
    print("准备数据子集...")
    print("="*60)
    
    prepare_cvtg2k_subset()
    prepare_longtext_subset()
    
    benchmarks = [
        ('DrawTextExt', os.path.join(BENCHMARK_DIR, 'DrawTextExt.jsonl')),
        ('AnyText', os.path.join(BENCHMARK_DIR, 'AnyText.jsonl')),
        ('CVTG2K', os.path.join(BENCHMARK_DIR, 'CVTG2K_subset.jsonl')),
        ('LongText', os.path.join(BENCHMARK_DIR, 'LongText_subset.jsonl')),
    ]
    models = ['fluxklein', 'qwenimage']
    
    # GPU 数量（可修改）
    num_gpus = 8
    
    print("\n" + "="*60)
    print(f"生成并行({num_gpus}卡) + 评测串行(1卡) Benchmark 脚本")
    print("="*60)
    
    for model_name in models:
        for name, path in benchmarks:
            if os.path.exists(path):
                run_benchmark(model_name, path, name, num_gpus)
            else:
                print(f"跳过: {path}")
    
    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)


if __name__ == "__main__":
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    
    mp.set_start_method('spawn', force=True)
    
    main()
