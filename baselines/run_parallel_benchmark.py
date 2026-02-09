#!/usr/bin/env python3
"""Unified benchmark script for text-to-image generation models (Parallel Version)."""

import os
import sys
import argparse
import json
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
import glob
import pandas as pd
import math

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.extend([
    ROOT_DIR,
    BASE_DIR,
    os.path.join(BASE_DIR, 'textflux'),
    os.path.join(BASE_DIR, 'TextCrafter/TextCrafter_Flux'),
    os.path.join(BASE_DIR, 'anytext'),
    os.path.join(BASE_DIR, 'qwenedit'),
    os.path.join(BASE_DIR, 'fluxfill'),
    os.path.join(BASE_DIR, 'fluxdev'),
    os.path.join(BASE_DIR, 'fluxklein'),
    os.path.join(BASE_DIR, 'glm_image'),
    os.path.join(BASE_DIR, 'z_image'),
    os.path.join(BASE_DIR, 'qwenimage'),
    os.path.join(BASE_DIR, 'nanobanana'),
])

# Model paths configuration
MODEL_PATHS = {
    'anytext': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/AnyText',
    'qwenedit': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/QwenEdit2509',
    'fluxfill': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux_fill',
    'fluxdev': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/FLUX.1-dev',
    'fluxklein': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein',
    'textflux': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux_fill',
    'textcrafter_flux': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/FLUX.1-dev',
    'glm_image': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/glm_image',
    'z_image': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image',
    'qwenimage': '/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen-image-2512',
}

# --- Model Wrappers ---

class ModelWrapper:
    """Base wrapper for all models."""
    def __init__(self, device="cuda", model_path=None):
        self.device = device
        self.model_path = model_path

    def generate(self, prompt, output_path, **kwargs):
        raise NotImplementedError

class AnyTextWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_anytext import AnyTextInpainter
        import numpy as np
        self.np = np
        
        path = self.model_path or MODEL_PATHS['anytext']
        path = path if os.path.exists(path) else "models"
        self.inpainter = AnyTextInpainter(model_dir=path, use_fp16=True, device=device)

    def generate(self, prompt, output_path, **kwargs):
        text_list = kwargs.get('text') or kwargs.get('sentence_list') or []
        text_prompt = " ".join([f'"{t}"' for t in text_list]) if isinstance(text_list, list) else f'"{text_list}"'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = self.inpainter.generate(
            img_prompt=prompt, text_prompt=text_prompt, draw_pos=None,
            seed=42, img_count=1, output_dir=os.path.dirname(output_path)
        )
        
        if results:
            img = results[0] if isinstance(results[0], Image.Image) else Image.fromarray(results[0])
            img.save(output_path)

class QwenEditWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_qwenedit import QwenEditGenerator
        self.generator = QwenEditGenerator(
            model_path=self.model_path or MODEL_PATHS['qwenedit'],
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        self.generator.generate(
            prompt=prompt,
            image=Image.new("RGB", (1024, 1024), "white"),
            output_path=output_path,
            seed=42
        )

class FluxFillWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_fluxfill import FluxFillGenerator
        self.generator = FluxFillGenerator(
            model_path=self.model_path or MODEL_PATHS['fluxfill'],
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        white_img = Image.new("RGB", (1024, 1024), "white")
        self.generator.generate(
            prompt=prompt, image=white_img, mask_image=white_img,
            output_path=output_path, seed=42
        )

class TextFluxWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_textflux import TextFluxGenerator
        
        base_path = self.model_path or MODEL_PATHS['textflux']
        self.generator = TextFluxGenerator(
            pipeline_path=base_path,
            transformer_path=os.path.join(base_path, "transformer"),
            lora_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/textflux-lora-beta",
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        white_img = Image.new("RGB", (1024, 1024), "white")
        self.generator.generate(
            image=white_img, mask_image=white_img, prompt=prompt,
            output_path=output_path, seed=42,
            num_inference_steps=50, guidance_scale=30.0
        )

class FluxDevWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_fluxdev import FluxDevGenerator
        self.generator = FluxDevGenerator(
            model_path=self.model_path or MODEL_PATHS['fluxdev'],
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        self.generator.generate(
            prompt=prompt,
            output_path=output_path,
            seed=42,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=1024,
            width=1024
        )

class FluxKleinWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_fluxklein import FluxKleinGenerator
        self.generator = FluxKleinGenerator(
            model_path=self.model_path or MODEL_PATHS['fluxklein'],
            device=device,
            enable_cpu_offload=False
        )

    def generate(self, prompt, output_path, **kwargs):
        self.generator.generate(
            prompt=prompt,
            image=None,
            output_path=output_path,
            seed=42,
            num_inference_steps=50,
            guidance_scale=4.0,
            height=1024,
            width=1024
        )

class TextCrafterFluxWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from diffusers import FluxPipeline
        from textcrafter_pipeline_flux import textcrafter_FluxPipeline
        
        print(f"Initializing TextCrafter Flux on {device}...")
        self.ldm_flux = FluxPipeline.from_pretrained(
            self.model_path or MODEL_PATHS['textcrafter_flux'],
            torch_dtype=torch.bfloat16
        ).to(device)
        self.pipe = textcrafter_FluxPipeline.from_pipeline(self.ldm_flux)
        print(f"TextCrafter Flux initialized on {device}.")

    def generate(self, prompt, output_path, **kwargs):
        from pre_generation import pre_generation
        from rectangles import generate_rectangles_gurobi
        
        carrier_list = kwargs.get('carrier_list', [])
        sentence_list = kwargs.get('sentence_list', [])
        min_area = kwargs.get('min_area', 0.65)
        height, width, seed = 512, 512, 0
        
        max_pixels = pre_generation(
            ldm_flux=self.ldm_flux, NUM_DIFFUSION_STEPS=8,
            height=height, width=width, seed=seed,
            prompt=prompt, carrier_list=carrier_list
        )
        
        rectangles = generate_rectangles_gurobi(points=max_pixels, min_area=min_area)
        ins_params = {k: [r[k.split('_')[-2]] for r in rectangles] 
                     for k in ['insulation_m_offset_list', 'insulation_n_offset_list',
                              'insulation_m_scale_list', 'insulation_n_scale_list']}
        
        image = self.pipe(
            sentence_list=sentence_list, carrier_list=carrier_list,
            prompt=prompt, height=height, width=width,
            insulation_steps=3, cross_replace_steps=1.0,
            seed=seed, addition=0.4, num_inference_steps=30,
            guidance_scale=3.5, **ins_params
        ).images[0]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

class GlmImageWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_glm_image import GlmImageGenerator
        self.generator = GlmImageGenerator(
            model_path=self.model_path or MODEL_PATHS['glm_image'],
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        self.generator.generate(
            prompt=prompt,
            output_path=output_path,
            seed=42,
            num_inference_steps=50,
            guidance_scale=1.5,
            height=1024,
            width=1024
        )

class ZImageWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        import inference_z_image
        self.generator = inference_z_image.ZImageGenerator(
            model_path=self.model_path or MODEL_PATHS['z_image'],
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        self.generator.generate(
            prompt=prompt,
            output_path=output_path,
            seed=42,
            num_inference_steps=9,
            guidance_scale=0.0,
            height=1024,
            width=1024
        )

class QwenImageWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from diffusers import DiffusionPipeline
        import torch

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        path = self.model_path or MODEL_PATHS['qwenimage']
        self.pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=dtype).to(device)
        self.device = device

    def generate(self, prompt, output_path, **kwargs):
        result = self.pipe(
            prompt=prompt + ", Ultra HD, 4K, cinematic composition.",
            negative_prompt="",
            width=1024,
            height=1024,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device=self.device).manual_seed(42)
        )
        result.images[0].save(output_path)

class NanoBananaWrapper(ModelWrapper):
    def __init__(self, device="cuda", model_path=None):
        super().__init__(device, model_path)
        from inference_nanobanana import NanoBananaGenerator
        # API-based model, device parameter is not used but kept for compatibility
        self.generator = NanoBananaGenerator(device=device)

    def generate(self, prompt, output_path, **kwargs):
        self.generator.generate(
            prompt=prompt,
            output_path=output_path,
            seed=42,
            temperature=1.0,
            aspect_ratio="1:1",
            image_size="1K"
        )

# Model registry
MODELS = {
    'textflux': TextFluxWrapper,
    'textcrafter_flux': TextCrafterFluxWrapper,
    'anytext': AnyTextWrapper,
    'qwenedit': QwenEditWrapper,
    'fluxfill': FluxFillWrapper,
    'fluxdev': FluxDevWrapper,
    'fluxklein': FluxKleinWrapper,
    'glm_image': GlmImageWrapper,
    'z_image': ZImageWrapper,
    'qwenimage': QwenImageWrapper,
    'nanobanana': NanoBananaWrapper
}

# --- Data Loading ---

def load_dataset(benchmark, base_eval_dir):
    """Load benchmark dataset."""
    data = []
    
    if benchmark == 'CVTG-2K':
        # Only load *_combined.json (format: {"0": "prompt", "1": "prompt", ...})
        for subset in ['CVTG', 'CVTG-Style']:
            subset_dir = os.path.join(base_eval_dir, 'CVTG-2K', subset)
            for json_file in glob.glob(os.path.join(subset_dir, '*_combined.json')):
                area = int(os.path.basename(json_file).split('_')[0])
                with open(json_file, 'r') as f:
                    combined = json.load(f)
                for idx_str, prompt in combined.items():
                    index = int(idx_str)
                    data.append({
                        'prompt': prompt,
                        'index': index,
                        'benchmark_subset': subset,
                        'area': area,
                        'id': f"{subset}_{area}_{index}",
                        'carrier_list': [],
                        'sentence_list': [],
                        'text': [],
                    })
    elif benchmark == 'LongText-Bench':
        # Read as text_prompts.jsonl format: category, length, prompt, text, text_length, prompt_id (no carrier_list/sentence_list)
        longtext_dir = os.path.join(base_eval_dir, 'LongText-Bench')
        for jsonl_file in sorted(glob.glob(os.path.join(longtext_dir, '*.jsonl'))):
            print(f"Loading {os.path.basename(jsonl_file)}...")
            with open(jsonl_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    lang_prefix = "zh" if "zh" in jsonl_file else "en"
                    prompt_id = item.get('prompt_id', len(data))
                    data.append({
                        'prompt': item['prompt'],
                        'text': item.get('text', []),
                        'prompt_id': prompt_id,
                        'category': item.get('category', ''),
                        'length': item.get('length', ''),
                        'text_length': item.get('text_length', 0),
                        'id': f"longtext_{lang_prefix}_{prompt_id}",
                        'carrier_list': [],
                        'sentence_list': [],
                    })
    elif benchmark in ('OneIG-Bench', 'OneIG-Bench-ZH'):
        # OneIG-Bench JSON format (list of dicts). For ZH use prompt_cn; for EN use prompt_en.
        oneig_dir = os.path.join(base_eval_dir, 'OneIG-Bench')
        json_name = "OneIG-Bench-ZH.json" if benchmark == "OneIG-Bench-ZH" else "OneIG-Bench.json"
        json_path = os.path.join(oneig_dir, json_name)
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        lang_prefix = "zh" if benchmark == "OneIG-Bench-ZH" else "en"
        prompt_key = "prompt_cn" if lang_prefix == "zh" else "prompt_en"

        # Deterministic ordering
        items = sorted(items, key=lambda x: str(x.get("id", "")))
        for item in items:
            sample_id = str(item["id"])
            data.append({
                "prompt": str(item[prompt_key]),
                "prompt_id": sample_id,
                "index": int(sample_id),
                "category": str(item.get("category", "")),
                "class": str(item.get("class", "")),
                "type": str(item.get("type", "")),
                "prompt_length": str(item.get("prompt_length", "")),
                "id": f"oneig_{lang_prefix}_{sample_id}",
                # Keep fields for compatibility with model wrappers / evaluator
                "carrier_list": [],
                "sentence_list": [],
                "text": [],
            })
    elif benchmark == 'UnseenWords':
        # UnseenWords benchmark - rare Chinese characters testing
        # JSONL format: category, length, prompt, text, text_length, prompt_id
        import glob
        unseen_dir = os.path.join(base_eval_dir, 'UnseenWords')
        jsonl_files = glob.glob(os.path.join(unseen_dir, '*.jsonl'))
        
        for jsonl_file in sorted(jsonl_files):
            print(f"Loading {os.path.basename(jsonl_file)}...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    prompt_id = item.get('prompt_id', len(data))
                    text_list = item.get('text', [])
                    data.append({
                        'prompt': item['prompt'],
                        'text': text_list,
                        'prompt_id': prompt_id,
                        'category': item.get('category', ''),
                        'length': item.get('length', ''),
                        'text_length': item.get('text_length', 0),
                        'id': f"unseen_{prompt_id}",
                        'carrier_list': [],
                        'sentence_list': text_list if isinstance(text_list, list) else [text_list],
                    })
    return data


# --- Worker Function ---

def worker_fn(rank, world_size, args, dataset, output_dir):
    """Worker: rank r gets dataset[start_idx:end_idx]; prompt for local index i
    is benchmark prompt at global index (start_idx + i)."""
    
    # Stagger initialization to avoid race conditions in file-based caches (e.g. inductor, triton)
    import time
    time.sleep(rank * 2.0)

    # Monkey patch torch.compile to avoid Inductor/Triton initialization race conditions
    # causing "duplicate template name" errors in bitsandbytes -> diffusers imports.
    import torch
    def no_op_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda x: x
        return model
    torch.compile = no_op_compile
    
    total_items = len(dataset)
    items_per_gpu = math.ceil(total_items / world_size)
    start_idx = rank * items_per_gpu
    end_idx = min(start_idx + items_per_gpu, total_items)
    my_dataset = dataset[start_idx:end_idx]
    if not my_dataset:
        return
    device = f"cuda:{rank}"
    # Check if we need to generate anything in this slice
    need_generate_slice = not args.resume or any(
        not os.path.exists(os.path.join(output_dir, f"result_{item['id']}.png"))
        for item in my_dataset
    )

    if not need_generate_slice:
        print(f"[GPU {rank}] All items already generated, skipping.")
        return

    # Initialize model
    model = MODELS[args.model](device=device, model_path=args.model_path)

    # Generate
    
    for item in tqdm(my_dataset, desc=f"GPU {rank}", position=rank):
        output_path = os.path.join(output_dir, f"result_{item['id']}.png")
        if args.resume and os.path.exists(output_path):
            continue
        
        kwargs = {
            'carrier_list': item.get('carrier_list', []),
            'sentence_list': item.get('sentence_list', []),
            'text': item.get('text', [])
        }
        
        if args.benchmark == 'CVTG-2K':
            min_areas = (0.65, 0.3, 0.2, 0.15, 0.12)
            if 'area' in item:
                idx = item['area'] - 1
                if 0 <= idx < len(min_areas):
                    kwargs['min_area'] = min_areas[idx]
        
        model.generate(item['prompt'], output_path, **kwargs)

# --- Evaluation ---

def evaluate_results(output_dir, dataset, metrics=['ocr']):
    """Evaluate generated images."""
    print(f"\nEvaluating {output_dir}...")
    from eval.core.metrics import OCRMetrics

    ocr_evaluator = OCRMetrics() if 'ocr' in metrics else None
    results = []
    for item in tqdm(dataset, desc="Evaluating"):
        img_path = os.path.join(output_dir, f"result_{item['id']}.png")
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        row = {'id': item['id'], 'prompt': item['prompt']}
        if ocr_evaluator:
            gt_text = item.get('text') or item.get('sentence_list') or []
            gt_text = " ".join(gt_text) if isinstance(gt_text, list) else str(gt_text)
            row['ground_truth'] = gt_text
            if gt_text.strip():
                ocr_res = ocr_evaluator.compute_accuracy(
                    image, gt_text, mask=None
                )
                row['ocr_accuracy'] = ocr_res['ocr_acc']
                row['ocr_ned'] = ocr_res['ocr_ned']
            else:
                # For benchmarks without explicit GT text (e.g., OneIG-Bench), skip OCR scoring.
                row['ocr_accuracy'] = None
        results.append(row)
    
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        if 'ocr_accuracy' in df.columns:
            print(f"Mean OCR Accuracy: {df['ocr_accuracy'].mean():.4f}")
    else:
        print("No results to evaluate.")

# --- Main ---

def main():
    # Workaround for torch._inductor "duplicate template name" error in multiprocessing
    # This must be set before any torch/diffusers imports that might trigger compilation
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    parser = argparse.ArgumentParser(description="Unified benchmark script (Parallel)")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                       help="Model to run")
    parser.add_argument("--benchmark", type=str, required=True,
                       choices=['CVTG-2K', 'LongText-Bench', 'OneIG-Bench', 'OneIG-Bench-ZH', 'UnseenWords'], help="Benchmark dataset")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Custom model path (overrides default)")
    parser.add_argument("--debug", action='store_true',
                       help="Debug mode (5 samples only)")
    parser.add_argument("--resume", action='store_true',
                       help="Skip already generated images and evaluate all existing results")
    parser.add_argument("--skip-eval", action='store_true',
                       help="Skip evaluation, only generate images")
    parser.add_argument("--gpus", type=int, default=8,
                       help="Number of GPUs to use")
    parser.add_argument("--output_dir",type=str,default=None,
                       help="Output directory")
    args = parser.parse_args()
    
    # Setup
    eval_dir = os.path.join(ROOT_DIR, 'eval')
    output_dir = args.output_dir or os.path.join(BASE_DIR, 'results', args.model, args.benchmark)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.benchmark}...")
    dataset = load_dataset(args.benchmark, eval_dir)
    if args.debug:
        print("Debug mode: 5 samples only")
        dataset = dataset[:5]
    print(f"Loaded {len(dataset)} items")
    
    
    
    # Check total generation need (optional, but good for skipping spawn if done)
    # But for parallel, let's just spawn and let workers check.
    
    print(f"Launching {args.gpus} processes...")
    mp.set_start_method('spawn', force=True)
    mp.spawn(
        worker_fn,
        args=(args.gpus, args, dataset, output_dir),
        nprocs=args.gpus,
        join=True
    )
    
    print("All workers finished.")

    # Evaluate (Single process)
    if not args.skip_eval:
        evaluate_results(output_dir, dataset, metrics=['ocr'])
    else:
        print("Skipping evaluation (--skip-eval)")

if __name__ == "__main__":
    main()
