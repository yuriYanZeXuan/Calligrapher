import os
import sys
import argparse
import json
import torch
import subprocess
from PIL import Image
from tqdm import tqdm
import glob

# Add paths for baselines to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'textflux'))
sys.path.append(os.path.join(BASE_DIR, 'TextCrafter/TextCrafter_Flux'))
sys.path.append(os.path.join(BASE_DIR, 'anytext'))
sys.path.append(os.path.join(BASE_DIR, 'qwenedit'))
sys.path.append(os.path.join(BASE_DIR, 'fluxfill'))

class ModelWrapper:
    def __init__(self, device="cuda"):
        self.device = device

    def generate(self, prompt, **kwargs):
        raise NotImplementedError

class AnyTextWrapper(ModelWrapper):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from inference_anytext import AnyTextInpainter
        # Assuming model_dir is at a standard location or passed via env/args
        # For now, hardcoding or using a default. 
        # The user's environment seems to have weights in /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/
        # Let's try to find AnyText weights there or use default.
        # inference_anytext.py defaults to 'models'.
        self.model_dir = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/AnyText" # Educated guess based on other paths
        if not os.path.exists(self.model_dir):
             self.model_dir = "models" # Fallback
             
        self.inpainter = AnyTextInpainter(model_dir=self.model_dir, use_fp16=True)

    def generate(self, prompt, output_path, **kwargs):
        # AnyText needs text_prompt separately
        text_list = kwargs.get('text', [])
        if not text_list and 'sentence_list' in kwargs:
             text_list = kwargs['sentence_list']
             
        # Format text_prompt: "text1" "text2"
        if isinstance(text_list, list):
            text_prompt = " ".join([f'"{t}"' for t in text_list])
        else:
            text_prompt = f'"{text_list}"' if text_list else ""
            
        # If text_prompt is empty, AnyText might behave unexpectedly for T2I if it expects text.
        # But we'll try.
        
        # T2I Generation mode (no source/mask)
        # We use the 'generate' method of AnyTextInpainter
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # The generate method saves images to output_dir. 
        # We need to intercept or rename.
        # AnyText returns 'results' which is a list of images (numpy arrays usually, let's check inference_anytext.py)
        # inference_anytext.py: results, ... = self.model(...)
        # save_images(results, output_dir)
        # It returns 'results'.
        
        results = self.inpainter.generate(
            img_prompt=prompt,
            text_prompt=text_prompt,
            draw_pos=None, # Auto-generate position
            seed=42,
            img_count=1,
            output_dir=os.path.dirname(output_path) # It saves here
        )
        
        # AnyText saves files with its own naming convention.
        # We need to save the result to output_path.
        # results is a list of images.
        if results:
            img = results[0]
            # img is likely numpy array (H, W, 3)
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img.save(output_path)

class QwenEditWrapper(ModelWrapper):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from inference_qwenedit import QwenEditGenerator
        self.generator = QwenEditGenerator(
            model_path="Qwen/Qwen-Image-Edit", # Default or specific path
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        # Create white image for editing
        width, height = 1024, 1024
        image = Image.new("RGB", (width, height), "white")
        
        self.generator.generate(
            prompt=prompt,
            image=image,
            output_path=output_path,
            seed=42
        )

class FluxFillWrapper(ModelWrapper):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from inference_fluxfill import FluxFillGenerator
        self.generator = FluxFillGenerator(
            model_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux_fill",
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        # Create white image and mask
        width, height = 1024, 1024
        image = Image.new("RGB", (width, height), "white")
        mask_image = Image.new("RGB", (width, height), "white") # Full mask
        
        self.generator.generate(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            output_path=output_path,
            seed=42
        )

class TextFluxWrapper(ModelWrapper):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from inference_textflux import TextFluxGenerator
        # Using default paths from the original script, or we could make them configurable
        self.generator = TextFluxGenerator(
            pipeline_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux_fill",
            transformer_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux_fill/transformer",
            lora_path="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/textflux-lora-beta",
            device=device
        )

    def generate(self, prompt, output_path, **kwargs):
        # Create white image and mask for editing model
        width, height = 1024, 1024
        image = Image.new("RGB", (width, height), "white")
        mask_image = Image.new("RGB", (width, height), "white")
        
        self.generator.generate(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            output_path=output_path,
            seed=42,
            num_inference_steps=50,
            guidance_scale=30.0
        )

class TextCrafterFluxWrapper(ModelWrapper):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from diffusers import FluxPipeline
        from textcrafter_pipeline_flux import textcrafter_FluxPipeline
        
        print("Initializing TextCrafter Flux pipeline...")
        ldm_flux = FluxPipeline.from_pretrained(
            "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
        self.pipe = textcrafter_FluxPipeline.from_pipeline(ldm_flux)
        self.ldm_flux = ldm_flux # Needed for pre_generation
        print("TextCrafter Flux pipeline initialized.")

    def generate(self, prompt, output_path, **kwargs):
        # TextCrafter specific arguments
        carrier_list = kwargs.get('carrier_list', [])
        sentence_list = kwargs.get('sentence_list', [])
        
        # If carrier_list is empty (e.g. LongText-Bench), we might need a fallback or it might fail.
        # For now, we pass what we have.
        
        # Copy the core logic of inference from eval.py to here to avoid global variable issues
        from pre_generation import pre_generation
        from rectangles import generate_rectangles_gurobi
        
        height = 512
        width = 512
        seed = 0
        min_area = kwargs.get('min_area', 0.65) # Default for area=2
        
        # If carrier_list is missing, TextCrafter might not work as intended.
        # Assuming CVTG data provides it.
        
        max_pixels = pre_generation(
            ldm_flux=self.ldm_flux,
            NUM_DIFFUSION_STEPS=8,
            height=height,
            width=width,
            seed=seed,
            prompt=prompt,
            carrier_list=carrier_list
        )
        
        rectangles = generate_rectangles_gurobi(points=max_pixels, min_area=min_area)
        
        insulation_m_offset_list = [r['m_offset'] for r in rectangles]
        insulation_n_offset_list = [r['n_offset'] for r in rectangles]
        insulation_m_scale_list = [r['m_scale'] for r in rectangles]
        insulation_n_scale_list = [r['n_scale'] for r in rectangles]
        
        image = self.pipe(
            sentence_list=sentence_list,
            insulation_m_offset_list=insulation_m_offset_list,
            insulation_n_offset_list=insulation_n_offset_list,
            insulation_m_scale_list=insulation_m_scale_list,
            insulation_n_scale_list=insulation_n_scale_list,
            insulation_steps=3,
            carrier_list=carrier_list,
            cross_replace_steps=1.0,
            seed=seed,
            addition=0.4,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=30,
            guidance_scale=3.5,
        ).images[0]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

def load_dataset(benchmark, base_eval_dir):
    data = []
    if benchmark == 'CVTG-2K':
        # Load all json files from CVTG and CVTG-Style
        subsets = ['CVTG', 'CVTG-Style']
        for subset in subsets:
            subset_dir = os.path.join(base_eval_dir, 'CVTG-2K', subset)
            json_files = glob.glob(os.path.join(subset_dir, '*.json'))
            for json_file in json_files:
                if 'combined' in json_file: continue # Skip combined files if they exist
                
                # Extract area from filename (e.g., 2.json -> 2)
                try:
                    area = int(os.path.basename(json_file).split('.')[0])
                except:
                    area = 2 # Default
                    
                with open(json_file, 'r') as f:
                    content = json.load(f)
                    items = content.get('data_list', [])
                    for item in items:
                        item['benchmark_subset'] = subset
                        item['area'] = area
                        item['id'] = f"{subset}_{area}_{item['index']}"
                        data.append(item)
                        
    elif benchmark == 'LongText-Bench':
        jsonl_path = os.path.join(base_eval_dir, 'LongText-Bench', 'text_prompts.jsonl')
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['id'] = f"longtext_{item['prompt_id']}"
                # Normalize keys
                item['carrier_list'] = [] # Not available
                item['sentence_list'] = [item['prompt']] # Use prompt as sentence list?
                data.append(item)
                
    return data

def run_evaluation(generated_dir, benchmark_dir, metrics):
    print(f"\nStarting evaluation for {generated_dir}...")
    eval_script = os.path.join(os.path.dirname(BASE_DIR), 'eval', 'run_evaluation.py')
    
    cmd = [
        sys.executable, eval_script,
        '--generated_dir', generated_dir,
        '--benchmark_dir', benchmark_dir,
        '--metrics'
    ] + metrics
    
    subprocess.run(cmd, check=False)

def main():
    parser = argparse.ArgumentParser(description="Unified Generation and Evaluation Script")
    parser.add_argument("--model", type=str, required=True, 
                        choices=['textflux', 'textcrafter_flux', 'anytext', 'qwenedit', 'fluxfill'], 
                        help="Model to run")
    parser.add_argument("--benchmark", type=str, required=True, choices=['CVTG-2K', 'LongText-Bench'], help="Benchmark to run")
    parser.add_argument("--debug", action='store_true', default=True, help="Run in debug mode (only 5 samples)")
    parser.add_argument("--no-debug", action='store_false', dest='debug', help="Disable debug mode")
    parser.add_argument("--resume", action='store_true', default=True, help="Resume from existing results")
    parser.add_argument("--no-resume", action='store_false', dest='resume', help="Overwrite existing results")
    
    args = parser.parse_args()
    
    # Setup paths
    eval_dir = os.path.join(os.path.dirname(BASE_DIR), 'eval')
    output_base_dir = os.path.join(BASE_DIR, 'results', args.model, args.benchmark)
    
    # Load Model
    if args.model == 'textflux':
        model = TextFluxWrapper()
    elif args.model == 'textcrafter_flux':
        model = TextCrafterFluxWrapper()
    elif args.model == 'anytext':
        model = AnyTextWrapper()
    elif args.model == 'qwenedit':
        model = QwenEditWrapper()
    elif args.model == 'fluxfill':
        model = FluxFillWrapper()
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Load Data
    print(f"Loading dataset {args.benchmark}...")
    dataset = load_dataset(args.benchmark, eval_dir)
    print(f"Loaded {len(dataset)} items.")
    
    if args.debug:
        print("Debug mode enabled: Processing only first 5 items.")
        dataset = dataset[:5]
        
    # Generation Loop
    print(f"Starting generation... Output dir: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    for item in tqdm(dataset):
        # Construct output filename
        # Format: result_{id}.png to match run_evaluation.py expectation
        filename = f"result_{item['id']}.png"
        output_path = os.path.join(output_base_dir, filename)
        
        if args.resume and os.path.exists(output_path):
            continue
            
        try:
            # Prepare kwargs for specific models
            kwargs = {
                'carrier_list': item.get('carrier_list', []),
                'sentence_list': item.get('sentence_list', []),
            }
            if 'area' in item:
                # Map area to min_area for TextCrafter if needed
                min_area_default = (0.65, 0.3, 0.2, 0.15, 0.12)
                area_idx = item['area'] - 1
                if 0 <= area_idx < len(min_area_default):
                    kwargs['min_area'] = min_area_default[area_idx]
            
            model.generate(item['prompt'], output_path, **kwargs)
            
            # Save metadata for evaluation (run_evaluation.py needs to know the prompt and ref_id)
            # run_evaluation.py loads metadata from the filename or a separate file?
            # Checking run_evaluation.py -> load_images_for_evaluation:
            # It tries to find the corresponding ground truth in benchmark_dir.
            # It seems it expects the filename to match something or contain IDs.
            # Let's save a metadata json alongside just in case, or ensure filename mapping is correct.
            # Actually run_evaluation.py parses filename to get 'ref_id'.
            # "parse_generated_filename" in utils.py likely handles this.
            
        except Exception as e:
            print(f"Error generating {item['id']}: {e}")
            import traceback
            traceback.print_exc()
            
    # Evaluation
    # Note: run_evaluation.py expects benchmark_dir to be the root of the dataset
    # For CVTG-2K, it seems to expect the folder structure.
    # We pass the root eval dir or the specific benchmark dir?
    # args.benchmark_dir in run_evaluation defaults to Calligrapher_bench_testing.
    # We should pass the correct path.
    
    benchmark_path = os.path.join(eval_dir, args.benchmark)
    
    # Metrics selection
    metrics = ['ocr'] # Default metrics for T2I benchmarks
    
    evaluate_results(output_base_dir, dataset, metrics)

def evaluate_results(output_dir, dataset, metrics=['ocr']):
    print(f"\nStarting evaluation for {output_dir}...")
    
    # Add root dir to sys.path so 'from eval.xxx' works
    root_dir = os.path.dirname(BASE_DIR)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)  # Insert at front to prioritize eval package
        
    from eval.eval_ocr import OCREvaluator
    import pandas as pd
    
    ocr_evaluator = OCREvaluator() if 'ocr' in metrics else None
    
    results = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        img_path = os.path.join(output_dir, f"result_{item['id']}.png")
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
            
        row = {'id': item['id'], 'prompt': item['prompt']}
        
        # Get ground truth text
        # CVTG and LongText-Bench have 'text' field (list of strings)
        # If not present, try to infer from prompt or sentence_list
        ground_truth_text = item.get('text', [])
        if not ground_truth_text and 'sentence_list' in item:
             ground_truth_text = item['sentence_list']
        
        # If ground_truth_text is a list, join it or pass as is?
        # eval_ocr.py usually expects a string or list.
        # Let's assume it handles list or we join it.
        # Checking eval_ocr.py would be good, but let's assume standard behavior or join.
        # Usually OCR eval compares detected text with target text.
        
        if ocr_evaluator:
            # Prepare GT
            ground_truth_text = item.get('text', [])
            if not ground_truth_text and 'sentence_list' in item:
                 ground_truth_text = item['sentence_list']
            
            if isinstance(ground_truth_text, list):
                ground_truth_text = " ".join(ground_truth_text)
            elif not isinstance(ground_truth_text, str):
                ground_truth_text = str(ground_truth_text) # Fallback
            
            row['ground_truth'] = ground_truth_text
            
            try:
                acc = ocr_evaluator.calculate_ocr_accuracy(image, ground_truth_text, mask=None)
                row['ocr_accuracy'] = acc
            except Exception as e:
                print(f"OCR Error for {item['id']}: {e}")
                row['ocr_accuracy'] = 0.0
                
        results.append(row)
        
    # Save CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Print summary
        if 'ocr_accuracy' in df.columns:
            print(f"Mean OCR Accuracy: {df['ocr_accuracy'].mean():.4f}")
    else:
        print("No results to evaluate.")

if __name__ == "__main__":
    main()
