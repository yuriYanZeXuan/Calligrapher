import os
import argparse
import subprocess
import json
from tqdm import tqdm

# This script assumes it's being run from the `Calligrapher/eval` directory.
# Adjust paths if necessary.
ANYTEXT_INFERENCE_SCRIPT = "../baselines/anytext/inference_anytext.py"
TEXTDIFFUSER2_INFERENCE_SCRIPT = "../baselines/textdiffuser2/inference_textdiffuser2.py"
CALLIGRAPHER_INFERENCE_SCRIPT = "../infer_calligrapher_self_custom.py" # Assuming this is the script for Calligrapher

BENCHMARK_DIR = "../../dataset/Calligrapher_bench_testing"
SELF_BENCH_FILE = os.path.join(BENCHMARK_DIR, "self_bench.txt")

# --- IMPORTANT: Please configure these paths before running ---
# Path to the directory containing AnyText2 model checkpoints (e.g., anytext_v2.0.ckpt)
ANYTEXT_MODEL_DIR = "../../AnyText2/models" 
# Path to the layout planner model for TextDiffuser2
TEXTDIFFUSER2_LAYOUT_PLANNER_PATH = "/path/to/your/textdiffuser2/layout_planner_model" 
# Path to the main diffusion model checkpoint for TextDiffuser2
TEXTDIFFUSER2_DIFFUSION_MODEL_PATH = "/path/to/your/textdiffuser2/diffusion_model"
# Path to the Calligrapher model checkpoint
CALLIGRAPHER_MODEL_PATH = "/path/to/your/calligrapher/model.pth"
TEXTDIFFUSER2_BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5" # Or a local path to SD v1.5
TEXTDIFFUSER2_DIFFUSION_MODEL_PATH = "/path/to/your/textdiffuser2/inpainting_model" # IMPORTANT: Use a model fine-tuned for inpainting
CALLIGRAPHER_PATH_DICT_JSON = "../path_dict.json"
# ----------------------------------------------------------------

def parse_self_bench():
    """Parses the self_bench.txt file."""
    samples = []
    with open(SELF_BENCH_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                samples.append({
                    "id": parts[0],
                    "source": parts[1],
                    "ref": parts[2],
                    "prompt": parts[3]
                })
    return samples

def run_anytext_inference(samples, output_root):
    print("\n--- Running AnyText2 Inference ---")
    output_dir = os.path.join(output_root, "anytext")
    os.makedirs(output_dir, exist_ok=True)
    
    for sample in tqdm(samples, desc="AnyText2"):
        ref_path = os.path.join(BENCHMARK_DIR, sample["ref"])
        # AnyText needs text prompts wrapped in quotes
        text_prompt = f'"{sample["prompt"]}"'
        
        cmd = [
            "python", ANYTEXT_INFERENCE_SCRIPT,
            "--model_dir", ANYTEXT_MODEL_DIR,
            "--img_prompt", "a sign with text on it", # Generic prompt
            "--text_prompt", text_prompt,
            "--ref_path", ref_path,
            "--output_dir", os.path.join(output_dir, sample["id"]),
            "--seed", "42"
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

def run_textdiffuser2_inference(samples, output_root):
    print("\n--- Running TextDiffuser2 Inference ---")
    output_dir = os.path.join(output_root, "textdiffuser2")
    os.makedirs(output_dir, exist_ok=True)

    if "your/path" in TEXTDIFFUSER2_LAYOUT_PLANNER_PATH:
        print("!!! WARNING: TextDiffuser2 paths are not configured. Skipping inference.")
        return

    for sample in tqdm(samples, desc="TextDiffuser2 Inpainting"):
        source_path = os.path.join(BENCHMARK_DIR, sample["source"])
        ref_path = os.path.join(BENCHMARK_DIR, sample["ref"])
        mask_path = os.path.join(BENCHMARK_DIR, sample["ref"]) # Assuming mask is the same as ref for inpainting
        
        cmd = [
            "python", TEXTDIFFUSER2_INFERENCE_SCRIPT,
            "--mode", "inpaint",
            "--base_model_path", TEXTDIFFUSER2_BASE_MODEL_PATH,
            "--diffusion_model_path", TEXTDIFFUSER2_DIFFUSION_MODEL_PATH,
            "--prompt", sample["prompt"], # For inpainting, prompt is just the text to render
            "--source_path", source_path,
            "--ref_path", ref_path,
            "--mask_path", mask_path,
            "--output_path", os.path.join(output_dir, f'{sample["id"]}_{sample["prompt"].replace(" ","_")}.png'),
            "--seed", "42"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running TextDiffuser2 inpainting for sample {sample['id']}: {e}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")


def run_calligrapher_inference(samples, output_root):
    print("\n--- Running Calligrapher Inference ---")
    output_dir = os.path.join(output_root, "calligrapher")
    os.makedirs(output_dir, exist_ok=True)

    if "your/path" in CALLIGRAPHER_MODEL_PATH:
        print("!!! WARNING: Calligrapher model path is not configured. Skipping inference.")
        return
        
    for sample in tqdm(samples, desc="Calligrapher"):
        source_path = os.path.join(BENCHMARK_DIR, sample["source"])
        ref_path = os.path.join(BENCHMARK_DIR, sample["ref"])
        
        # This is a placeholder command. You'll need to adapt it to how
        # `infer_calligrapher_self_custom.py` actually works.
        cmd = [
            "python", CALLIGRAPHER_INFERENCE_SCRIPT,
            "--model_path", CALLIGRAPHER_MODEL_PATH,
            "--source_image", source_path,
            "--ref_image", ref_path,
            "--prompt", sample["prompt"],
            "--output_dir", os.path.join(output_dir, sample["id"]),
            "--seed", "42"
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_evaluation(results_root):
    print("\n--- Running Evaluations ---")
    # This is a placeholder for running the actual eval scripts.
    # You would loop through each model's output directory and run
    # eval_ocr.py, eval_vlm.py, etc. on the generated images.
    
    eval_scripts = ["eval_ocr.py", "eval_vlm.py", "eval_fid.py", "eval_dino.py"]
    models = ["calligrapher", "anytext", "textdiffuser2"]
    
    for model in models:
        model_output_dir = os.path.join(results_root, model)
        if not os.path.exists(model_output_dir):
            print(f"No results found for {model}, skipping evaluation.")
            continue
            
        print(f"\n--- Evaluating {model} ---")
        for script in eval_scripts:
            print(f"Running {script} for {model}...")
            # Example command structure, will need to be adapted
            # based on how each evaluation script takes input.
            cmd = [
                "python", script,
                "--generated_images_dir", model_output_dir,
                "--benchmark_dir", BENCHMARK_DIR
                # Potentially other args like --output_report etc.
            ]
            # result = subprocess.run(cmd, capture_output=True, text=True)
            # print(result.stdout)
            # print(result.stderr)
            print(f"TODO: Implement the actual call to {script} with correct arguments.")


def main():
    parser = argparse.ArgumentParser(description="Run inference and evaluation for multiple text-to-image models.")
    parser.add_argument("--models", type=str, nargs='+', default=["all"], 
                        choices=["calligrapher", "anytext", "textdiffuser2", "all"],
                        help="Which model(s) to run inference for.")
    parser.add_argument("--skip_inference", action="store_true", help="Skip the inference step and only run evaluation.")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip the evaluation step and only run inference.")
    parser.add_argument("--output_root", type=str, default="../evaluation_results", help="Root directory to save all generated images and reports.")
    
    args = parser.parse_args()

    # Expand "all"
    models_to_run = args.models
    if "all" in models_to_run:
        models_to_run = ["calligrapher", "anytext", "textdiffuser2"]

    samples = parse_self_bench()
    
    if not args.skip_inference:
        print("="*50)
        print("STEP 1: RUNNING INFERENCE")
        print("="*50)
        if "anytext" in models_to_run:
            run_anytext_inference(samples, args.output_root)
        if "textdiffuser2" in models_to_run:
            run_textdiffuser2_inference(samples, args.output_root)
        if "calligrapher" in models_to_run:
            run_calligrapher_inference(samples, args.output_root)
    else:
        print("Skipping inference step as requested.")

    if not args.skip_evaluation:
        print("\n" + "="*50)
        print("STEP 2: RUNNING EVALUATION")
        print("="*50)
        run_evaluation(args.output_root)
    else:
        print("Skipping evaluation step as requested.")
        
    print("\nAll tasks complete.")

if __name__ == "__main__":
    main()
